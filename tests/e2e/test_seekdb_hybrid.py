"""Native seekdb hybrid budgets, filters, and same-database baseline quality."""

from __future__ import annotations

import math
import random
import re
from statistics import median
from time import perf_counter

from sqlalchemy import event

from powercontext.builtin.artifacts.memory import (
    EmbeddingProfile,
    Memory,
    MemoryEntryInput,
    MemorySearchRequest,
    MemoryService,
)
from powercontext.builtin.artifacts.memory.fusion import admit_fts_candidates, admit_vector_candidates, fuse_rankings
from powercontext.builtin.inference import EmbeddingResult
from powercontext.builtin.persistence.artifacts import ArtifactRepository
from powercontext.builtin.persistence.memory import RelationalMemoryBackend
from powercontext.builtin.persistence.memory_index import CompositeMemoryIndex
from powercontext.builtin.persistence.oceanbase.memory_index import OceanBaseMemoryFTSIndex, OceanBaseMemoryVectorIndex
from powercontext.builtin.persistence.seekdb import SeekDBConfig
from powercontext.builtin.persistence.seekdb.memory_index import SeekDBMemoryIndex
from powercontext.builtin.runtime import BuiltinConfig, open_builtin_contexts
from powercontext.builtin.tags import MemoryEntryTagTarget, TagFilter

PROFILE = EmbeddingProfile(
    profile_id="hybrid-regression-384", model="offline", dimension=384, distance="l2", normalization="unit"
)


class _Embedding:
    profile = PROFILE

    async def embed(self, texts):
        vectors = []
        for text in texts:
            number = int(re.search(r"record(\d+)", text).group(1))
            generator = random.Random(number)  # noqa: S311 - Deterministic offline test vectors.
            vector = [generator.uniform(-1, 1) for _ in range(PROFILE.dimension)]
            norm = math.sqrt(sum(value * value for value in vector))
            vectors.append(tuple(value / norm for value in vector))
        return EmbeddingResult(vectors=tuple(vectors))


async def test_native_hybrid_filters_before_limits_and_keeps_baseline_relevance(tmp_path):
    embedding = _Embedding()
    baseline = CompositeMemoryIndex(OceanBaseMemoryFTSIndex(), OceanBaseMemoryVectorIndex(PROFILE))
    unified = SeekDBMemoryIndex(PROFILE)
    timings = []
    async with open_builtin_contexts(
        BuiltinConfig(database=SeekDBConfig(path=tmp_path / "seekdb")), embedding_model=embedding
    ) as contexts:
        memory = (await contexts.get("regression-scope")).artifacts.memory
        head = await memory.remember(
            memory=None,
            entries=tuple(
                MemoryEntryInput(kind="fact", text=f"Quartz spectrometer record{number:03d} calibration evidence.")
                for number in range(42)
            ),
            mode="append",
        )
        other_service = MemoryService(
            backend=RelationalMemoryBackend(
                database=contexts.database,
                scope_id="regression-scope",
                artifacts=ArtifactRepository((Memory,)),
                index=unified,
            ),
            embedding_model=embedding,
        )
        other = await other_service.remember(
            memory=None,
            entries=(MemoryEntryInput(kind="fact", text="Other artifact quartz record007 calibration."),),
            mode="append",
        )
        foreign = (await contexts.get("foreign-scope")).artifacts.memory
        await foreign.remember(
            memory=None,
            entries=(MemoryEntryInput(kind="fact", text="Foreign scope quartz record007 calibration."),),
            mode="append",
        )
        target = next(entry for entry in await memory.entries(head) if "record007" in entry.text)
        other_target = (await memory.entries(other))[0]
        for artifact, entry, labels in (
            (head, target, ("selected", "verified")),
            (other, other_target, ("selected",)),
        ):
            tag_target = MemoryEntryTagTarget(artifact_id=artifact.artifact_id, entry_id=entry.entry_id)
            old = await contexts.records.get_tags("regression-scope", tag_target)
            await contexts.records.replace_tags("regression-scope", tag_target, labels, expected_etag=old.etag)
        for number in range(8):
            query = f"quartz record{number:03d}"
            request = MemorySearchRequest(
                query=query,
                analyzed_query=query,
                memories=(head.as_ref(),),
                candidate_limit=20,
                mode="hybrid",
                query_vector=(await embedding.embed((query,))).vectors[0],
                embedding_profile=PROFILE,
            )
            async with contexts.database.transaction() as connection:
                old = await baseline.search(connection, "regression-scope", request)
                new = await unified.search(connection, "regression-scope", request)
            old_hits = fuse_rankings(
                fts=admit_fts_candidates(query, old.fts),
                vector=admit_vector_candidates(old.vector),
                limit=10,
            )
            new_hits = fuse_rankings(
                fts=admit_fts_candidates(query, new.fts),
                vector=admit_vector_candidates(new.vector),
                limit=10,
            )
            expected = f"record{number:03d}"
            assert expected in old_hits[0].text and expected in new_hits[0].text
            assert all(hit.memory_ref == head.as_ref() for hit in new_hits)
            started = perf_counter()
            page = await memory.search(query, memories=(head,), mode="hybrid", limit=10)
            timings.append((perf_counter() - started) * 1000)
            assert expected in page.hits[0].text
        for match, expected in (("all", {target.entry_id}), ("any", {target.entry_id, other_target.entry_id})):
            statements = []

            def record(_connection, _cursor, statement, parameters, _context, _executemany, captured=statements):
                if "AGAINST" in statement.upper() or "l2_distance" in statement:
                    captured.append((statement, parameters))

            event.listen(contexts.database.engine.sync_engine, "before_cursor_execute", record)
            try:
                page = await memory.search(
                    "quartz record007",
                    memories=(head, other),
                    mode="hybrid",
                    limit=10,
                    tag_filter=TagFilter(tags=("selected", "verified"), match=match),
                )
            finally:
                event.remove(contexts.database.engine.sync_engine, "before_cursor_execute", record)
            assert {hit.entry_id for hit in page.hits} == expected
            assert len(statements) == 1
            statement, parameters = statements[0]
            async with contexts.database.transaction() as connection:
                plan = "\n".join(
                    str(row[0]) for row in (await connection.exec_driver_sql(f"EXPLAIN {statement}", parameters)).all()
                )
            assert "TEXT RETRIEVAL SCAN" in plan and "VECTOR INDEX" in plan
    print(f"Synthetic 384-dimensional native search median: {median(timings):.3f} ms")
