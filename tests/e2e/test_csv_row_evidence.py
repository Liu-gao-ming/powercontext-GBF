"""Exercise captured CSV evidence through SQLite ingestion and exact citations."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from powercontext.builtin.artifacts.memory import MemoryEntryInput
from powercontext.builtin.artifacts.memory.models import MemoryCitation
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import (
    BuiltinConfig,
    GetMemoryEntryRequest,
    SearchMemoryRequest,
    SubmitSourceObservation,
    open_builtin_contexts,
    open_builtin_runtime,
)
from powercontext.builtin.scope import ScopeDraft
from powercontext.sources import (
    SourceDefinitionRegistry,
    SourceObservation,
    TextEvidence,
    manifest_for_definition,
    project_source_for_transport,
)
from powercontext.sources.csv_rows import CSV_ROW_SOURCE_DEFINITION, CSVDocumentInput, parse_csv_document
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY


def test_ordered_unicode_rows_survive_restart_and_exact_citation(tmp_path: Path) -> None:
    rows = parse_csv_document(
        CSVDocumentInput(
            document_id="机场/%:记录",
            primary_key=" id ",
            content='z, id ,空值,note\r\n北京, 001 ,,"UniqueAirport first\r\nsecond ""quoted"""\r\n上海,001,,other\r\n'.encode(),
        )
    )
    registry = SourceDefinitionRegistry((CSV_ROW_SOURCE_DEFINITION,))
    config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'rows.db'}"))

    class RowPipeline:
        async def extract(self, request, /) -> tuple[MemoryEntryInput, ...]:
            return tuple(
                MemoryEntryInput(
                    kind="csv-row",
                    text=source.projection(TEXT_EVIDENCE_PROJECTION_KEY)["content"],
                    sources=(source,),
                )
                for source in request.sources
            )

    async def exercise() -> None:
        observations = tuple([project_source_for_transport(registry, await registry.resolve(row)) for row in rows])
        async with open_builtin_runtime(config) as runtime:
            assert runtime.scopes is not None
            scope = await runtime.scopes.create(
                ScopeDraft(title="CSV evidence", summary="Ordered original rows", idempotency_key="csv-evidence")
            )
            await runtime.ingestion.register(manifest_for_definition(CSV_ROW_SOURCE_DEFINITION))
            for observation in observations:
                await runtime.ingestion.submit(
                    SubmitSourceObservation(scope_id=scope.scope_id, observation=observation)
                )

        async with open_builtin_contexts(config) as contexts:
            entries = await (await contexts.get(scope.scope_id)).sources.journal.entries()
            by_id = {entry.source_ref.source_id: entry.source for entry in entries}
            assert len(entries) == len(rows)
            for observation, row in zip(observations, rows, strict=True):
                persisted = by_id[observation.name]
                assert isinstance(persisted, SourceObservation)
                evidence = TextEvidence.model_validate(persisted.projection(TEXT_EVIDENCE_PROJECTION_KEY))
                assert evidence == TextEvidence.model_validate(observation.projection(TEXT_EVIDENCE_PROJECTION_KEY))
                metadata_row = evidence.metadata["row"]
                assert isinstance(metadata_row, dict)
                assert list(metadata_row) == list(row.columns)
                assert list(json.loads(evidence.content)) == list(row.columns)
                assert tuple(json.loads(evidence.content).values()) == row.values

        async with open_builtin_runtime(config, candidate_pipeline=RowPipeline()) as runtime:
            # Replays in a different order must not create new journal records.
            for observation in reversed(observations):
                await runtime.ingestion.submit(
                    SubmitSourceObservation(scope_id=scope.scope_id, observation=observation)
                )
            memory = runtime.memory.for_scope(scope.scope_id)
            processed = 0
            for _ in range(4):
                result = await memory.flush()
                if not result.processed:
                    break
                processed += result.source_count
            assert processed == len(rows)
            search = await memory.search(SearchMemoryRequest(query="UniqueAirport", mode="fts", limit=10))
            assert len(search.hits) == 1
            hit = search.hits[0]
            citation = MemoryCitation(
                memory_ref=hit.memory_ref,
                entry_id=hit.entry_id,
                entry_version_id=hit.entry_version_id,
            )

        async with open_builtin_runtime(config) as runtime:
            entry = await runtime.memory.for_scope(scope.scope_id).get(GetMemoryEntryRequest(citation=citation))
            assert entry.entry.text == json.dumps(
                dict(zip(rows[0].columns, rows[0].values, strict=True)), ensure_ascii=False, separators=(",", ":")
            )
            assert tuple(source.source_id for source in entry.entry.sources) == (observations[0].name,)

    asyncio.run(exercise())
