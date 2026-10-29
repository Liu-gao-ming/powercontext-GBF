"""Real scoped Memory retrieval of financial evidence across documents."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from powercontext.builtin.artifacts.memory import MemoryEntryInput
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import (
    BuiltinConfig,
    GetMemoryEntryRequest,
    RememberMemoryRequest,
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
from powercontext.sources.pdf_pages import (
    PDF_PAGE_SOURCE_DEFINITION,
    PDFDocumentInput,
    parse_pdf_document,
    search_pdf_pages,
)
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY
from tests.pdf_support import make_pdf

pytest.importorskip("pypdf")


def test_scoped_financial_recall_distinct_pages_and_restart_citations(tmp_path: Path) -> None:  # noqa: C901 - one ingestion/restart/search lifecycle
    config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'pdf.db'}"))
    registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
    texts = (
        "Consolidated Statement of Income\nNet revenue 950\nOperating income 120\nNet income 80\nPrinted page 47",
        "Consolidated Statement of Cash Flows\nDepreciation and amortization 20\nCapital expenditures 15\nPrinted page 49",
        "Legal proceedings\nPending litigation concerning a distribution agreement. Printed page 99",
        *(f"Pension plan disclosures\nBenefit obligation assets interest rates section {index}" for index in range(30)),
    )
    pages = parse_pdf_document(PDFDocumentInput(document_id="ACME_2024_10K", content=make_pdf(texts)))
    other_pages = parse_pdf_document(PDFDocumentInput(document_id="OTHER_2024_10K", content=make_pdf(texts[:3])))

    class Pipeline:
        async def extract(self, request, /) -> tuple[MemoryEntryInput, ...]:
            entries = []
            for source in request.sources:
                evidence = TextEvidence.model_validate(source.projection(TEXT_EVIDENCE_PROJECTION_KEY))
                entries.append(MemoryEntryInput(kind="pdf-evidence", text=evidence.content, sources=(source,)))
                if evidence.metadata["page_number"] == 1:
                    entries.append(MemoryEntryInput(kind="alternate-index", text=evidence.content, sources=(source,)))
            return tuple(entries)

    async def exercise() -> None:
        observations = tuple([
            project_source_for_transport(registry, await registry.resolve(page)) for page in (*pages, *other_pages)
        ])
        async with open_builtin_runtime(config) as runtime:
            assert runtime.scopes is not None
            scope = await runtime.scopes.create(
                ScopeDraft(title="Public reports", summary="Financial pages", idempotency_key="reports")
            )
            private = await runtime.scopes.create(
                ScopeDraft(title="Other scope", summary="Separate evidence", idempotency_key="other")
            )
            await runtime.ingestion.register(manifest_for_definition(PDF_PAGE_SOURCE_DEFINITION))
            for observation in (*observations, observations[0]):
                await runtime.ingestion.submit(
                    SubmitSourceObservation(scope_id=scope.scope_id, observation=observation)
                )
            await runtime.ingestion.submit(
                SubmitSourceObservation(scope_id=private.scope_id, observation=observations[0])
            )

        async with open_builtin_runtime(config, candidate_pipeline=Pipeline()) as runtime:
            memory = runtime.memory.for_scope(scope.scope_id)
            for _ in range(len(observations) + 1):
                if not (await memory.flush()).processed:
                    break
            # A second Memory representation of one physical page must not
            # consume another citation slot. An unrelated entry must be ignored.
            await memory.remember(
                RememberMemoryRequest(
                    entries=(
                        MemoryEntryInput(
                            kind="unrelated", text="ACME EBITDA income cash flow legal proceedings litigation " * 10
                        ),
                    )
                )
            )

        async with open_builtin_contexts(config) as contexts:
            stored = await (await contexts.get(scope.scope_id)).sources.journal.entries()
            assert len(stored) == len(observations)
            by_id = {}
            for item in stored:
                assert isinstance(item.source, SourceObservation)
                by_id[item.source.name] = item.source

        async with open_builtin_runtime(config) as runtime:
            memory = runtime.memory.for_scope(scope.scope_id)
            citations = await search_pdf_pages(
                memory,
                "What is ACME FY2024 EBITDA less capex using operating income, depreciation and amortization?",
                limit=20,
            )
            assert 2 <= len(citations) <= 20
            seen = set()
            locations = set()
            for citation in citations:
                entry = (await memory.get(GetMemoryEntryRequest(citation=citation))).entry
                assert len(entry.sources) == 1
                source_id = entry.sources[0].source_id
                assert source_id not in seen
                seen.add(source_id)
                evidence = TextEvidence.model_validate(by_id[source_id].projection(TEXT_EVIDENCE_PROJECTION_KEY))
                locations.add((evidence.metadata["document_id"], evidence.metadata["page_number"]))
                assert entry.text == evidence.content.strip()
            assert {("ACME_2024_10K", 1), ("ACME_2024_10K", 2)} <= locations
            legal = await search_pdf_pages(memory, "ACME ongoing legal battles", limit=1)
            assert len(legal) == 1
            entry = (await memory.get(GetMemoryEntryRequest(citation=legal[0]))).entry
            evidence = TextEvidence.model_validate(
                by_id[entry.sources[0].source_id].projection(TEXT_EVIDENCE_PROJECTION_KEY)
            )
            assert evidence.metadata["document_id"] == "ACME_2024_10K" and evidence.metadata["page_number"] == 3
            assert await search_pdf_pages(memory, "anything", limit=0) == ()
            assert await search_pdf_pages(memory, "quasar astronomical telescope", limit=20) == ()
            assert (
                await search_pdf_pages(runtime.memory.for_scope(private.scope_id), "ACME operating income", limit=20)
                == ()
            )

        async with open_builtin_runtime(config) as runtime:
            for citation in citations:
                assert (
                    await runtime.memory.for_scope(scope.scope_id).get(GetMemoryEntryRequest(citation=citation))
                ).entry.sources

    asyncio.run(exercise())
