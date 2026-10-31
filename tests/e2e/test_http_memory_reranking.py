"""HTTP reranking retains exact persisted Memory entries across a restart."""

from __future__ import annotations

import json
import math

import httpx

from powercontext.builtin.artifacts.memory import MemoryCitation, MemoryEntryInput
from powercontext.builtin.artifacts.memory.http_reranker import HTTPMemoryReranker
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import BuiltinConfig, open_builtin_contexts


async def test_http_selection_reopens_exact_memory_versions_without_changing_entries(tmp_path):
    config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'rerank.db'}"))
    async with open_builtin_contexts(config) as contexts:
        service = (await contexts.get("http-reranking")).artifacts.memory
        memory = await service.remember(
            memory=None,
            entries=tuple(MemoryEntryInput(kind="fact", text=f"Project evidence {index}.") for index in range(6)),
            mode="append",
        )
        assert memory is not None
        before = await service.entries(memory)
        coarse = await service.search("project", memories=(memory,), limit=6, mode="fts")
        assert len(coarse.hits) == 6
    request_ranks = []

    def handler(request):
        body = json.loads(request.content)
        assert body["query"] == "project"
        request_ranks.extend(item["rank"] for item in body["candidates"])
        assert all(item["text"] == coarse.hits[item["rank"] - 1].text for item in body["candidates"])
        # Equal scores span request boundaries; later candidates are most useful.
        return httpx.Response(
            200,
            json={
                "scores": [{"rank": item["rank"], "score": item["rank"] // 2} for item in reversed(body["candidates"])]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        reranker = HTTPMemoryReranker("https://scorer.invalid", client=client, batch_size=2, policy_id="test-http")
        async with open_builtin_contexts(config, memory_reranker=reranker) as contexts:
            service = (await contexts.get("http-reranking")).artifacts.memory
            result = await service.search("project", memories=(memory,), limit=3, mode="fts")
            assert result.rerank is not None
            assert result.rerank.policy_id == "test-http"
            assert result.rerank.selected_ranks == (6, 4, 5)
            assert result.rerank.usage.requests == math.ceil(len(result.rerank.candidate_hits) / 2)
            assert result.hits == tuple(coarse.hits[rank - 1] for rank in (6, 4, 5))
            assert await service.entries(memory) == before
            citations = tuple(
                MemoryCitation(memory_ref=hit.memory_ref, entry_id=hit.entry_id, entry_version_id=hit.entry_version_id)
                for hit in result.hits
            )
    assert sorted(request_ranks) == list(range(1, 7))
    async with open_builtin_contexts(config) as contexts:
        service = (await contexts.get("http-reranking")).artifacts.memory
        for citation, hit in zip(citations, result.hits, strict=True):
            exact = await service.validate_citation(citation)
            assert (exact.entry_id, exact.entry_version_id, exact.text) == (
                hit.entry_id,
                hit.entry_version_id,
                hit.text,
            )
        assert await service.entries(memory) == before
