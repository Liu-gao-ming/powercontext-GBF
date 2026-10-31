"""HTTP scores select original Memory identities across complete batches."""

from __future__ import annotations

import asyncio
import json
import math

import httpx
import pytest

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory.http_reranker import HTTPMemoryReranker
from powercontext.builtin.artifacts.memory.models import MemoryHit


def _hits(count: int) -> tuple[MemoryHit, ...]:
    return tuple(
        MemoryHit(
            memory_ref=ArtifactRef(family="memory", artifact_id="evidence", revision=3),
            entry_id=f"entry-{rank}",
            entry_version_id=f"version-{rank}",
            text=f"证据 {rank % 5}",
            score=1 / rank,
            matched_by=("fts",),
        )
        for rank in range(1, count + 1)
    )


@pytest.mark.parametrize(("batch_size", "limit"), [(1, 1), (7, 10), (32, 173), (200, 173)])
async def test_complete_global_pool_with_unordered_responses_and_cross_batch_ties(batch_size, limit):
    candidates = _hits(173)
    original = tuple(hit.model_dump() for hit in candidates)
    scores = {rank: (rank % 11) - 5.5 for rank in range(1, len(candidates) + 1)}
    sent = []

    def handler(request):
        body = json.loads(request.content)
        assert body["query"] == "查找证据"
        assert 1 <= len(body["candidates"]) <= batch_size
        for item in body["candidates"]:
            assert item["text"] == candidates[item["rank"] - 1].text
        sent.extend(item["rank"] for item in body["candidates"])
        return httpx.Response(
            200,
            json={
                "scores": [
                    {"rank": item["rank"], "score": scores[item["rank"]]} for item in reversed(body["candidates"])
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        decision = await HTTPMemoryReranker("https://scorer.invalid", client=client, batch_size=batch_size).rerank(
            "查找证据", candidates, limit
        )
        assert not client.is_closed
    assert decision.selected_ranks == tuple(sorted(scores, key=lambda rank: (-scores[rank], rank))[:limit])
    assert sorted(sent) == list(range(1, 174))
    assert decision.usage.requests == math.ceil(len(candidates) / batch_size)
    assert not decision.used_fallback and decision.discarded_rank_count == 0
    assert tuple(hit.model_dump() for hit in candidates) == original


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {"scores": None},
        {"scores": {}},
        {"scores": []},
        {"scores": [{"rank": 1, "score": 0}], "extra": 1},
        {"scores": [None]},
        {"scores": [{"rank": 1}]},
        {"scores": [{"rank": 1, "score": 0, "text": "extra"}]},
        *({"scores": [{"rank": rank, "score": 0}]} for rank in [True, False, 1.0, "1", 0, -1, 2]),
        *(
            {"scores": [{"rank": 1, "score": score}]}
            for score in [True, None, "0.5", [], {}, float("nan"), float("inf"), -float("inf")]
        ),
    ],
)
async def test_response_validation_rejects_types_missing_fields_and_nonfinite_scores(payload):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=json.dumps(payload).encode()))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(ValueError):
            await HTTPMemoryReranker("https://scorer.invalid", client=client).rerank("q", _hits(1), 1)


@pytest.mark.parametrize("ranks", [[1, 1], [1, 3], [2], [1, 2, 3]])
async def test_each_batch_must_cover_only_its_own_ranks(ranks):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        body = json.loads(request.content)
        returned = [item["rank"] for item in body["candidates"]] if calls == 1 else ranks
        return httpx.Response(200, json={"scores": [{"rank": rank, "score": 1} for rank in returned]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError):
            await HTTPMemoryReranker("https://scorer.invalid", client=client, batch_size=2).rerank("q", _hits(4), 1)
    assert calls == 2


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5, "2"])
def test_invalid_batch_size_is_rejected(batch_size):
    with pytest.raises(ValueError):
        HTTPMemoryReranker("https://scorer.invalid", client=None, batch_size=batch_size)


@pytest.mark.parametrize(("count", "limit"), [(0, 1), (2, 0), (2, -1), (2, 3), (2, True), (2, 1.0), (2, "1")])
async def test_invalid_selection_bounds_do_not_send_requests(count, limit):
    def handler(request):
        pytest.fail("invalid bounds must fail before HTTP")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError):
            await HTTPMemoryReranker("https://scorer.invalid", client=client).rerank("q", _hits(count), limit)


@pytest.mark.parametrize("status", [302, 400, 429, 500])
async def test_http_status_errors_propagate_before_json_validation(status):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(status, text="not JSON"))
    ) as client:
        with pytest.raises(httpx.HTTPStatusError) as caught:
            await HTTPMemoryReranker("https://scorer.invalid", client=client).rerank("q", _hits(1), 1)
    assert caught.value.response.status_code == status


async def test_invalid_json_raises_value_error():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text="{broken"))
    ) as client:
        with pytest.raises(ValueError):
            await HTTPMemoryReranker("https://scorer.invalid", client=client).rerank("q", _hits(1), 1)


async def test_finite_integer_scores_preserve_order_without_float_overflow_or_rounding():
    values = [10**400, 10**400 + 1, 1.0, -(10**400)]

    def handler(request):
        return httpx.Response(200, json={"scores": [{"rank": i, "score": score} for i, score in enumerate(values, 1)]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await HTTPMemoryReranker("https://scorer.invalid", client=client).rerank("q", _hits(4), 4)
    assert result.selected_ranks == (2, 1, 3, 4)


@pytest.mark.parametrize("error_class", [httpx.ReadTimeout, httpx.ConnectError])
async def test_original_transport_exception_is_preserved(error_class):
    failure = error_class("offline")

    def handler(request):
        raise failure

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(error_class) as caught:
            await HTTPMemoryReranker("https://scorer.invalid", client=client).rerank("q", _hits(1), 1)
    assert caught.value is failure


async def test_shared_reranker_keeps_simultaneous_queries_and_usage_independent():
    async def handler(request):
        body = json.loads(request.content)
        await asyncio.sleep(0)
        direction = 1 if body["query"] == "ascending" else -1
        return httpx.Response(
            200,
            json={"scores": [{"rank": item["rank"], "score": direction * item["rank"]} for item in body["candidates"]]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        reranker = HTTPMemoryReranker("https://scorer.invalid", client=client, batch_size=2)
        ascending, descending = await asyncio.gather(
            reranker.rerank("ascending", _hits(5), 3), reranker.rerank("descending", _hits(3), 2)
        )
    assert ascending.selected_ranks == (5, 4, 3)
    assert descending.selected_ranks == (1, 2)
    assert (ascending.usage.requests, descending.usage.requests) == (3, 2)
