# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Identity-preserving Memory reranking with a batched HTTP scoring service."""

from __future__ import annotations

import math

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage


class HTTPMemoryReranker:
    """Score every global rank, then select from the complete candidate pool.

    The caller owns the HTTP client and its timeout, transport, and retry
    configuration. HTTP failures propagate without a local-ranking fallback.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = "http-cross-encoder-v1",
    ) -> None:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")  # noqa: TRY003
        if not isinstance(policy_id, str) or not policy_id:
            raise ValueError("policy_id must be a non-empty string")  # noqa: TRY003
        self.policy_id = policy_id
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Return stable global ranks and the number of HTTP requests issued."""
        if not candidates:
            raise ValueError("rerank requires at least one candidate")  # noqa: TRY003
        if type(limit) is not int or not 1 <= limit <= len(candidates):
            raise ValueError("limit must be an integer within the candidate count")  # noqa: TRY003

        scores: dict[int, int | float] = {}
        requests = 0
        for start in range(0, len(candidates), self._batch_size):
            batch = candidates[start : start + self._batch_size]
            ranks = set(range(start + 1, start + len(batch) + 1))
            requests += 1
            response = await self._client.post(
                self._endpoint,
                json={
                    "query": query,
                    "candidates": [
                        {"rank": rank, "text": candidate.text} for rank, candidate in enumerate(batch, start=start + 1)
                    ],
                },
            )
            response.raise_for_status()
            scores.update(_batch_scores(response.json(), ranks))

        selected = sorted(scores, key=lambda rank: (-scores[rank], rank))[:limit]
        return MemoryRerankDecision(selected_ranks=tuple(selected), usage=InferenceUsage(requests=requests))


def _batch_scores(payload: object, expected: set[int]) -> dict[int, int | float]:
    """Reject incomplete, ambiguous, or cross-batch scoring responses."""
    if not isinstance(payload, dict) or set(payload) != {"scores"}:
        raise ValueError("rerank response must contain exactly scores")  # noqa: TRY003
    records = payload["scores"]
    if not isinstance(records, list) or len(records) != len(expected):
        raise ValueError("scores must cover exactly the requested batch")  # noqa: TRY003

    scores: dict[int, int | float] = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {"rank", "score"}:
            raise ValueError("each score must contain exactly rank and score")  # noqa: TRY003
        rank, score = record["rank"], record["score"]
        if type(rank) is not int or rank not in expected or rank in scores:
            raise ValueError("score ranks must cover the requested batch once")  # noqa: TRY003
        if type(score) not in (int, float):
            raise ValueError("score must be a finite number")  # noqa: TRY003
        if type(score) is float and not math.isfinite(score):
            raise ValueError("score must be a finite number")  # noqa: TRY003
        scores[rank] = score
    return scores


__all__ = ["HTTPMemoryReranker"]
