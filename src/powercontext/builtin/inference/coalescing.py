"""Cross-call embedding deduplication with cancellation-safe batches and an LRU."""

from __future__ import annotations

import asyncio
import math
from collections import OrderedDict
from dataclasses import dataclass

from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage
from powercontext.builtin.inference.protocols import EmbeddingModel


@dataclass(frozen=True, slots=True)
class _Embedded:
    vector: EmbeddingVector
    batch: object
    usage: InferenceUsage


class CoalescingEmbeddingModel:
    """Share pending work in one event loop; charge each caller's batches once."""

    def __init__(
        self, model: EmbeddingModel, *, max_batch_size: int = 64, coalesce_ms: float = 5.0, cache_size: int = 1024
    ) -> None:
        if type(max_batch_size) is not int or max_batch_size <= 0:
            raise ValueError("max_batch_size must be a positive integer")  # noqa: TRY003
        if type(cache_size) is not int or cache_size < 0:
            raise ValueError("cache_size must be a non-negative integer")  # noqa: TRY003
        if type(coalesce_ms) not in (int, float) or not math.isfinite(coalesce_ms) or coalesce_ms < 0:
            raise ValueError("coalesce_ms must be finite and non-negative")  # noqa: TRY003
        self.profile = model.profile
        self._model = model
        self._max_batch = max_batch_size
        self._delay = coalesce_ms / 1000
        self._capacity = cache_size
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        self._pending: dict[str, asyncio.Future[_Embedded]] = {}
        self._queued: dict[str, asyncio.Future[_Embedded]] = {}
        self._flush_task: asyncio.Task[None] | None = None
        self._batches: set[asyncio.Task[None]] = set()

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        vectors: dict[str, EmbeddingVector] = {}
        waiting: dict[str, asyncio.Future[_Embedded]] = {}
        loop = asyncio.get_running_loop()
        for text in texts:
            if text in vectors:
                self._cache.move_to_end(text)
                continue
            if text in waiting:
                continue
            if text in self._cache:
                vectors[text] = self._cache[text]
                self._cache.move_to_end(text)
                continue
            future = self._pending.get(text)
            if future is None:
                future = loop.create_future()
                # A cancelled last waiter must not leave an unobserved exception.
                future.add_done_callback(_observe_failure)
                self._pending[text] = future
                self._queued[text] = future
            waiting[text] = future
        if self._queued and self._flush_task is None:
            self._flush_task = asyncio.create_task(self._flush())
        results = await asyncio.gather(*(asyncio.shield(future) for future in waiting.values()))
        used: dict[object, InferenceUsage] = {}
        for text, result in zip(waiting, results, strict=True):
            vectors[text] = result.vector
            used[result.batch] = result.usage
        usage = InferenceUsage(
            requests=sum(item.requests for item in used.values()),
            input_tokens=_tokens(tuple(item.input_tokens for item in used.values())),
            output_tokens=_tokens(tuple(item.output_tokens for item in used.values())),
        )
        return EmbeddingResult(vectors=tuple(vectors[text] for text in texts), usage=usage)

    async def _flush(self) -> None:
        try:
            await asyncio.sleep(self._delay)
            queued, self._queued = self._queued, {}
            items = tuple(queued.items())
            for start in range(0, len(items), self._max_batch):
                task = asyncio.create_task(self._run_batch(items[start : start + self._max_batch]))
                self._batches.add(task)
                task.add_done_callback(self._batches.discard)
        finally:
            self._flush_task = None

    async def _run_batch(self, items: tuple[tuple[str, asyncio.Future[_Embedded]], ...]) -> None:
        try:
            result = await self._model.embed(tuple(text for text, _ in items))
            if len(result.vectors) != len(items):
                raise ValueError("embedding vector count does not match its batch")  # noqa: TRY003, TRY301
        except asyncio.CancelledError:
            for text, future in items:
                self._pending.pop(text, None)
                future.cancel()
            raise
        except Exception as error:
            for text, future in items:
                self._pending.pop(text, None)
                future.set_exception(error)
        else:
            batch = object()
            for (text, future), vector in zip(items, result.vectors, strict=True):
                if self._capacity:
                    self._cache[text] = vector
                    self._cache.move_to_end(text)
                    while len(self._cache) > self._capacity:
                        self._cache.popitem(last=False)
                self._pending.pop(text, None)
                future.set_result(_Embedded(vector, batch, result.usage))


def _tokens(values: tuple[int | None, ...]) -> int | None:
    return None if not values or all(value is None for value in values) else sum(value or 0 for value in values)


def _observe_failure(future: asyncio.Future[_Embedded]) -> None:
    if not future.cancelled():
        future.exception()
