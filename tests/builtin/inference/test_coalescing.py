"""Cross-call sharing must preserve per-caller results, cancellation, and usage."""

from __future__ import annotations

import asyncio

import pytest

from powercontext.builtin.artifacts.memory import EmbeddingProfile
from powercontext.builtin.inference import EmbeddingResult, InferenceUsage
from powercontext.builtin.inference.coalescing import CoalescingEmbeddingModel


class _Model:
    profile = EmbeddingProfile(profile_id="coalesce-regression", model="offline", dimension=1)

    def __init__(self):
        self.calls = []

    async def embed(self, texts):
        self.calls.append(texts)
        return EmbeddingResult(
            vectors=tuple((float(len(text)),) for text in texts),
            usage=InferenceUsage(requests=3, input_tokens=7, output_tokens=0),
        )


async def test_late_call_joins_running_batch_without_cache_and_usage_counts_shared_batch_once():
    started = asyncio.Event()
    release = asyncio.Event()

    class Blocked(_Model):
        async def embed(self, texts):
            result = await super().embed(texts)
            if texts == ("alpha", "b"):
                started.set()
                await release.wait()
            return result

    base = Blocked()
    model = CoalescingEmbeddingModel(base, max_batch_size=2, coalesce_ms=0, cache_size=0)
    first = asyncio.create_task(model.embed(("alpha", "b", "alpha")))
    await started.wait()
    late = asyncio.create_task(model.embed(("b", "ccc", "alpha", "b")))
    await asyncio.sleep(0)
    release.set()
    a, b = await asyncio.gather(first, late)
    assert a.vectors == ((5.0,), (1.0,), (5.0,))
    assert b.vectors == ((1.0,), (3.0,), (5.0,), (1.0,))
    assert base.calls == [("alpha", "b"), ("ccc",)]
    assert (a.usage.requests, a.usage.input_tokens, a.usage.output_tokens) == (3, 7, 0)
    assert (b.usage.requests, b.usage.input_tokens, b.usage.output_tokens) == (6, 14, 0)
    assert model.profile is base.profile
    await model.embed(("alpha",))
    assert base.calls[-1] == ("alpha",)


async def test_failed_batch_does_not_poison_an_independent_batch_or_retry():
    failure = RuntimeError("controlled batch failure")

    class Flaky(_Model):
        failed = False

        async def embed(self, texts):
            result = await super().embed(texts)
            if "bad" in texts and not self.failed:
                self.failed = True
                raise failure
            return result

    base = Flaky()
    model = CoalescingEmbeddingModel(base, max_batch_size=1, coalesce_ms=0)
    broken, healthy = await asyncio.gather(model.embed(("bad",)), model.embed(("good",)), return_exceptions=True)
    assert broken is failure
    assert healthy.vectors == ((4.0,),)
    assert (await model.embed(("good",))).usage.requests == 0
    assert (await model.embed(("bad",))).vectors == ((3.0,),)
    assert base.calls == [("bad",), ("good",), ("bad",)]


async def test_cancelling_every_original_waiter_leaves_running_work_for_a_late_waiter():
    started, release = asyncio.Event(), asyncio.Event()

    class Blocked(_Model):
        async def embed(self, texts):
            result = await super().embed(texts)
            started.set()
            await release.wait()
            return result

    base = Blocked()
    model = CoalescingEmbeddingModel(base, coalesce_ms=0, cache_size=0)
    original = asyncio.create_task(model.embed(("shared",)))
    await started.wait()
    original.cancel()
    with pytest.raises(asyncio.CancelledError):
        await original
    late = asyncio.create_task(model.embed(("shared", "shared")))
    await asyncio.sleep(0)
    release.set()
    result = await late
    assert result.vectors == ((6.0,), (6.0,))
    assert result.usage.requests == 3
    assert base.calls == [("shared",)]


async def test_duplicate_cache_visits_update_lru_and_cache_eviction_does_not_change_waiting_results():
    base = _Model()
    model = CoalescingEmbeddingModel(base, coalesce_ms=0, cache_size=2)
    await model.embed(("a", "bb"))
    assert (await model.embed(("a", "bb", "a"))).usage.requests == 0
    await model.embed(("ccc",))
    assert (await model.embed(("a",))).usage.requests == 0
    await model.embed(("bb",))
    assert base.calls == [("a", "bb"), ("ccc",), ("bb",)]
    tiny = CoalescingEmbeddingModel(base, max_batch_size=2, coalesce_ms=0, cache_size=1)
    result = await tiny.embed(("one", "two", "three", "one"))
    assert result.vectors == ((3.0,), (3.0,), (5.0,), (3.0,))
    assert result.usage.requests == 6


@pytest.mark.parametrize(
    "options",
    [
        {"max_batch_size": 0},
        {"max_batch_size": True},
        {"max_batch_size": 1.5},
        {"cache_size": -1},
        {"cache_size": True},
        {"coalesce_ms": -1},
        {"coalesce_ms": float("nan")},
        {"coalesce_ms": float("inf")},
        {"coalesce_ms": True},
    ],
)
def test_invalid_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        CoalescingEmbeddingModel(_Model(), **options)
