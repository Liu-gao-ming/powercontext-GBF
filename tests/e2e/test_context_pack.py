"""Incremental prepare follows normal recall, authorization, and empty results."""

from __future__ import annotations

import pytest

from powercontext.builtin.artifacts.memory import MemoryEntryInput
from powercontext.builtin.runtime import (
    BuiltinConfig,
    CaptureSource,
    ContextAssembly,
    PrepareContextRequest,
    open_builtin_runtime,
)
from powercontext.builtin.runtime.context_pack import ContextPack, ContextPackDelta, apply_context_pack_delta
from powercontext.builtin.scope import ScopeDraft
from powercontext.builtin.sources import ContentSource


class _Pipeline:
    async def extract(self, request, /):
        return tuple(
            MemoryEntryInput(kind="fact", text=source.content, sources=(source,))
            for source in request.sources
            if isinstance(source, ContentSource)
        )


async def test_serialized_receiver_tracks_append_query_switch_and_empty_result():
    async with open_builtin_runtime(BuiltinConfig(), candidate_pipeline=_Pipeline()) as runtime:
        scope = await runtime.scopes.create(ScopeDraft(title="Pack", summary="Evidence", idempotency_key="pack"))
        context = runtime.context.for_scope(scope.scope_id)
        request = PrepareContextRequest(query="ocean", max_bytes=4096)
        empty = await context.prepare_incremental(request)
        assert isinstance(empty, ContextPack) and empty.items == ()
        receiver = ContextPack.from_bytes(empty.to_bytes())
        distinct = set()
        for source_id, text, query in [
            ("a", "ocean 证据 first", "ocean"),
            ("b", "mountain 第二项证据", "mountain"),
            ("c", "ocean appended evidence", "ocean"),
        ]:
            await runtime.sources.for_scope(scope.scope_id).capture(
                CaptureSource(source_id=source_id, content=text, metadata={})
            )
            while (await runtime.memory.for_scope(scope.scope_id).flush()).processed:
                pass
            request = PrepareContextRequest(query=query, max_bytes=4096)
            expected = await context.prepare(request)
            delta = await context.prepare_incremental(request, base_pack=receiver)
            assert isinstance(delta, ContextPackDelta)
            receiver = apply_context_pack_delta(receiver, ContextPackDelta.from_bytes(delta.to_bytes()))
            assert b"".join(receiver.items) == (expected.content or "").encode()
            assert expected.content
            distinct.add(expected.content)
        assert len(distinct) == 3
        request = PrepareContextRequest(query="ocean", assembly=ContextAssembly(sections=()))
        expected = await context.prepare(request)
        assert expected.content is None
        delta = await context.prepare_incremental(request, base_pack=receiver)
        receiver = apply_context_pack_delta(receiver, ContextPackDelta.from_bytes(delta.to_bytes()))
        assert receiver.items == ()
        assert b"".join(receiver.items) == b""


async def test_incremental_prepare_preserves_the_normal_scope_authorizer():
    async with open_builtin_runtime(BuiltinConfig()) as runtime:
        scope = await runtime.scopes.create(ScopeDraft(title="Pack", summary="Restricted", idempotency_key="auth"))
        context = runtime.context.for_scope(scope.scope_id)
        observed = []

        async def reject(scope_ids):
            observed.append(scope_ids)
            raise PermissionError("denied")

        with pytest.raises(PermissionError):
            await context.prepare_incremental(PrepareContextRequest(query="evidence"), authorize_scopes=reject)
        assert observed == [(scope.scope_id,)]
