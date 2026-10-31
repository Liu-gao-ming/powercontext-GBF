"""Native prepare and incremental transport retain literal, restartable citations."""

from __future__ import annotations

import json

from powercontext.builtin.artifacts.memory import MemoryCitation, MemoryEntryInput
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import (
    BuiltinConfig,
    GetMemoryEntryRequest,
    PrepareContextRequest,
    RememberMemoryRequest,
    open_builtin_runtime,
)
from powercontext.builtin.runtime.context_pack import ContextPack
from powercontext.builtin.scope import ScopeDraft


async def test_native_prepare_selection_survives_restart_and_serialized_transport(tmp_path):
    config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'evidence.db'}"))
    original = "Background administration. " * 80 + "Cobalt calibration requires 37 degrees."
    request = PrepareContextRequest(query="cobalt calibration", max_bytes=1200, evidence_selection=True)
    async with open_builtin_runtime(config) as runtime:
        scope = await runtime.scopes.create(
            ScopeDraft(title="Evidence", summary="Literal source offsets", idempotency_key="evidence")
        )
        result = await runtime.memory.for_scope(scope.scope_id).remember(
            RememberMemoryRequest(entries=(MemoryEntryInput(kind="fact", text=original),))
        )
        authorized = []

        async def authorize(scopes):
            authorized.append(scopes)

        context = runtime.context.for_scope(scope.scope_id)
        prepared = await context.prepare(request, authorize_scopes=authorize)
        assert authorized == [(scope.scope_id,)]
        assert prepared.content_bytes == len(prepared.content.encode()) <= request.max_bytes
        assert "Cobalt calibration requires 37 degrees." in prepared.content
        pack = await context.prepare_incremental(request)
        receiver = ContextPack.from_bytes(pack.to_bytes())
        assert b"".join(receiver.items) == prepared.content.encode()
    async with open_builtin_runtime(config) as runtime:
        restored = await runtime.context.for_scope(scope.scope_id).prepare(request)
        assert restored == prepared
        items = json.loads(restored.content.splitlines()[-2])["items"]
        for item in items:
            reference = dict(item["citation"])
            assert reference.pop("scope_id") == scope.scope_id
            citation = MemoryCitation.model_validate(reference)
            assert citation == result.entry.citation
            exact = await runtime.memory.for_scope(scope.scope_id).get(GetMemoryEntryRequest(citation=citation))
            assert exact.entry.text == original
            assert item["content"] == original[item["start"] : item["end"]]
