"""Lifecycle recovery uses durable intent and native immutable Memory versions."""

from __future__ import annotations

import asyncio
import sqlite3

import pytest
from pydantic import ValidationError

from powercontext.builtin.artifacts.memory import MemoryEntryInput
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import (
    BuiltinConfig,
    CaptureSource,
    GetMemoryEntryRequest,
    RememberMemoryRequest,
    open_builtin_runtime,
)
from powercontext.builtin.runtime.source_lifecycle import SourceLifecycleError, SourceLifecycleEvent
from powercontext.builtin.scope import ScopeDraft


@pytest.mark.parametrize(
    "field,value",
    [(field, value) for field in ("revision", "sequence") for value in (True, "1", 1.0, 0, -1)]
    + [("available", value) for value in (1, "false", None)]
    + [(field, value) for field in ("node_id", "reason") for value in ("", " \t", 1)]
    + [("extra", "invalid")],
)
def test_event_does_not_coerce_invalid_fields(field, value):
    data = {"node_id": "source", "revision": 3, "sequence": 1, "available": False, "reason": "withdrawn"}
    data[field] = value
    with pytest.raises(ValidationError):
        SourceLifecycleEvent(**data)


def test_partial_native_write_failure_recovers_after_restart_without_duplicate_revisions(tmp_path):
    async def scenario():
        path = tmp_path / "recovery.db"
        config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{path}"))
        event = SourceLifecycleEvent(node_id="source", revision=3, sequence=1, available=False, reason="withdrawn")
        citations = []
        async with open_builtin_runtime(config) as runtime:
            scope = (
                await runtime.scopes.create(
                    ScopeDraft(title="Recovery", summary="Fault recovery", idempotency_key="recovery")
                )
            ).scope_id
            service = runtime.lifecycle.for_scope(scope)
            source = await runtime.sources.for_scope(scope).capture(
                CaptureSource(source_id="origin", content="Source evidence", metadata={})
            )
            await service.register_source("source", source.source_ref, revision=3)
            for number in range(3):
                memory = await runtime.memory.for_scope(scope).remember(
                    RememberMemoryRequest(entries=(MemoryEntryInput(kind="fact", text=f"Fact {number}"),))
                )
                citations.append(memory.entry.citation)
                await service.register_memory(f"memory-{number}", memory.entry.citation, parents=("source",))
            head = (await runtime.memory.for_scope(scope).list()).memory_ref
            with sqlite3.connect(path) as database:
                database.execute(
                    "CREATE TRIGGER fail_lifecycle_write BEFORE INSERT ON pc_artifacts "
                    "WHEN NEW.family = 'memory' AND NEW.revision = "
                    f"{head.revision + 2} BEGIN SELECT RAISE(ABORT, 'injected lifecycle write failure'); END"
                )
            with pytest.raises(Exception, match="injected lifecycle write failure"):
                await service.apply(event)
            entries = await runtime.memory.for_scope(scope).list(include_inactive=True)
            assert sum(record.state == "inactive" for record in entries.entries) == 1
            assert entries.memory_ref.revision == head.revision + 1
        with sqlite3.connect(path) as database:
            database.execute("DROP TRIGGER fail_lifecycle_write")
        async with open_builtin_runtime(config) as runtime:
            service = runtime.lifecycle.for_scope(scope)
            assert await service.apply(event) == []
            entries = await runtime.memory.for_scope(scope).list(include_inactive=True)
            assert all(record.state == "inactive" for record in entries.entries)
            assert entries.memory_ref.revision == head.revision + 3
            assert await service.apply(event) == []
            assert (await runtime.memory.for_scope(scope).list()).memory_ref == entries.memory_ref
            restored = event.model_copy(update={"sequence": 2, "available": True})
            assert await service.apply(restored) == ["memory-0", "memory-1", "memory-2", "source"]
            for number, citation in enumerate(citations):
                assert await service.resolve(f"memory-{number}") == citation
                entry = await runtime.memory.for_scope(scope).get(GetMemoryEntryRequest(citation=citation))
                assert entry.entry.text == f"Fact {number}"

    asyncio.run(scenario())


def test_scope_isolation_and_invalid_reference_do_not_pollute_registered_graph():
    async def scenario():
        async with open_builtin_runtime(BuiltinConfig(database=SQLiteConfig())) as runtime:
            scopes = []
            for title in ("First", "Second"):
                scope = (
                    await runtime.scopes.create(
                        ScopeDraft(title=title, summary="Isolated scope", idempotency_key=title)
                    )
                ).scope_id
                source = await runtime.sources.for_scope(scope).capture(
                    CaptureSource(source_id="origin", content=title, metadata={})
                )
                service = runtime.lifecycle.for_scope(scope)
                await service.register_source("source", source.source_ref, revision=3)
                memory = await runtime.memory.for_scope(scope).remember(
                    RememberMemoryRequest(entries=(MemoryEntryInput(kind="fact", text=title),))
                )
                await service.register_memory("memory", memory.entry.citation, parents=("source",))
                scopes.append((scope, service, memory.entry.citation))
            first, second = scopes
            with pytest.raises(SourceLifecycleError):
                await first[1].register_memory("foreign", second[2], parents=("source",))
            event = SourceLifecycleEvent(node_id="source", revision=3, sequence=1, available=False, reason="withdrawn")
            assert await first[1].apply(event) == []
            assert await second[1].resolve("memory") == second[2]
            assert await first[1].resolve("memory", purpose="audit") == first[2]
            with pytest.raises(SourceLifecycleError):
                await first[1].resolve("memory")
            assert await first[1].apply(event.model_copy(update={"sequence": 2, "available": True})) == [
                "memory",
                "source",
            ]

    asyncio.run(scenario())
