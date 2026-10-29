"""Pi checkpoints, failure recovery and large-event citations in real SQLite."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from powercontext.builtin.artifacts.memory import MemoryEntryInput
from powercontext.builtin.artifacts.memory.models import MemoryCitation
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import (
    BuiltinConfig,
    CommitConnectorCheckpoint,
    GetMemoryEntryRequest,
    SearchMemoryRequest,
    SubmitSourceObservation,
    open_builtin_contexts,
    open_builtin_runtime,
)
from powercontext.builtin.scope import ScopeDraft
from powercontext.sources import (
    ConnectorBinding,
    SourceDefinitionRegistry,
    SourceObservation,
    TextEvidence,
    manifest_for_definition,
    project_source_for_transport,
)
from powercontext.sources.connectors import ConnectorLifecycle
from powercontext.sources.pi_session import (
    PI_EVENT_SOURCE_DEFINITION,
    PiEventInput,
    PiSessionConnector,
    PiSessionInput,
    parse_pi_session,
)
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY


def test_branches_failure_repair_restart_and_large_tool_citation(tmp_path: Path) -> None:  # noqa: C901 - one recovery/citation lifecycle
    path = tmp_path / "session.jsonl"
    config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'pi.db'}"))
    registry = SourceDefinitionRegistry((PI_EVENT_SOURCE_DEFINITION,))
    large_text = "ToolAnchor 完整结果 " * 2000
    records: list[dict[str, Any]] = [
        {"type": "message", "id": "root", "parentId": None, "message": {"role": "user", "content": "RootAnchor"}},
        {
            "type": "message",
            "id": "left",
            "parentId": "root",
            "message": {"role": "assistant", "content": "LeftAnchor"},
        },
        {
            "type": "message",
            "id": "right",
            "parentId": "root",
            "message": {"role": "toolResult", "toolName": "read", "toolCallId": "call-1", "content": large_text},
            "extra": {"retain": [1, "中文"]},
        },
    ]

    def write() -> None:
        header = {"type": "session", "version": 3, "id": "branches"}
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in [header, *records]) + "\n", encoding="utf-8"
        )

    class Sink:
        def __init__(self, runtime, fail_event):
            self.runtime, self.fail_event = runtime, fail_event

        async def submit(self, binding, item_id, definition_name, value):
            assert isinstance(value, PiEventInput)
            if value.event_id == self.fail_event:
                raise RuntimeError("injected durable submission failure")  # noqa: TRY003
            observation = project_source_for_transport(registry, await registry.resolve(value))
            return (
                await self.runtime.ingestion.submit(
                    SubmitSourceObservation(scope_id=binding.scope_id, observation=observation)
                )
            ).source_ref

    class Checkpoints:
        def __init__(self, runtime):
            self.runtime = runtime

        async def load(self, binding):
            return (await self.runtime.ingestion.checkpoint(binding)).checkpoint

        async def save(self, binding, checkpoint, *, expected):
            await self.runtime.ingestion.commit(
                CommitConnectorCheckpoint(binding=binding, checkpoint=checkpoint, expected=expected)
            )

    class Pipeline:
        async def extract(self, request, /):
            entries = []
            for source in request.sources:
                text = source.projection(TEXT_EVIDENCE_PROJECTION_KEY)["content"]
                for start in range(0, len(text), 1500):
                    entries.append(
                        MemoryEntryInput(kind="pi-event", text=text[start : start + 1500], sources=(source,))
                    )
            return tuple(entries)

    async def run(leaf, fail_event=None):
        async with open_builtin_runtime(config) as runtime:
            assert runtime.scopes is not None
            scope = await runtime.scopes.create(
                ScopeDraft(title="Pi", summary="Complete evidence", idempotency_key="pi")
            )
            binding = ConnectorBinding(
                scope_id=scope.scope_id, binding_id="pi", connector_name="pi-session-history", connector_version="1"
            )
            await runtime.ingestion.register(manifest_for_definition(PI_EVENT_SOURCE_DEFINITION))
            result = await ConnectorLifecycle(sink=Sink(runtime, fail_event), checkpoints=Checkpoints(runtime)).run(
                PiSessionConnector(path, leaf), binding
            )
        return scope.scope_id, result

    async def exercise() -> None:
        write()
        scope_id, first = await run("left")
        assert len(first.items) == 2
        records[1]["message"]["content"] = "changed off-branch"
        write()
        with pytest.raises(ValueError):
            await run("right")
        records[1]["message"]["content"] = "LeftAnchor"
        write()
        _, switched = await run("right")
        assert switched.previous_checkpoint == first.committed_checkpoint
        assert len(switched.items) == 1
        # A full-record change must fail even if the text projection is identical.
        records[2]["extra"]["retain"] = [2, "中文"]
        write()
        with pytest.raises(ValueError):
            await run("right")
        records[2]["extra"]["retain"] = [1, "中文"]
        records.extend([
            {"type": "custom_message", "id": "tail1", "parentId": "right", "content": "TailOneAnchor"},
            {"type": "custom_message", "id": "tail2", "parentId": "tail1", "content": "TailTwoAnchor"},
        ])
        write()
        with pytest.raises(RuntimeError, match="injected"):
            await run("tail2", fail_event="tail2")
        _, repaired = await run("tail2")
        assert repaired.previous_checkpoint == switched.committed_checkpoint
        assert len(repaired.items) == 2
        _, replay = await run("tail2")
        assert replay.items == ()

        async with open_builtin_contexts(config) as contexts:
            entries = await (await contexts.get(scope_id)).sources.journal.entries()
            assert len(entries) == 5  # including the off-branch historical event
            stored = {}
            for entry in entries:
                assert isinstance(entry.source, SourceObservation)
                stored[entry.source.name] = entry.source
            events = parse_pi_session(PiSessionInput(content=path.read_bytes()))
            tool = next(event for event in events if event.event_id == "right")
            source = stored[(await registry.resolve(tool)).name]
            assert source.payload["record"] == records[2]
            evidence = TextEvidence.model_validate(source.projection(TEXT_EVIDENCE_PROJECTION_KEY))
            assert evidence.content == f"toolResult: [tool_result read call-1] {large_text}"

        async with open_builtin_runtime(config, candidate_pipeline=Pipeline()) as runtime:
            memory = runtime.memory.for_scope(scope_id)
            for _ in range(6):
                if not (await memory.flush()).processed:
                    break
            citations = []
            for query, event_id in [
                ("RootAnchor", "root"),
                ("LeftAnchor", "left"),
                ("ToolAnchor", "right"),
                ("TailOneAnchor", "tail1"),
                ("TailTwoAnchor", "tail2"),
            ]:
                page = await memory.search(SearchMemoryRequest(query=query, mode="fts", limit=100))
                assert page.hits
                hit = page.hits[0]
                citations.append((
                    event_id,
                    MemoryCitation(
                        memory_ref=hit.memory_ref, entry_id=hit.entry_id, entry_version_id=hit.entry_version_id
                    ),
                ))

        async with open_builtin_runtime(config) as runtime:
            for event_id, citation in citations:
                result = await runtime.memory.for_scope(scope_id).get(GetMemoryEntryRequest(citation=citation))
                for reference in result.entry.sources:
                    original = stored[reference.source_id]
                    assert original.payload["event_id"] == event_id
                    evidence = TextEvidence.model_validate(original.projection(TEXT_EVIDENCE_PROJECTION_KEY))
                    assert result.entry.text in evidence.content

    asyncio.run(exercise())
