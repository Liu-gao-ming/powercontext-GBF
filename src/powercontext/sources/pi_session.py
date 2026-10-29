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

"""Offline Pi v3 branch evidence and lifecycle-owned incremental checkpoints."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from powercontext.sources.connectors import (
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
    ConnectorSubmissionStatus,
)
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence


class PiSessionInput(BaseModel):
    """Complete UTF-8 JSONL archive and an optional selected graph node."""

    content: bytes
    leaf_id: str | None = None


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _visible_blocks(value: JsonValue, *, tool_calls: bool = False) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for block in value:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(text := block.get("text"), str):
            parts.append(text)
        elif tool_calls and block.get("type") == "toolCall":
            name, call_id = block.get("name"), block.get("id")
            if not isinstance(name, str) or not isinstance(call_id, str):
                raise ValueError("Pi tool calls must declare a string name and id")  # noqa: TRY003
            parts.append(f"[tool_call {name} {call_id}] {_compact_json(block.get('arguments', {}))}")
    return "\n".join(parts)


def _event_text(record: dict[str, JsonValue]) -> tuple[str, str] | None:
    event_type = record.get("type")
    if event_type in {"compaction", "branch_summary"}:
        role = str(event_type)
        summary = record.get("summary", "")
        if not isinstance(summary, str):
            raise ValueError("Pi summaries must be strings")  # noqa: TRY003
        body = summary
    elif event_type == "custom_message":
        role = "custom"
        body = _visible_blocks(record.get("content"))
    elif event_type == "message":
        message = record.get("message")
        if not isinstance(message, dict) or not isinstance(role := message.get("role"), str) or not role:
            raise ValueError("Pi messages must declare a non-empty role")  # noqa: TRY003
        if role == "bashExecution":
            command, output = message.get("command", ""), message.get("output", "")
            if not isinstance(command, str) or not isinstance(output, str):
                raise ValueError("Pi bash command and output must be strings")  # noqa: TRY003
            body = f"$ {command}\n{output}"
        else:
            body = _visible_blocks(message.get("content"), tool_calls=True)
            if role == "toolResult" and body:
                tool_name, call_id = message.get("toolName", ""), message.get("toolCallId", "")
                if not isinstance(tool_name, str) or not isinstance(call_id, str):
                    raise ValueError("Pi tool results must declare string tool identifiers")  # noqa: TRY003
                body = f"[tool_result {tool_name} {call_id}] {body}"
    else:
        return None
    return (role, f"{role}: {body}") if body else None


class PiEventInput(BaseModel):
    """One visible event retaining the full original JSON record."""

    model_config = ConfigDict(frozen=True)

    session_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]

    @model_validator(mode="after")
    def validate_event(self) -> PiEventInput:
        if (
            self.record.get("id") != self.event_id
            or "parentId" not in self.record
            or self.record["parentId"] != self.parent_id
            or self.record.get("type") != self.event_type
            or _event_text(self.record) != (self.role, self.content)
        ):
            raise ValueError("Pi event fields must match its original visible record")  # noqa: TRY003
        return self


@dataclass(frozen=True)
class _ParsedSession:
    session_id: str
    records: dict[str, dict[str, JsonValue]]
    events: tuple[PiEventInput, ...]


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")  # noqa: TRY003


def _session_header(record: dict[str, JsonValue]) -> str:
    session_id = record.get("id")
    if (
        record.get("type") != "session"
        or type(record.get("version")) is not int
        or record["version"] != 3
        or not isinstance(session_id, str)
        or not session_id
    ):
        raise ValueError("Pi archives must begin with a version 3 session header")  # noqa: TRY003
    return session_id


def _validate_node(record: dict[str, JsonValue], earlier: dict[str, dict[str, JsonValue]]) -> str:
    event_id = record.get("id")
    if not isinstance(event_id, str) or not event_id or event_id in earlier:
        raise ValueError("Pi nodes must have unique non-empty string ids")  # noqa: TRY003
    if "parentId" not in record:
        raise ValueError("Pi nodes must declare parentId")  # noqa: TRY003
    parent = record["parentId"]
    if parent is not None and (not isinstance(parent, str) or (parent and parent not in earlier)):
        raise ValueError("Pi parents must refer to earlier nodes or be empty")  # noqa: TRY003
    event_type = record.get("type")
    if not isinstance(event_type, str) or not event_type or event_type == "session":
        raise ValueError("Pi nodes must declare an event type")  # noqa: TRY003
    return event_id


def _parse_session(value: PiSessionInput) -> _ParsedSession:
    records: dict[str, dict[str, JsonValue]] = {}
    visible: dict[str, PiEventInput] = {}
    session_id: str | None = None
    last_id: str | None = None
    # StringIO recognizes physical CR/LF lines without splitting valid Unicode
    # line separators inside JSON strings.
    for line in io.StringIO(value.content.decode("utf-8-sig"), newline=None):
        if not line.strip():
            continue
        record = json.loads(line, parse_constant=_reject_json_constant)
        if not isinstance(record, dict):
            raise ValueError("Pi JSONL records must be objects")  # noqa: TRY003, TRY004
        if session_id is None:
            session_id = _session_header(record)
            continue
        event_id = _validate_node(record, records)
        parent = record["parentId"]
        event_type = record.get("type")
        records[event_id] = record
        last_id = event_id
        text = _event_text(record)
        if text is not None:
            role, content = text
            visible[event_id] = PiEventInput(
                session_id=session_id,
                event_id=event_id,
                parent_id=parent,
                event_type=event_type,
                role=role,
                content=content,
                record=record,
            )
    if session_id is None:
        raise ValueError("Pi archives must contain a session header")  # noqa: TRY003
    leaf = last_id if value.leaf_id is None else value.leaf_id
    if value.leaf_id is not None and leaf not in records:
        raise ValueError("Pi selected leaf does not exist")  # noqa: TRY003
    path: list[PiEventInput] = []
    while leaf:
        if leaf in visible:
            path.append(visible[leaf])
        parent = records[leaf]["parentId"]
        leaf = parent if isinstance(parent, str) else None
    return _ParsedSession(session_id, records, tuple(reversed(path)))


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Validate an entire archive, then return visible ancestors root to leaf."""
    return _parse_session(value).events


def _event_identity(session_id: str, event_id: str) -> str:
    # Hash a framed pair rather than truncating or concatenating provider ids.
    # This also supports arbitrarily long ids within Source's 256-char bound.
    digest = hashlib.sha256(_compact_json([session_id, event_id]).encode("utf-8")).hexdigest()
    return f"pi:{digest}"


class PiEventSource(Source):
    """Captured original evidence, independent of Memory entry size limits."""

    model_config = ConfigDict(frozen=True)

    materialization: Literal[SourceMaterialization.CAPTURED] = SourceMaterialization.CAPTURED
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]

    @model_validator(mode="after")
    def validate_capture(self) -> PiEventSource:
        event = PiEventInput.model_validate(self.model_dump())
        if self.name != _event_identity(event.session_id, event.event_id):
            raise ValueError("Pi Source name must equal its event identity")  # noqa: TRY003
        return self


class _PiEventAdapter:
    name = "pi-event"
    input_class = PiEventInput
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        return PiEventSource(name=_event_identity(value.session_id, value.event_id), **value.model_dump())

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput.model_validate(source.model_dump())


class _PiTextProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type="pi-event",
            source_id=source.name,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


PI_EVENT_SOURCE_DEFINITION = AdapterSourceDefinition(_PiEventAdapter(), projections=(_PiTextProjection(),))


class _PiCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1] = 1
    session_id: str
    imported: dict[str, str] = Field(default_factory=dict)


class PiSessionConnector:
    """Discover selected events; let ConnectorLifecycle commit accepted progress."""

    name = "pi-session-history"
    version = "1"
    source_definitions = frozenset({"pi-event"})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self.path = Path(path)
        self.leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        content = await asyncio.to_thread(self.path.read_bytes)
        parsed = _parse_session(PiSessionInput(content=content, leaf_id=self.leaf_id))
        previous = (
            _PiCheckpoint(session_id=parsed.session_id)
            if session.checkpoint is None
            else _PiCheckpoint.model_validate(session.checkpoint)
        )
        if previous.session_id != parsed.session_id:
            raise ValueError("Pi checkpoint belongs to a different session")  # noqa: TRY003
        # Check all imported branches before submitting anything, including an
        # event that has since been changed into an invisible control record.
        digests = {
            event_id: hashlib.sha256(_compact_json(record).encode("utf-8")).hexdigest()
            for event_id, record in parsed.records.items()
        }
        if any(digests.get(event_id) != digest for event_id, digest in previous.imported.items()):
            raise ValueError("an imported Pi event was changed or removed")  # noqa: TRY003
        imported = dict(previous.imported)
        for event in parsed.events:
            if event.event_id in imported:
                continue
            outcome = await session.submit(_event_identity(event.session_id, event.event_id), "pi-event", event)
            if outcome.status is not ConnectorSubmissionStatus.ACCEPTED:
                return ConnectorRunCompletion(status=ConnectorRunStatus.INCOMPLETE, checkpoint=session.checkpoint)
            imported[event.event_id] = digests[event.event_id]
        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE,
            checkpoint=_PiCheckpoint(session_id=parsed.session_id, imported=imported).model_dump(mode="json"),
        )


__all__ = [
    "PI_EVENT_SOURCE_DEFINITION",
    "PiEventInput",
    "PiEventSource",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
]
