"""Pi archive parsing and evidence identity beyond the public workload."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from powercontext.sources import SourceDefinitionRegistry, SourceMaterialization, TextEvidence
from powercontext.sources.pi_session import PI_EVENT_SOURCE_DEFINITION, PiSessionInput, parse_pi_session
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY


def _archive(records: list[dict[str, Any]], *, session_id: str = "session") -> bytes:
    header = {"type": "session", "version": 3, "id": session_id}
    return "\n".join(json.dumps(row, ensure_ascii=False) for row in [header, *records]).encode()


def _message(event_id: str, parent: str | None, content: object) -> dict[str, Any]:
    return {"id": event_id, "parentId": parent, "type": "message", "message": {"role": "assistant", "content": content}}


def test_mixed_blocks_exact_text_and_control_ancestors() -> None:
    records = [
        _message("root", None, "  原文\u0085\u2028\u2029  "),
        _message("hidden", "root", [{"type": "thinking", "thinking": "secret"}, {"type": "image", "data": "x"}]),
        {"type": "model_change", "id": "control", "parentId": "hidden"},
        _message(
            "tool",
            "control",
            [
                {"type": "text", "text": "before"},
                {"type": "toolCall", "name": "read", "id": "call:一", "arguments": {"z": [1, False], "a": "中文"}},
                {"type": "text", "text": "after"},
            ],
        ),
        {"type": "compaction", "id": "summary", "parentId": "tool", "summary": "完整摘要"},
        {"type": "custom_message", "id": "custom", "parentId": "summary", "content": "  自定义  "},
    ]
    events = parse_pi_session(PiSessionInput(content=b"\xef\xbb\xbf" + _archive(records)))
    assert [event.event_id for event in events] == ["root", "tool", "summary", "custom"]
    assert [event.content for event in events] == [
        "assistant:   原文\u0085\u2028\u2029  ",
        'assistant: before\n[tool_call read call:一] {"a":"中文","z":[1,false]}\nafter',
        "compaction: 完整摘要",
        "custom:   自定义  ",
    ]
    assert events[1].parent_id == "control"
    assert events[1].record == records[3]
    assert parse_pi_session(PiSessionInput(content=_archive(records), leaf_id="control")) == events[:1]
    assert parse_pi_session(PiSessionInput(content=_archive([]))) == ()


@pytest.mark.parametrize(
    "raw",
    [
        b'{"type":"session","version":3.0,"id":"s"}',
        b'{"type":"session","version":true,"id":"s"}',
        b'{"type":"session","version":3,"id":""}',
        b'{"type":"session","version":3,"id":"s"}\n{"type":"model_change","id":"a"}',
        b'{"type":"session","version":3,"id":"s"}\n{"type":"model_change","id":"a","parentId":0}',
        b'{"type":"session","version":3,"id":"s"}\n{"type":"model_change","id":"a","parentId":"later"}',
        b'{"type":"session","version":3,"id":"s"}\nnull',
        b'{"type":"session","version":3,"id":"s"}\n{"type":"model_change","id":"a","parentId":null,"x":NaN}',
        b'{"type":"session","version":3,"id":"s"}\n\xff',
        _archive([_message("a", None, "valid"), _message("a", None, "duplicate")]),
        _archive([_message("a", None, "valid")]) + b'\n{"broken":',
    ],
)
def test_invalid_input_is_rejected_even_with_an_earlier_explicit_leaf(raw: bytes) -> None:
    with pytest.raises(ValueError):
        parse_pi_session(PiSessionInput(content=raw, leaf_id="a"))


def test_identity_is_bounded_stable_and_uses_both_ids() -> None:
    async def exercise() -> None:
        registry = SourceDefinitionRegistry((PI_EVENT_SOURCE_DEFINITION,))
        names = []
        for session_id, event_id in [("a:b", "c"), ("a", "b:c"), ("一" * 1000, "二" * 1000)]:
            record = _message(event_id, None, "original")
            event = parse_pi_session(PiSessionInput(content=_archive([record], session_id=session_id)))[0]
            source = await registry.resolve(event)
            assert len(source.name) <= 256
            assert source.materialization is SourceMaterialization.CAPTURED
            assert await registry.read(source) == event
            evidence = TextEvidence.model_validate(registry.project(source, TEXT_EVIDENCE_PROJECTION_KEY))
            assert evidence.source_id == source.name
            assert evidence.metadata == {
                "session_id": session_id,
                "event_id": event_id,
                "parent_id": None,
                "event_type": "message",
                "role": "assistant",
            }
            changed = parse_pi_session(
                PiSessionInput(content=_archive([_message(event_id, None, "changed")], session_id=session_id))
            )[0]
            assert (await registry.resolve(changed)).name == source.name
            names.append(source.name)
        assert len(set(names)) == len(names)

    asyncio.run(exercise())
