"""Independent context wire checks, UTF-8 boundaries, and stateless deltas."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json

import pytest

from powercontext.builtin.runtime.context_pack import (
    ContextPack,
    ContextPackDelta,
    ContextPackError,
    apply_context_pack_delta,
)


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def test_none_content_is_the_same_empty_pack_as_empty_text():
    assert ContextPack.from_content(None) == ContextPack.from_content("") == ContextPack.build(())


@pytest.mark.parametrize("size", [0, 1, 9, 99, 999, 9999])
def test_digest_and_self_inclusive_byte_count_match_independent_serialization(size):
    items = (b"a" * size, "中文🦀\r\n".encode(), b"")
    pack = ContextPack.build(items)
    unsigned = {"schema": "powercontext.context-pack.v1", "items": [base64.b64encode(item).decode() for item in items]}
    expected_digest = "sha256:" + hashlib.sha256(_wire(unsigned)).hexdigest()
    assert pack.digest == expected_digest
    assert _wire({**unsigned, "digest": expected_digest, "byte_count": pack.byte_count}) == pack.to_bytes()
    assert pack.byte_count == len(pack.to_bytes())
    assert ContextPack.from_bytes(pack.to_bytes()) == pack


@pytest.mark.parametrize("maximum", [4, 5, 7, 16, 511, 512, 513])
def test_bounded_unicode_chunks_are_deterministic_and_keep_exact_bytes(maximum):
    content = "\ufeffFirst\r\n海洋🦀 e\u0301\x00\u2028\n" * 70
    pack = ContextPack.from_content(content, chunk_bytes=maximum)
    assert pack == ContextPack.from_content(content.encode(), chunk_bytes=maximum)
    assert b"".join(pack.items) == content.encode()
    assert all(len(item) <= maximum and item.decode().encode() == item for item in pack.items)


@pytest.mark.parametrize("maximum", [0, -1, True, 1.0, "4", None])
def test_invalid_chunk_bounds(maximum):
    with pytest.raises(ContextPackError):
        ContextPack.from_content("a", chunk_bytes=maximum)


@pytest.mark.parametrize(("content", "maximum"), [("🦀", 3), ("ab中", 2), (b"\xff", 8), ("\ud800", 8)])
def test_invalid_utf8_and_impossible_character_boundaries(content, maximum):
    with pytest.raises(ContextPackError):
        ContextPack.from_content(content, chunk_bytes=maximum)


@pytest.mark.parametrize("changed", [(b"a", b"b", b"tail"), (b"head", b"b"), (b"a", b"middle"), (), (b"a",)])
def test_wire_only_receiver_reconstructs_every_valid_common_prefix(changed):
    base = ContextPack.build((b"a", b"b"))
    target = ContextPack.build(changed)
    delta = ContextPackDelta.between(base, target)
    assert delta.byte_count == len(delta.to_bytes())
    for prefix in range(delta.prefix_items + 1):
        payload = json.loads(delta.to_bytes())
        payload.update(prefix_items=prefix, appended=[base64.b64encode(item).decode() for item in changed[prefix:]])
        rebuilt = apply_context_pack_delta(
            ContextPack.from_bytes(base.to_bytes()), ContextPackDelta.from_bytes(_wire(payload))
        )
        assert rebuilt.items == target.items
        assert rebuilt.to_bytes() == target.to_bytes()


@pytest.mark.parametrize(
    "change",
    [
        {"schema": "unknown"},
        {"extra": 1},
        {"items": "YQ=="},
        {"items": [True]},
        {"items": ["Y Q=="]},
        {"items": ["YR=="]},
        {"items": ["/w=="]},
        {"digest": "sha256:" + "0" * 64},
        {"byte_count": True},
        {"byte_count": 1.0},
        {"byte_count": -1},
        {"byte_count": 100000},
    ],
)
def test_complete_pack_decoder_rejects_corrupt_fields(change):
    payload = json.loads(ContextPack.build((b"a",)).to_bytes())
    payload.update(change)
    with pytest.raises(ContextPackError):
        ContextPack.from_bytes(_wire(payload))


@pytest.mark.parametrize("decoder", [ContextPack.from_bytes, ContextPackDelta.from_bytes])
@pytest.mark.parametrize("data", [b"\xff", b"{", b"[]", b"null", b'{"schema":"x","schema":"x"}'])
def test_decoders_reject_nonobject_invalid_encoding_and_duplicate_json_fields(decoder, data):
    with pytest.raises(ContextPackError):
        decoder(data)


@pytest.mark.parametrize("field", ["schema", "base_digest", "target_digest", "prefix_items", "appended"])
def test_delta_requires_every_field(field):
    pack = ContextPack.build(())
    payload = json.loads(ContextPackDelta.between(pack, pack).to_bytes())
    del payload[field]
    with pytest.raises(ContextPackError):
        ContextPackDelta.from_bytes(_wire(payload))


@pytest.mark.parametrize("prefix", [True, False, -1, 1.0, "1", None])
def test_delta_prefix_has_strict_integer_type(prefix):
    pack = ContextPack.build(())
    with pytest.raises(ContextPackError):
        ContextPackDelta(pack.digest, pack.digest, prefix, ())


def test_40_round_synthetic_wire_cost_includes_first_pack_and_all_protocol_overheads():
    base = None
    raw_bytes = full_bytes = incremental_bytes = gzip_bytes = 0
    distinct = set()
    for round_index in range(40):
        # A synthetic append sequence measures protocol overhead, not official QMSum quality.
        content = ("会议上下文和引用\n" * 100) + ("new evidence\n" * (round_index // 10))
        target = ContextPack.from_content(content)
        raw = content.encode()
        distinct.add(raw)
        raw_bytes += len(raw)
        full_bytes += len(target.to_bytes())
        gzip_bytes += len(gzip.compress(raw, compresslevel=9, mtime=0))
        wire = target.to_bytes() if base is None else ContextPackDelta.between(base, target).to_bytes()
        incremental_bytes += len(wire)
        base = (
            ContextPack.from_bytes(wire)
            if base is None
            else apply_context_pack_delta(base, ContextPackDelta.from_bytes(wire))
        )
        assert b"".join(base.items) == raw
    assert len(distinct) >= 4
    assert 1 - incremental_bytes / raw_bytes >= 0.45
    # These are observations only: never require the protocol to beat gzip.
    print({
        "raw": raw_bytes,
        "full_pack": full_bytes,
        "incremental": incremental_bytes,
        "gzip": gzip_bytes,
        "saving_raw": 1 - incremental_bytes / raw_bytes,
        "saving_pack": 1 - incremental_bytes / full_bytes,
        "saving_gzip": 1 - incremental_bytes / gzip_bytes,
    })
