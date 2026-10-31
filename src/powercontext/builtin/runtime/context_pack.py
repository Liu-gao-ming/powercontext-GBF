"""Portable UTF-8 context packs and independently verifiable prefix deltas."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from dataclasses import dataclass
from typing import cast

_PACK_SCHEMA = "powercontext.context-pack.v1"
_DELTA_SCHEMA = "powercontext.context-pack-delta.v1"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


class ContextPackError(ValueError):
    """A pack, delta, or UTF-8 chunk violates the wire contract."""


def _json_bytes(value: dict[str, object]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _validate_items(items: tuple[bytes, ...]) -> None:
    if not isinstance(items, tuple) or any(not isinstance(item, bytes) for item in items):
        raise ContextPackError("chunks must be a tuple of bytes")  # noqa: TRY003
    try:
        for item in items:
            item.decode("utf-8")
    except UnicodeError as error:
        raise ContextPackError("each chunk must be valid UTF-8") from error  # noqa: TRY003


def _encoded(items: tuple[bytes, ...]) -> list[str]:
    return [base64.b64encode(item).decode("ascii") for item in items]


def _decoded(value: object) -> tuple[bytes, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ContextPackError("wire chunks must be an array of Base64 strings")  # noqa: TRY003
    try:
        items = tuple(base64.b64decode(item, validate=True) for item in value)
    except (ValueError, binascii.Error) as error:
        raise ContextPackError("invalid Base64 chunk") from error  # noqa: TRY003
    if _encoded(items) != value:
        raise ContextPackError("chunks must use standard Base64 encoding")  # noqa: TRY003
    _validate_items(items)
    return items


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ContextPackError("duplicate JSON field")  # noqa: TRY003
        result[key] = value
    return result


def _payload(data: bytes, schema: str, fields: set[str]) -> dict[str, object]:
    if not isinstance(data, bytes):
        raise ContextPackError("wire payload must be bytes")  # noqa: TRY003
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeError, ValueError, RecursionError) as error:
        raise ContextPackError("invalid JSON payload") from error  # noqa: TRY003
    if not isinstance(value, dict) or set(value) != fields or value["schema"] != schema:
        raise ContextPackError("invalid schema or fields")  # noqa: TRY003
    return value


def _validate_digest(value: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ContextPackError("invalid SHA-256 digest")  # noqa: TRY003


@dataclass(frozen=True, slots=True)
class ContextPack:
    """A complete context; the digest includes chunk boundaries as well as bytes."""

    items: tuple[bytes, ...]

    def __post_init__(self) -> None:
        _validate_items(self.items)

    @classmethod
    def build(cls, items: tuple[bytes, ...]) -> ContextPack:
        return cls(items)

    @classmethod
    def from_content(cls, content: str | bytes | None, *, chunk_bytes: int = 512) -> ContextPack:
        if type(chunk_bytes) is not int or chunk_bytes < 1:
            raise ContextPackError("chunk_bytes must be a positive integer")  # noqa: TRY003
        try:
            if content is None:
                data = b""
            elif isinstance(content, str):
                data = content.encode("utf-8")
            elif isinstance(content, bytes):
                content.decode("utf-8")
                data = content
            else:
                raise ContextPackError("content must be UTF-8 text or bytes")  # noqa: TRY003
        except UnicodeError as error:
            raise ContextPackError("invalid UTF-8 content") from error  # noqa: TRY003
        items: list[bytes] = []
        start = 0
        while start < len(data):
            end = min(start + chunk_bytes, len(data))
            while end < len(data) and end > start and data[end] & 0xC0 == 0x80:
                end -= 1
            if end == start:
                raise ContextPackError("chunk_bytes cannot fit the next UTF-8 character")  # noqa: TRY003
            items.append(data[start:end])
            start = end
        return cls.build(tuple(items))

    @property
    def digest(self) -> str:
        unsigned: dict[str, object] = {"schema": _PACK_SCHEMA, "items": _encoded(self.items)}
        return "sha256:" + hashlib.sha256(_json_bytes(unsigned)).hexdigest()

    def to_bytes(self) -> bytes:
        value: dict[str, object] = {
            "schema": _PACK_SCHEMA,
            "items": _encoded(self.items),
            "digest": self.digest,
            "byte_count": 0,
        }
        # Including byte_count itself requires a fixed point at digit boundaries.
        while True:
            wire = _json_bytes(value)
            if value["byte_count"] == len(wire):
                return wire
            value["byte_count"] = len(wire)

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPack:
        value = _payload(data, _PACK_SCHEMA, {"schema", "items", "digest", "byte_count"})
        pack = cls.build(_decoded(value["items"]))
        if value["digest"] != pack.digest:
            raise ContextPackError("pack digest mismatch")  # noqa: TRY003
        if type(value["byte_count"]) is not int or value["byte_count"] != pack.byte_count:
            raise ContextPackError("pack byte_count mismatch")  # noqa: TRY003
        return pack


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    """Retain a base prefix and append a suffix without a sender-side cache."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    def __post_init__(self) -> None:
        _validate_digest(self.base_digest)
        _validate_digest(self.target_digest)
        if type(self.prefix_items) is not int or self.prefix_items < 0:
            raise ContextPackError("prefix_items must be a non-negative integer")  # noqa: TRY003
        _validate_items(self.appended)

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        prefix = 0
        for old, new in zip(base.items, target.items, strict=False):
            if old != new:
                break
            prefix += 1
        return cls(base.digest, target.digest, prefix, target.items[prefix:])

    def to_bytes(self) -> bytes:
        return _json_bytes({
            "schema": _DELTA_SCHEMA,
            "base_digest": self.base_digest,
            "target_digest": self.target_digest,
            "prefix_items": self.prefix_items,
            "appended": _encoded(self.appended),
        })

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPackDelta:
        value = _payload(data, _DELTA_SCHEMA, {"schema", "base_digest", "target_digest", "prefix_items", "appended"})
        # Constructor validation remains authoritative for untrusted wire fields.
        return cls(
            cast(str, value["base_digest"]),
            cast(str, value["target_digest"]),
            cast(int, value["prefix_items"]),
            _decoded(value["appended"]),
        )


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    if base.digest != delta.base_digest:
        raise ContextPackError("delta base digest mismatch")  # noqa: TRY003
    if delta.prefix_items > len(base.items):
        raise ContextPackError("delta prefix exceeds the base")  # noqa: TRY003
    target = ContextPack.build(base.items[: delta.prefix_items] + delta.appended)
    if target.digest != delta.target_digest:
        raise ContextPackError("delta target digest mismatch")  # noqa: TRY003
    return target
