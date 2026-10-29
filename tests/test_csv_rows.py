"""CSV edge cases beyond the public airport workload."""

from __future__ import annotations

import asyncio
import json

import pytest

from powercontext.sources import SourceDefinitionRegistry, SourceMaterialization, TextEvidence
from powercontext.sources.csv_rows import (
    CSV_ROW_SOURCE_DEFINITION,
    CSVDocumentInput,
    CSVRowInput,
    CSVRowSource,
    csv_row_source_id,
    parse_csv_document,
)
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY


def test_unicode_order_and_string_values_survive_source_serialization() -> None:
    row = parse_csv_document(
        CSVDocumentInput(
            document_id="机场/%:一",
            primary_key=" id ",
            content="z, id ,空值,数字,布尔\n北京, 001 ,,001.00,false\n".encode(),
        )
    )[0]
    registry = SourceDefinitionRegistry((CSV_ROW_SOURCE_DEFINITION,))
    source = asyncio.run(registry.resolve(row))
    restored = CSVRowSource.model_validate_json(source.model_dump_json())
    assert restored.materialization is SourceMaterialization.CAPTURED
    assert asyncio.run(registry.read(restored)) == row
    evidence = TextEvidence.model_validate(registry.project(restored, TEXT_EVIDENCE_PROJECTION_KEY))
    assert evidence.content == '{"z":"北京"," id ":" 001 ","空值":"","数字":"001.00","布尔":"false"}'
    assert list(json.loads(evidence.content)) == list(row.columns)
    metadata_row = evidence.metadata["row"]
    assert isinstance(metadata_row, dict)
    assert list(metadata_row) == list(row.columns)
    assert set(evidence.metadata) == {"document_id", "primary_key", "primary_key_value", "columns", "row"}


@pytest.mark.parametrize(
    "content",
    [
        b"\nkey\nvalue\n",
        b"\xef\xbb\xbf",
        b'""\n',
        b'key\n""\n',
        b"key\n \t\n",
        b'key,name\nvalid,first\nbad,"quoted"tail\n',
        b'key,name\nvalid,first\nbad,"unterminated\n',
        b"key,name\nvalid,first\nbad,too,many\n",
        b"key,name\nvalid,first\nvalid,duplicate\n",
        b"key,name\nvalid,first\n\xff,bad\n",
    ],
)
def test_invalid_tail_and_empty_fields_fail_the_whole_document(content: bytes) -> None:
    with pytest.raises(ValueError):
        parse_csv_document(CSVDocumentInput(document_id="doc", primary_key="key", content=content))


def test_only_empty_physical_records_are_ignored() -> None:
    rows = parse_csv_document(
        CSVDocumentInput(
            document_id="doc",
            primary_key="key",
            content=b'key\r\n\r\n"a\r\nb"\r\n\r\nlast\r\n',
        )
    )
    assert tuple(row.values for row in rows) == (("a\r\nb",), ("last",))


def test_headers_are_exact_and_not_normalized() -> None:
    content = b"id, id ,ID\n1,2,3\n"
    row = parse_csv_document(CSVDocumentInput(document_id="doc", primary_key=" id ", content=content))[0]
    assert row.primary_key_value == "2"
    with pytest.raises(ValueError):
        parse_csv_document(CSVDocumentInput(document_id="doc", primary_key="id ", content=content))
    with pytest.raises(ValueError):
        parse_csv_document(CSVDocumentInput(document_id="doc", primary_key="", content=b"id\n"))


def test_identity_length_is_measured_after_percent_encoding() -> None:
    assert len(csv_row_source_id("d", "x" * 250)) == 256
    with pytest.raises(ValueError):
        csv_row_source_id("d", "x" * 251)
    assert len(csv_row_source_id("d", "中" * 27 + "x" * 7)) == 256
    with pytest.raises(ValueError):
        csv_row_source_id("d", "中" * 28)
    assert csv_row_source_id("a:b", "c") != csv_row_source_id("a", "b:c")
    assert csv_row_source_id("a%3Ab", "c") != csv_row_source_id("a:b", "c")


@pytest.mark.parametrize(
    "changes",
    [
        {"primary_key_value": "other"},
        {"values": ()},
        {"columns": ("id", "id"), "values": ("1", "1")},
        {"primary_key": " id "},
        {"document_id": "doc "},
    ],
)
def test_direct_row_inputs_reject_inconsistent_captures(changes: dict[str, object]) -> None:
    fields = {"document_id": "doc", "primary_key": "id", "primary_key_value": "1", "columns": ("id",), "values": ("1",)}
    with pytest.raises(ValueError):
        CSVRowInput.model_validate(fields | changes)


def test_csv_reader_permissive_unquoted_quotes_are_preserved() -> None:
    row = parse_csv_document(
        CSVDocumentInput(
            document_id="doc",
            primary_key="id",
            content=b'id,note\n1,unquoted"quote\n',
        )
    )[0]
    assert row.values == ("1", 'unquoted"quote')
