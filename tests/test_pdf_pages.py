"""Full-page parsing, exact values, revisions and failure boundaries."""

from __future__ import annotations

import asyncio
import hashlib
import io
import re
from collections import Counter

import pytest

from powercontext.sources import SourceDefinitionRegistry, SourceMaterialization, TextEvidence
from powercontext.sources.pdf_pages import (
    PDF_PAGE_SOURCE_DEFINITION,
    PDFDocumentInput,
    PDFPageInput,
    parse_pdf_document,
)
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY
from tests.pdf_support import make_pdf

pytest.importorskip("pypdf")


def test_physical_page_order_blank_page_and_exact_projection() -> None:
    raw = make_pdf(("Statement of income\nRevenue 1,250\nProfit (100)\nPrinted page 99", "", "Notes\nTax 22.50"))
    pages = parse_pdf_document(PDFDocumentInput(document_id=" 文件 /:一 ", content=raw))
    assert [page.page_number for page in pages] == [1, 2, 3]
    assert pages[1].text == ""
    assert Counter(re.findall(r"[a-z]+|\d+", pages[0].text.lower())) == Counter(
        re.findall(r"[a-z]+|\d+", "Statement of income Revenue 1,250 Profit (100) Printed page 99".lower())
    )
    assert all(page.document_sha256 == hashlib.sha256(raw).hexdigest() for page in pages)

    async def exercise() -> None:
        registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
        for page in pages:
            source = await registry.resolve(page)
            assert source.materialization is SourceMaterialization.CAPTURED
            assert await registry.read(source) == page
            evidence = TextEvidence.model_validate(registry.project(source, TEXT_EVIDENCE_PROJECTION_KEY))
            assert evidence.content == f"{page.document_id}\n{page.text}"
            assert evidence.source_type == "pdf-page" and evidence.source_id == source.name
            assert evidence.metadata == {
                "document_id": page.document_id,
                "document_sha256": page.document_sha256,
                "page_number": page.page_number,
            }

    asyncio.run(exercise())


def test_aes_empty_user_password_is_legal_and_nonempty_password_fails() -> None:
    from pypdf import PdfWriter

    original = make_pdf(("Permission controlled text 1234",))
    for password in ("", "required"):
        writer = PdfWriter(clone_from=io.BytesIO(original))
        writer.encrypt(user_password=password, owner_password="owner", algorithm="AES-256")  # noqa: S106 - fixture only
        output = io.BytesIO()
        writer.write(output)
        document = PDFDocumentInput(document_id="permissions", content=output.getvalue())
        if password:
            with pytest.raises(ValueError):
                parse_pdf_document(document)
        else:
            assert "controlled text 1234" in parse_pdf_document(document)[0].text


def test_corrupt_late_page_fails_the_entire_document() -> None:
    from pypdf import PdfWriter
    from pypdf.generic import NameObject, NumberObject

    writer = PdfWriter(clone_from=io.BytesIO(make_pdf(("Valid first page", "Second page"))))
    writer.pages[1][NameObject("/Contents")] = NumberObject(99)
    output = io.BytesIO()
    writer.write(output)
    with pytest.raises(ValueError):
        parse_pdf_document(PDFDocumentInput(document_id="corrupt", content=output.getvalue()))


@pytest.mark.parametrize("document_id", ["", " \n\t"])
def test_blank_document_ids_fail(document_id: str) -> None:
    with pytest.raises(ValueError):
        parse_pdf_document(PDFDocumentInput(document_id=document_id, content=make_pdf(("value",))))


def test_identity_is_bounded_and_includes_the_full_revision_and_document_id() -> None:
    async def exercise() -> None:
        registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
        names = set()
        for document_id, digest, number in [
            ("a:b", "a" * 64, 1),
            ("a", "b" * 64, 1),
            ("a:b", "b" * 64, 1),
            ("a:b", "a" * 64, 2),
            ("长" * 1000, "a" * 64, 1),
        ]:
            page = PDFPageInput(document_id=document_id, document_sha256=digest, page_number=number, text="Full body")
            source = await registry.resolve(page)
            assert len(source.name) <= 256
            assert (await registry.resolve(page)).name == source.name
            names.add(source.name)
        assert len(names) == 5

    asyncio.run(exercise())
