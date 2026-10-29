"""Small real PDFs for Source behavior tests, without an extra PDF dependency."""

from __future__ import annotations

import io
from collections.abc import Iterable


def make_pdf(texts: Iterable[str]) -> bytes:
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    for text in texts:
        page = writer.add_blank_page(width=400, height=700)
        if not text:
            continue
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
        })
        operations = ["BT /F1 10 Tf 20 650 Td"]
        for line in text.split("\n"):
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            operations.append(f"({escaped}) Tj 0 -14 Td")
        operations.append("ET")
        contents = DecodedStreamObject()
        contents.set_data("\n".join(operations).encode("latin-1"))
        page.replace_contents(contents.flate_encode())
    result = io.BytesIO()
    writer.write(result)
    return result.getvalue()
