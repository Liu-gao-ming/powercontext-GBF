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

"""Complete physical PDF pages and offline retrieval of durable page citations."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from powercontext.artifacts import MemoryCitation
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

if TYPE_CHECKING:
    from pypdf import PageObject

    from powercontext.builtin.runtime.models import MemoryEntriesPage, MemorySearchPage, SearchMemoryRequest


def _document_id(value: str) -> str:
    if not value.strip():
        raise ValueError("PDF document_id must not be blank")  # noqa: TRY003
    return value


class PDFDocumentInput(BaseModel):
    document_id: str
    content: bytes

    _validate_document = field_validator("document_id")(_document_id)


class PDFPageInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    page_number: int = Field(ge=1, strict=True)
    text: str

    _validate_document = field_validator("document_id")(_document_id)


def _page_text(page: PageObject) -> str:
    from pypdf.generic import ArrayObject, NullObject, StreamObject

    contents = page.get("/Contents")
    if contents is not None:
        contents = contents.get_object()
        streams = contents if isinstance(contents, ArrayObject) else (contents,)
        for stream in streams:
            stream = stream.get_object()
            if not isinstance(stream, StreamObject | NullObject):
                raise TypeError("PDF page content must be a stream or array of streams")  # noqa: TRY003
    return page.extract_text() or ""


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    """Extract every physical page before exposing any result, including blanks."""
    from pypdf import PdfReader

    _document_id(value.document_id)
    digest = hashlib.sha256(value.content).hexdigest()
    try:
        reader = PdfReader(io.BytesIO(value.content), strict=True)
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("PDF requires a non-empty password")  # noqa: TRY003, TRY301
        return tuple(
            PDFPageInput(
                document_id=value.document_id,
                document_sha256=digest,
                page_number=number,
                text=_page_text(page),
            )
            for number, page in enumerate(reader.pages, start=1)
        )
    except Exception as error:
        # pypdf reports damaged page trees, streams and fonts with several
        # exception types. This adapter's whole-document failure is atomic.
        raise ValueError("PDF could not be completely parsed") from error  # noqa: TRY003


def _page_identity(page: PDFPageInput) -> str:
    framed = json.dumps(
        [page.document_id, page.document_sha256, page.page_number],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"pdf:{hashlib.sha256(framed.encode('utf-8')).hexdigest()}"


class PDFPageSource(Source):
    model_config = ConfigDict(frozen=True)

    materialization: Literal[SourceMaterialization.CAPTURED] = SourceMaterialization.CAPTURED
    document_id: str
    document_sha256: str
    page_number: int
    text: str

    @model_validator(mode="after")
    def validate_page(self) -> PDFPageSource:
        page = PDFPageInput.model_validate(self.model_dump())
        if self.name != _page_identity(page):
            raise ValueError("PDF Source name must equal its revision and physical page identity")  # noqa: TRY003
        return self


class _PDFPageAdapter:
    name = "pdf-page"
    input_class = PDFPageInput
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        return PDFPageSource(name=_page_identity(value), **value.model_dump())

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput.model_validate(source.model_dump())


class _PDFTextProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type="pdf-page",
            source_id=source.name,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


PDF_PAGE_SOURCE_DEFINITION = AdapterSourceDefinition(_PDFPageAdapter(), projections=(_PDFTextProjection(),))


class _ScopedMemory(Protocol):
    async def list(self) -> MemoryEntriesPage: ...

    async def search(self, request: SearchMemoryRequest, /) -> MemorySearchPage: ...


_WORDS = re.compile(r"[a-z]+|\d+")
_STOP_WORDS = frozenset([
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    "any",
    "only",
    "all",
    "also",
    "not",
    "no",
    "than",
    "using",
    "use",
    "used",
    "shown",
    "show",
    "details",
    "detail",
    "question",
    "answer",
    "respond",
    "response",
    "give",
    "state",
    "calculate",
    "calculated",
    "define",
    "defined",
    "assuming",
    "assume",
    "perspective",
    "investment",
    "analyst",
    "reported",
    "report",
    "primarily",
    "directly",
    "explicitly",
    "important",
    "outlined",
    "relying",
    "rely",
    "fy",
    "fiscal",
    "year",
    "years",
    "usd",
    "dollar",
    "dollars",
    "million",
    "millions",
    "billion",
    "billions",
])
# General financial vocabulary, independent of a particular company, period,
# question, answer or evaluation workload. Each group is a separate recall facet.
_CONCEPTS = (
    (
        "geograph geography region countries operate",
        ("geographic segment international market", "business divisions segments"),
    ),
    ("legal litigation lawsuit battle", ("legal proceedings", "litigation contingencies")),
    ("restructuring restructure", ("restructuring charges costs", "statement income")),
    ("ebitda", ("operating income", "depreciation amortization", "statement income", "cash flows")),
    ("capex capital expenditure", ("capital expenditures", "property plant equipment")),
    ("cash flow", ("cash flows",)),
    ("income earnings profit margin revenue sales", ("statement income",)),
    ("balance assets liabilities liquidity ratio", ("balance sheet", "current assets liabilities")),
    ("gross margin", ("net sales revenue", "cost sales products goods")),
    ("net profit margin", ("net income", "net sales revenue")),
    ("return assets equity", ("net income", "total assets equity")),
    ("dividend repurchase shareholder", ("dividends share repurchase",)),
    ("debt leverage interest", ("debt interest borrowings",)),
    ("working capital", ("current assets liabilities",)),
)


def _stem(word: str) -> str:
    if word.isdigit() or len(word) < 4:
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and not word.endswith("ss") and len(word) > len(suffix) + 2:
            word = word[: -len(suffix)]
            break
    return word.removesuffix("e") if len(word) > 4 else word


def _terms(text: str) -> tuple[str, ...]:
    return tuple(
        _stem(word) for word in _WORDS.findall(unicodedata.normalize("NFKC", text).lower()) if word not in _STOP_WORDS
    )


def _question_terms(question: str) -> tuple[dict[str, float], tuple[str, ...]]:
    originals = tuple(dict.fromkeys(_terms(question)))
    terms = set(originals)
    facets = tuple(
        dict.fromkeys(
            facet for triggers, expansions in _CONCEPTS if terms.intersection(_terms(triggers)) for facet in expansions
        )
    )
    # Fiscal years recur in almost every table. Score the report's period
    # separately so dates cannot overwhelm the financial concepts in its body.
    weights = {term: 0.15 if term.isdigit() and len(term) == 4 else 1.0 for term in originals}
    for facet in facets:
        for word in _terms(facet):
            weights.setdefault(word, 0.4)
    return weights, facets


def _heading_terms(text: str) -> frozenset[str]:
    heading = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text[:600]).lower())
    topics = (
        (r"statements? of (?:income|operations|earnings)", "income profit statement"),
        (r"statements? of cash flows?", "cash flow depreciation amortization"),
        (r"balance sheets?|statements? of financial position", "balance sheet assets liabilities"),
        (r"legal proceedings|litigation|commitments and contingencies", "legal litigation proceedings"),
        (r"(?:our|business) divisions|geographic (?:information|segments?)", "geography geographic division segment"),
    )
    return frozenset(term for pattern, words in topics if re.search(pattern, heading) for term in _terms(words))


@dataclass(frozen=True)
class _PageEntry:
    citation: MemoryCitation
    source_id: str
    document_id: str
    terms: Counter[str]
    heading_terms: frozenset[str]


def _memory_pages(snapshot: MemoryEntriesPage) -> tuple[_PageEntry, ...]:
    entries: dict[str, _PageEntry] = {}
    for record in snapshot.entries:
        sources = record.entry.sources
        if len(sources) != 1 or sources[0].source_type != "pdf-page":
            continue
        document_id, _, text = record.entry.text.partition("\n")
        source_id = sources[0].source_id
        entries.setdefault(
            source_id,
            _PageEntry(record.citation, source_id, document_id, Counter(_terms(text)), _heading_terms(text)),
        )
    return tuple(entries.values())


async def _fts_recall(memory: _ScopedMemory, queries: tuple[str, ...]) -> Counter[tuple[str, str]]:
    from powercontext.builtin.runtime.models import SearchMemoryRequest

    scores: Counter[tuple[str, str]] = Counter()
    for query in queries[:10]:
        if not query:
            continue
        result = await memory.search(SearchMemoryRequest(query=query, mode="fts", limit=100))
        for rank, hit in enumerate(result.hits, 1):
            scores[(hit.entry_id, hit.entry_version_id)] += 1.0 / (20 + rank)
    return scores


def _bm25(page: _PageEntry, weights: dict[str, float], idf: dict[str, float], average_length: float) -> float:
    length = sum(page.terms.values())
    normalization = 1.2 * (0.25 + 0.75 * length / average_length)
    return sum(
        weight * idf.get(term, 0.0) * count * 2.2 / (count + normalization)
        for term, weight in weights.items()
        if (count := page.terms.get(term, 0))
    )


async def search_pdf_pages(memory: _ScopedMemory, question: str, limit: int = 10) -> tuple[MemoryCitation, ...]:
    """Search only active scoped Memory, reranking genuine page entry citations.

    A public Memory snapshot supplies the corpus for document-frequency scoring
    and low-overlap fallbacks. FTS adds lexical recall signals; neither path
    accesses source files, backend tables or process-global retrieval caches.
    """
    if limit <= 0 or not question.strip():
        return ()
    weights, facets = _question_terms(question)
    if not weights:
        return ()
    pages = _memory_pages(await memory.list())
    if not pages:
        return ()
    queries = tuple(
        dict.fromkeys((" ".join(word for word in _WORDS.findall(question.lower()) if word not in _STOP_WORDS), *facets))
    )
    fts_scores = await _fts_recall(memory, queries)
    return _rank_pages(pages, question, weights, facets, fts_scores, limit)


def _rank_pages(
    pages: tuple[_PageEntry, ...],
    question: str,
    weights: dict[str, float],
    facets: tuple[str, ...],
    fts_scores: Counter[tuple[str, str]],
    limit: int,
) -> tuple[MemoryCitation, ...]:
    frequencies: Counter[str] = Counter()
    for page in pages:
        frequencies.update(page.terms.keys())
    idf = {term: math.log1p((len(pages) - count + 0.5) / (count + 0.5)) for term, count in frequencies.items()}
    average_length = max(1.0, sum(sum(page.terms.values()) for page in pages) / len(pages))
    question_words = set(_terms(question))

    def score(page: _PageEntry, query_weights: dict[str, float]) -> float:
        value = _bm25(page, query_weights, idf, average_length)
        value += 4.0 * sum(query_weights.get(term, 0.0) for term in page.heading_terms)
        label = set(_terms(page.document_id)) - {"10", "k", "q", "annual"}
        # Identifiers and periods are repeated on every projected page, so score
        # them separately instead of allowing them to dominate corpus term counts.
        matches = label.intersection(question_words)
        company_match = any(not term.isdigit() for term in matches)
        period_match = any(term.isdigit() and len(term) == 4 for term in matches)
        return value * (1.0 + 0.8 * company_match + 0.15 * period_match)

    ranked = sorted(
        pages,
        key=lambda page: (
            score(page, weights) + 2.0 * fts_scores[(page.citation.entry_id, page.citation.entry_version_id)]
        ),
        reverse=True,
    )
    # Multi-part financial questions often need different statements. Reserve a
    # bounded share for the best pages of each facet, then fill by overall score.
    selected: dict[str, MemoryCitation] = {}
    reserve = min(2, limit // (len(facets) + 2)) if facets else 0
    for facet in facets:
        facet_weights = dict.fromkeys(_terms(facet), 1.0)
        best = sorted(pages, key=lambda page: score(page, facet_weights), reverse=True)
        for page in [page for page in best if score(page, facet_weights) > 0][:reserve]:
            selected.setdefault(page.source_id, page.citation)
    for page in ranked:
        if score(page, weights) > 0:
            selected.setdefault(page.source_id, page.citation)
        if len(selected) >= limit:
            break
    # Keep overall relevance ordering even for pages admitted by a recall facet.
    return tuple(selected[page.source_id] for page in ranked if page.source_id in selected)[:limit]


__all__ = [
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "parse_pdf_document",
    "search_pdf_pages",
]
