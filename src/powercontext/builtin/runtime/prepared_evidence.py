"""Query-ranked, literal Memory passages with exact, fully charged citations."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from powercontext.artifacts import ArtifactAddress
from powercontext.builtin.runtime.errors import PreparedContextInvariantError
from powercontext.builtin.runtime.prepared_text import TRUST_POLICY

if TYPE_CHECKING:
    from powercontext.builtin.runtime.models import PrepareContextRequest
    from powercontext.builtin.runtime.prepared_context import PreparedMemoryCandidates

_STOPWORDS = frozenset([
    "a",
    "an",
    "the",
    "of",
    "in",
    "to",
    "and",
    "or",
    "for",
    "on",
    "with",
    "by",
    "as",
    "at",
    "from",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "their",
    "there",
    "than",
    "then",
    "which",
    "who",
    "how",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "can",
    "could",
    "may",
    "might",
    "will",
    "would",
    "should",
])
_BOUNDARY = re.compile(r"[。！？]\s*|[.!?][\"')\]]*(?:\s+|$)|\n+")  # noqa: RUF001 - CJK punctuation boundaries.
_WORDS = re.compile(r"[^\W_]+", re.UNICODE)
_CJK = re.compile(r"[\u3400-\u9fff]+")


@dataclass(frozen=True)
class MemoryEntryAddress:
    """Identify one exact Memory entry version across Scope boundaries."""

    memory: ArtifactAddress
    entry_id: str
    entry_version_id: str


@dataclass(frozen=True)
class PreparedMemoryFragment:
    """An unchanged half-open character interval of a cited Memory entry."""

    address: MemoryEntryAddress
    start: int
    end: int
    text: str

    def __post_init__(self) -> None:
        if type(self.start) is not int or type(self.end) is not int or not 0 <= self.start < self.end:
            raise PreparedContextInvariantError("evidence-interval")
        if len(self.text) != self.end - self.start:
            raise PreparedContextInvariantError("evidence-length")


@dataclass(frozen=True)
class _Document:
    address: MemoryEntryAddress
    text: str
    spans: tuple[tuple[int, int], ...]


def _terms(text: str) -> Counter[str]:
    terms: Counter[str] = Counter()
    for word in _WORDS.findall(text.casefold()):
        if word in _STOPWORDS:
            continue
        if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        terms[word] += 1
        for run in _CJK.findall(word):
            terms.update(run[index : index + 2] for index in range(len(run) - 1))
        # A weaker morphological feature connects long inflected scientific terms.
        if len(word) > 7 and word.isascii() and word.isalpha():
            terms["~" + word[:6]] += 1
    return terms


def _sentences(text: str) -> tuple[tuple[int, int], ...]:
    spans = []
    start = 0
    for end in (*[match.end() for match in _BOUNDARY.finditer(text)], len(text)):
        left, right = start, end
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if left < right:
            spans.append((left, right))
        start = end
    return tuple(spans)


def render_evidence(fragments: Sequence[PreparedMemoryFragment]) -> str:
    items = []
    for fragment in fragments:
        address = fragment.address
        items.append({
            "citation": {
                "scope_id": address.memory.scope_id,
                "memory_ref": address.memory.artifact.model_dump(mode="json"),
                "entry_id": address.entry_id,
                "entry_version_id": address.entry_version_id,
            },
            "content": fragment.text,
            "start": fragment.start,
            "end": fragment.end,
        })
    payload = json.dumps({"trust": "untrusted_history", "items": items}, ensure_ascii=False, separators=(",", ":"))
    return f"{TRUST_POLICY}\n\nBEGIN_POWERCONTEXT_PREPARED_CONTEXT_V1\n{payload}\nEND_POWERCONTEXT_PREPARED_CONTEXT_V1"


def _fragments(documents: Sequence[_Document], selected: dict[int, set[int]]) -> tuple[PreparedMemoryFragment, ...]:
    fragments = []
    for number, indexes in selected.items():
        document = documents[number]
        groups: list[list[int]] = []
        for index in sorted(indexes):
            if groups and index == groups[-1][-1] + 1:
                groups[-1].append(index)
            else:
                groups.append([index])
        for group in groups:
            start, end = document.spans[group[0]][0], document.spans[group[-1]][1]
            fragments.append(PreparedMemoryFragment(document.address, start, end, document.text[start:end]))
    return tuple(fragments)


def _documents(candidates: Sequence[PreparedMemoryCandidates]) -> list[_Document]:
    documents = []
    seen = set()
    for group in candidates:
        for hit in group.hits:
            if group.memory_ref != hit.memory_ref or hit.memory_ref.family != "memory":
                raise PreparedContextInvariantError("memory-ref-mismatch")
            address = MemoryEntryAddress(
                ArtifactAddress(scope_id=group.scope_id, artifact=hit.memory_ref), hit.entry_id, hit.entry_version_id
            )
            identity = (
                group.scope_id,
                hit.memory_ref.artifact_id,
                hit.memory_ref.revision,
                hit.entry_id,
                hit.entry_version_id,
            )
            if identity in seen:
                continue
            seen.add(identity)
            documents.append(_Document(address, hit.text, _sentences(hit.text)))
    return documents


def _score_documents(documents: Sequence[_Document], query: Counter[str]) -> list[list[float]]:
    sentences = [_terms(document.text[start:end]) for document in documents for start, end in document.spans]
    if not sentences:
        return [[] for _ in documents]
    frequency = Counter(term for sentence in sentences for term in sentence)
    weights = {
        term: math.log1p((len(sentences) - frequency[term] + 0.5) / (frequency[term] + 0.5))
        * (0.35 if term.startswith("~") else 1.0)
        for term in query
    }
    average = sum(sum(sentence.values()) for sentence in sentences) / len(sentences) or 1.0
    scores = []
    for sentence in sentences:
        norm = 1.2 * (0.25 + 0.75 * sum(sentence.values()) / average)
        scores.append(sum(weight * sentence[term] * 2.2 / (sentence[term] + norm) for term, weight in weights.items()))
    by_document = []
    offset = 0
    for document in documents:
        relevance = scores[offset : offset + len(document.spans)]
        offset += len(document.spans)
        # Neighbouring sentences retain definitions and evidence continuations.
        best = max(relevance, default=0)
        augmented = [
            score
            + max(
                (
                    0.12 * relevance[other] / abs(index - other)
                    for other in range(max(0, index - 2), min(len(relevance), index + 3))
                    if other != index
                ),
                default=0,
            )
            for index, score in enumerate(relevance)
        ]
        by_document.append(augmented if best else relevance)
    return by_document


def select_memory_evidence(
    request: PrepareContextRequest, candidates: Sequence[PreparedMemoryCandidates]
) -> tuple[str | None, tuple[PreparedMemoryFragment, ...]]:
    documents = _documents(candidates)
    query = _terms(request.query)
    scores = _score_documents(documents, query)
    fragments = _choose_passages(request.max_bytes, documents, scores)
    clipped = _fit_long_sentence(request, documents, scores, query)
    if clipped and _query_coverage(clipped, query) > _query_coverage(fragments, query):
        fragments = clipped
    return (render_evidence(fragments), fragments) if fragments else (None, ())


def _query_coverage(fragments: Sequence[PreparedMemoryFragment], query: Counter[str]) -> float:
    # Do not let a short, weakly matching sentence hide a stronger oversized
    # sentence. Coverage is evaluated per original entry, including split spans.
    entries: dict[tuple[str, str, int, str, str], Counter[str]] = {}
    for fragment in fragments:
        address = fragment.address
        ref = address.memory.artifact
        identity = (address.memory.scope_id, ref.artifact_id, ref.revision, address.entry_id, address.entry_version_id)
        entries.setdefault(identity, Counter()).update(_terms(fragment.text))
    return max(
        (sum(0.35 if term.startswith("~") else 1.0 for term in query if terms[term]) for terms in entries.values()),
        default=0,
    )


def _choose_passages(
    max_bytes: int, documents: Sequence[_Document], scores: Sequence[Sequence[float]]
) -> tuple[PreparedMemoryFragment, ...]:
    selected: dict[int, set[int]] = {}
    fragments: tuple[PreparedMemoryFragment, ...] = ()
    used_bytes = 0
    while True:
        winner = None
        for number, document in enumerate(documents):
            chosen = selected.get(number, set())
            for start in range(len(document.spans)):
                for end in range(start + 1, min(start + 4, len(document.spans)) + 1):
                    added = set(range(start, end)) - chosen
                    gain = sum(scores[number][index] for index in added)
                    if not gain:
                        continue
                    trial = {**selected, number: chosen | added}
                    proposed = _fragments(documents, trial)
                    size = len(render_evidence(proposed).encode("utf-8"))
                    if size > max_bytes:
                        continue
                    # Joining adjacent evidence saves citation overhead. Stable loop
                    # order breaks ties without confusing equal text across scopes.
                    utility = gain / max(1, size - used_bytes) ** 0.35
                    if winner is None or utility > winner[0]:
                        winner = utility, trial, proposed, size
        if winner is None:
            break
        _, selected, fragments, used_bytes = winner
    return fragments


def _fit_long_sentence(
    request: PrepareContextRequest,
    documents: Sequence[_Document],
    scores: Sequence[Sequence[float]],
    query: Counter[str],
) -> tuple[PreparedMemoryFragment, ...]:
    """A sentence may itself exceed the wire budget; retain a literal query window."""
    ranked = sorted(
        ((score, number, index) for number, values in enumerate(scores) for index, score in enumerate(values) if score),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    for _, number, index in ranked:
        document = documents[number]
        start, end = document.spans[index]
        full = PreparedMemoryFragment(document.address, start, end, document.text[start:end])
        if len(render_evidence((full,)).encode("utf-8")) <= request.max_bytes:
            continue
        probe = PreparedMemoryFragment(document.address, start, start + 1, document.text[start : start + 1])
        overhead = len(render_evidence((probe,)).encode("utf-8")) - len(probe.text.encode("utf-8"))
        character_budget = (
            max(1, request.max_bytes - overhead) * (end - start) // len(document.text[start:end].encode("utf-8"))
        )
        center = _query_center(document.text, start, end, query, character_budget)
        lower, upper = 1, end - start
        best = ()
        while lower <= upper:
            width = (lower + upper) // 2
            left = max(start, min(center - width // 2, end - width))
            right = left + width
            fragment = PreparedMemoryFragment(document.address, left, right, document.text[left:right])
            if len(render_evidence((fragment,)).encode("utf-8")) <= request.max_bytes:
                best = (fragment,)
                lower = width + 1
            else:
                upper = width - 1
        if best:
            return best
    return ()


def _query_center(text: str, start: int, end: int, query: Counter[str], budget: int) -> int:
    matches = sorted(
        [
            (term, (match.start() + match.end()) // 2)
            for term in query
            if not term.startswith("~")
            for match in re.compile(re.escape(term), re.IGNORECASE).finditer(text, start, end)
        ],
        key=lambda item: item[1],
    )
    if not matches:
        return start
    # Cluster around distinct query terms, rather than midway between distant
    # occurrences (which could put the entire window in unrelated text).
    radius = max(1, budget // 2)
    left = right = 0
    window: Counter[str] = Counter()
    best_score, best_center = -1, start
    for _, center in matches:
        while right < len(matches) and matches[right][1] <= center + radius:
            window[matches[right][0]] += 1
            right += 1
        while left < len(matches) and matches[left][1] < center - radius:
            term = matches[left][0]
            window[term] -= 1
            if not window[term]:
                del window[term]
            left += 1
        if len(window) > best_score:
            best_score, best_center = len(window), center
    return best_center
