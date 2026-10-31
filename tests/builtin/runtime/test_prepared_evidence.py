"""Literal evidence selection, origin isolation, and complete wire budgets."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory import MemoryHit
from powercontext.builtin.runtime import PrepareContextRequest
from powercontext.builtin.runtime.prepared_context import PreparedContextBuilder, PreparedMemoryCandidates
from powercontext.builtin.runtime.prepared_text import TRUST_POLICY

REF = ArtifactRef(family="memory", artifact_id="memory-evidence", revision=9)


def _group(texts, scope="scope-evidence"):
    return PreparedMemoryCandidates(
        scope_id=scope,
        memory_ref=REF,
        hits=tuple(
            MemoryHit(
                memory_ref=REF,
                entry_id=f"entry-{index}",
                entry_version_id=f"version-{index}",
                text=text,
                score=1.0,
                matched_by=("fts",),
            )
            for index, text in enumerate(texts)
        ),
    )


def _build(groups, query="cobalt calibration", budget=1400, **options):
    return PreparedContextBuilder().build_scopes_result(
        request=PrepareContextRequest(query=query, max_bytes=budget, evidence_selection=True, **options),
        current_scope_id=groups[0].scope_id if groups else "scope-evidence",
        memory_candidates=groups,
    )


def _assert_literal(build, groups, budget):
    body = build.context.content or ""
    assert build.context.content_bytes == len(body.encode("utf-8")) <= budget
    assert len(build.fragments) == len(build.origins)
    if not body:
        assert not build.fragments
        return
    assert body.startswith(TRUST_POLICY + "\n\nBEGIN_POWERCONTEXT_PREPARED_CONTEXT_V1\n")
    assert body.endswith("\nEND_POWERCONTEXT_PREPARED_CONTEXT_V1")
    payload = json.loads(body.splitlines()[-2])
    assert payload["trust"] == "untrusted_history"
    assert len(payload["items"]) == len(build.fragments)
    identities = set()
    for item, fragment, origin in zip(payload["items"], build.fragments, build.origins, strict=True):
        address = fragment.address
        assert origin == address
        group = next(group for group in groups if group.scope_id == address.memory.scope_id)
        hit = next(hit for hit in group.hits if hit.entry_id == address.entry_id)
        assert address.memory.artifact == hit.memory_ref
        assert address.entry_version_id == hit.entry_version_id
        assert type(fragment.start) is int and type(fragment.end) is int
        assert 0 <= fragment.start < fragment.end <= len(hit.text)
        assert fragment.text == hit.text[fragment.start : fragment.end] == item["content"]
        assert item["start"] == fragment.start and item["end"] == fragment.end
        assert item["citation"] == {
            "scope_id": group.scope_id,
            "memory_ref": REF.model_dump(mode="json"),
            "entry_id": hit.entry_id,
            "entry_version_id": hit.entry_version_id,
        }
        identity = (group.scope_id, hit.entry_id, fragment.start, fragment.end)
        assert identity not in identities
        identities.add(identity)


@pytest.mark.parametrize("budget", (512, 650, 800, 1000, 1400, 1800))
def test_chinese_long_sentence_escaping_and_budget_include_every_wire_byte(budget):
    group = _group(("无关背景" * 180 + '温度传感器"校准\\记录37度' + "后续说明" * 40,))
    build = _build((group,), query="温度传感器 校准", budget=budget)
    _assert_literal(build, (group,), budget)
    if budget >= 800:
        assert build.fragments
        assert any("温度传感器" in fragment.text for fragment in build.fragments)


def test_query_switch_chooses_separated_regions_in_same_entry():
    text = (
        "Cobalt calibration reached 37 degrees. "
        + "Routine maintenance. " * 50
        + "Amber valve pressure reached 19 bars."
    )
    group = _group((text,))
    cobalt = _build((group,), query="cobalt calibration", budget=1000)
    amber = _build((group,), query="amber valve pressure", budget=1000)
    _assert_literal(cobalt, (group,), 1000)
    _assert_literal(amber, (group,), 1000)
    assert "37" in "".join(fragment.text for fragment in cobalt.fragments)
    assert "19" in "".join(fragment.text for fragment in amber.fragments)
    both = _build((group,), query="cobalt calibration amber valve pressure", budget=1400)
    _assert_literal(both, (group,), 1400)
    assert len(both.fragments) >= 2
    assert all(fragment.address.entry_id == "entry-0" for fragment in both.fragments)


def test_long_sentence_window_keeps_cluster_instead_of_unrelated_midpoint():
    text = "cobalt " + "background " * 250 + "cobalt calibration target 37 degrees " + "background " * 20
    group = _group((text,))
    build = _build((group,), budget=1000)
    _assert_literal(build, (group,), 1000)
    assert "calibration target 37" in build.fragments[0].text


def test_oversized_relevant_sentence_competes_with_short_weak_match():
    group = _group((
        "Calibration background information.",
        "Cobalt " + "unrelated " * 300 + "cobalt calibration target 37 degrees " + "unrelated " * 50,
    ))
    build = _build((group,), budget=1000)
    _assert_literal(build, (group,), 1000)
    assert any("calibration target 37" in fragment.text for fragment in build.fragments)


def test_equal_text_across_scopes_keeps_distinct_addresses_and_duplicate_hits_do_not_repeat():
    first = _group(("Cobalt calibration evidence.",), "scope-first")
    duplicate = PreparedMemoryCandidates(scope_id=first.scope_id, memory_ref=REF, hits=first.hits * 2)
    second = _group(("Cobalt calibration evidence.",), "scope-second")
    build = _build((duplicate, second), budget=1800)
    _assert_literal(build, (first, second), 1800)
    assert len(build.fragments) == 2
    assert {fragment.address.memory.scope_id for fragment in build.fragments} == {"scope-first", "scope-second"}


def test_empty_irrelevant_and_default_behavior_and_probe():
    builder = PreparedContextBuilder()
    group = _group(("Cobalt calibration evidence.",))
    default = builder.build_scopes_result(
        request=PrepareContextRequest(query="cobalt", max_bytes=1200),
        current_scope_id=group.scope_id,
        memory_candidates=(group,),
    )
    explicit = builder.build_scopes_result(
        request=PrepareContextRequest(query="cobalt", max_bytes=1200, evidence_selection=False),
        current_scope_id=group.scope_id,
        memory_candidates=(group,),
    )
    assert default == explicit and default.fragments == ()
    for groups in ((), (_group(("", "  ")),), (group,)):
        build = _build(groups, query="unrelated-query")
        assert build.context.status == "empty" and not build.fragments
    request = PrepareContextRequest(query="cobalt", max_bytes=1200, evidence_selection=True)
    build = _build((group,), query=request.query, budget=request.max_bytes, include_code=True)
    probe = builder.probe_budget(request=request, current_scope_id=group.scope_id, memory_candidates=(group,))
    assert probe.delivered_items == len(build.fragments)
    assert probe.unused_bytes == request.max_bytes - build.context.content_bytes


@pytest.mark.parametrize("value", (None, 1, 0, "true", "false"))
def test_selection_flag_is_strict(value):
    with pytest.raises(ValidationError):
        PrepareContextRequest(query="query", evidence_selection=value)
