"""Count admission must not lose a page when its durable request rolls back."""

from __future__ import annotations

import asyncio

import pytest

from powercontext.builtin.persistence.processing_intents import ArtifactProcessingIntentRepository
from powercontext.builtin.runtime import artifact_processing as processing
from powercontext.builtin.runtime.artifact_processing import ArtifactProcessingBinding, ArtifactProcessingSupervisor
from tests.builtin.runtime.test_processing_scheduler import _intent, _Launcher, _profile, _request


class _InterruptedRequest(RuntimeError):
    pass


class _FailingCountIntents(ArtifactProcessingIntentRepository):
    fail_once = True

    async def request(self, connection, scope_id, binding_name, /):
        row = await super().request(connection, scope_id, binding_name)
        if scope_id == "b" and self.fail_once:
            self.fail_once = False
            raise _InterruptedRequest
        return row


async def test_rolled_back_count_admission_keeps_requests_and_discovery_recoverable(tmp_path, monkeypatch):
    monkeypatch.setattr(processing, "_DISCOVERY_PAGE_SIZE", 2)
    async with _profile(tmp_path, "count-admission") as profile:
        for scope in ("a", "b", "c", "d", "e"):
            await _request(profile.database, "memory-binding", scope, requested=False)
        intents = _FailingCountIntents()
        launcher = _Launcher(profile.database)

        async def eligible(connection, scopes):
            return frozenset(scopes)

        supervisor = ArtifactProcessingSupervisor(
            database=profile.database,
            bindings=(
                ArtifactProcessingBinding(
                    "memory-binding", "memory", launcher, max_workers=2, automatic_count_filter=eligible
                ),
            ),
            lease_mode="single-process",
            intents=intents,
        )
        await supervisor._acquire()
        state = supervisor._families["memory-binding"]
        try:
            with pytest.raises(_InterruptedRequest):
                await supervisor._discover_count(state)
            assert state.count_after == 0
            assert not state.ready
            assert all([
                (await _intent(profile.database, "memory-binding", scope)).requested_generation == 0
                for scope in ("a", "b", "c", "d", "e")
            ])
            for _ in range(3):
                state.next_count_at = 0
                await supervisor._discover_count(state)
                await supervisor._dispatch(state)
                await asyncio.gather(*(running.task for running in state.running.values()))
                await supervisor._reap(state)
            assert [assignment.scope_id for assignment in launcher.assignments] == ["a", "b", "c", "d", "e"]
        finally:
            await supervisor.close()
