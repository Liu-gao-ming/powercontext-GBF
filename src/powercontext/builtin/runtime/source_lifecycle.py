"""Durable evidence DAGs and recoverable native Memory availability changes."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import insert, select, update

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory import MemoryCitation
from powercontext.builtin.persistence.tables import SOURCE_LIFECYCLE_NODES_TABLE
from powercontext.builtin.sources import validate_scope_id
from powercontext.sources import SourceRef

if TYPE_CHECKING:
    from powercontext.builtin.persistence.database import AsyncDatabase
    from powercontext.builtin.runtime.application import BuiltinRuntime


class SourceLifecycleError(ValueError):
    """A lifecycle binding, dependency, event, or use is invalid."""


class SourceLifecycleEvent(BaseModel):
    """One exact, replayable event on a registered Source lifecycle version."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    node_id: str
    revision: int = Field(gt=0)
    sequence: int = Field(gt=0)
    available: bool
    reason: str

    @field_validator("node_id", "reason")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("lifecycle identifiers and reasons must not be blank")  # noqa: TRY003
        return value


class _Node(BaseModel):
    model_config = ConfigDict(extra="forbid")
    node_id: str
    kind: Literal["source", "memory", "topic"]
    reference: SourceRef | MemoryCitation | ArtifactRef
    parents: tuple[str, ...] = ()
    policy: Literal["all", "any"] = "all"
    revision: int = 1
    available: bool = True
    applied: bool = False
    event: SourceLifecycleEvent | None = None

    def binding(self) -> tuple[object, ...]:
        return self.kind, self.reference, self.parents, self.policy, self.revision


class SourceLifecycleApplication:
    """Select a lifecycle service without retaining process-local graph state."""

    def __init__(self, runtime: BuiltinRuntime, database: AsyncDatabase | None) -> None:
        self._runtime = runtime
        self._database = database

    def for_scope(self, scope_id: str) -> ScopedSourceLifecycle:
        return ScopedSourceLifecycle(self._runtime, self._database, validate_scope_id(scope_id))


class ScopedSourceLifecycle:
    def __init__(self, runtime: BuiltinRuntime, database: AsyncDatabase | None, scope_id: str) -> None:
        self._runtime = runtime
        self._database = database
        self.scope_id = scope_id

    async def register_source(self, node_id: str, source_ref: SourceRef, *, revision: int = 1) -> None:
        if type(revision) is not int or revision < 1 or not isinstance(source_ref, SourceRef):
            raise SourceLifecycleError("invalid source lifecycle binding")  # noqa: TRY003
        await self._register(_Node(node_id=_node_id(node_id), kind="source", reference=source_ref, revision=revision))

    async def register_memory(
        self, node_id: str, citation: MemoryCitation, *, parents: tuple[str, ...], policy: Literal["all", "any"] = "all"
    ) -> None:
        if not isinstance(citation, MemoryCitation):
            raise SourceLifecycleError("invalid Memory citation")  # noqa: TRY003
        await self._register(self._derivative(node_id, "memory", citation, parents, policy))

    async def register_topic(
        self,
        node_id: str,
        artifact_ref: ArtifactRef,
        *,
        parents: tuple[str, ...],
        policy: Literal["all", "any"] = "all",
    ) -> None:
        if not isinstance(artifact_ref, ArtifactRef) or artifact_ref.family != "topic-memory":
            raise SourceLifecycleError("invalid Topic reference")  # noqa: TRY003
        await self._register(self._derivative(node_id, "topic", artifact_ref, parents, policy))

    @staticmethod
    def _derivative(
        node_id: str,
        kind: Literal["memory", "topic"],
        reference: MemoryCitation | ArtifactRef,
        parents: tuple[str, ...],
        policy: Literal["all", "any"],
    ) -> _Node:
        node_id = _node_id(node_id)
        if policy not in ("all", "any") or not isinstance(parents, (tuple, list)) or not parents:
            raise SourceLifecycleError("invalid parent policy")  # noqa: TRY003
        if any(not isinstance(parent, str) or not parent.strip() for parent in parents):
            raise SourceLifecycleError("invalid parent identifier")  # noqa: TRY003
        if len(set(parents)) != len(parents) or node_id in parents:
            raise SourceLifecycleError("duplicate or cyclic dependency")  # noqa: TRY003
        return _Node(node_id=node_id, kind=kind, reference=reference, parents=tuple(parents), policy=policy)

    async def _register(self, node: _Node) -> None:
        async with self._runtime._scope_operation(self.scope_id), self._runtime._locked(self.scope_id):
            nodes = await self._load()
            previous = nodes.get(node.node_id)
            if previous is not None:
                if previous.binding() != node.binding():
                    raise SourceLifecycleError("node is already bound differently")  # noqa: TRY003
                await self._synchronize(nodes)
                return
            if any(parent not in nodes for parent in node.parents):
                raise SourceLifecycleError("parents must already be registered in this scope")  # noqa: TRY003
            # Two policies cannot independently control the same physical body.
            if node.kind != "source" and any(
                item.kind == node.kind and item.reference == node.reference for item in nodes.values()
            ):
                raise SourceLifecycleError("derived body is already registered")  # noqa: TRY003
            await self._validate_reference(node)
            nodes[node.node_id] = node
            self._propagate(nodes)
            await self._save(node, new=True)
            await self._synchronize(nodes)

    async def apply(self, event: SourceLifecycleEvent) -> list[str]:
        if not isinstance(event, SourceLifecycleEvent):
            raise SourceLifecycleError("expected a validated lifecycle event")  # noqa: TRY003
        async with self._runtime._scope_operation(self.scope_id), self._runtime._locked(self.scope_id):
            nodes = await self._load()
            source = nodes.get(event.node_id)
            if source is None or source.kind != "source" or source.revision != event.revision:
                raise SourceLifecycleError("unknown source or lifecycle revision mismatch")  # noqa: TRY003
            previous = source.event
            if previous is not None and (
                event.sequence < previous.sequence or (event.sequence == previous.sequence and event != previous)
            ):
                raise SourceLifecycleError("stale or conflicting lifecycle event")  # noqa: TRY003
            if previous != event:
                if any(not node.applied for node in nodes.values()):
                    await self._synchronize(nodes)
                source.event = event
                source.available = event.available
                self._propagate(nodes)
                # Persist intent and every target state atomically before native writes.
                await self._save_all(nodes)
            await self._synchronize(nodes)
            return sorted(node.node_id for node in nodes.values() if node.available)

    async def resolve(
        self, node_id: str, *, purpose: Literal["generation", "audit"] = "generation"
    ) -> SourceRef | MemoryCitation | ArtifactRef:
        node_id = _node_id(node_id)
        if purpose not in ("generation", "audit"):
            raise SourceLifecycleError("unknown lifecycle purpose")  # noqa: TRY003
        async with self._runtime._scope_operation(self.scope_id):
            nodes = await self._load()
            node = nodes.get(node_id)
            if node is None or (purpose == "generation" and (not node.available or not node.applied)):
                raise SourceLifecycleError("node is unknown or unavailable for generation")  # noqa: TRY003
            return node.reference

    @staticmethod
    def _propagate(nodes: dict[str, _Node]) -> None:
        remaining = set(nodes)
        evaluated: set[str] = set()
        while remaining:
            ready = [key for key in sorted(remaining) if all(parent in evaluated for parent in nodes[key].parents)]
            if not ready:
                raise SourceLifecycleError("cyclic lifecycle graph")  # noqa: TRY003
            for key in ready:
                node = nodes[key]
                if node.parents:
                    support = [nodes[parent].available for parent in node.parents]
                    available = all(support) if node.policy == "all" else any(support)
                    if available != node.available:
                        node.available = available
                        node.applied = False
                evaluated.add(key)
                remaining.remove(key)

    async def _validate_reference(self, node: _Node) -> None:
        ref = node.reference
        try:
            if isinstance(ref, SourceRef):
                await self._runtime._records().get_source(self.scope_id, ref.source_type, ref.source_id)
            elif isinstance(ref, MemoryCitation):
                async with self._runtime._context(self.scope_id) as context:
                    await context.artifacts.memory.validate_citation(ref)
            else:
                await self._runtime._records().get_artifact_revision(
                    self.scope_id, ref.family, ref.artifact_id, ref.revision
                )
        except Exception as error:
            raise SourceLifecycleError("lifecycle reference cannot be read in this scope") from error  # noqa: TRY003

    async def _synchronize(self, nodes: dict[str, _Node]) -> None:
        for node in nodes.values():
            if node.applied:
                continue
            if isinstance(node.reference, MemoryCitation):
                async with self._runtime._context(self.scope_id) as context:
                    service = context.artifacts.memory
                    ref = node.reference
                    head, entries = await service.head_entries(ref.memory_ref.artifact_id)
                    entry = next((item for item in entries if item.entry_id == ref.entry_id), None)
                    if entry is None or entry.entry_version_id != ref.entry_version_id:
                        raise SourceLifecycleError("bound Memory body changed")  # noqa: TRY003
                    operation = service.reactivate if node.available else service.forget
                    await operation(head, entries=(entry,), reason="source lifecycle support")
            # A crash after the native commit is safe: state transitions are idempotent.
            node.applied = True
            await self._save(node)

    def _database_owner(self) -> AsyncDatabase:
        if self._database is None:
            raise SourceLifecycleError("lifecycle requires a persistent Runtime")  # noqa: TRY003
        return self._database

    async def _load(self) -> dict[str, _Node]:
        table = SOURCE_LIFECYCLE_NODES_TABLE
        async with self._database_owner().transaction() as connection:
            values = await connection.scalars(select(table.c.payload).where(table.c.scope_id == self.scope_id))
            nodes = [_Node.model_validate_json(bytes(value)) for value in values]
        return {node.node_id: node for node in nodes}

    def _values(self, node: _Node) -> dict[str, object]:
        topic = node.reference if node.kind == "topic" else None
        return {
            "scope_id": self.scope_id,
            "node_key": hashlib.sha256(node.node_id.encode()).hexdigest(),
            "payload": node.model_dump_json().encode(),
            "available": node.available,
            "topic_artifact_id": topic.artifact_id if isinstance(topic, ArtifactRef) else None,
            "topic_revision": topic.revision if isinstance(topic, ArtifactRef) else None,
        }

    async def _save(self, node: _Node, *, new: bool = False) -> None:
        table = SOURCE_LIFECYCLE_NODES_TABLE
        values = self._values(node)
        async with self._database_owner().transaction() as connection:
            statement = (
                insert(table)
                if new
                else update(table).where(table.c.scope_id == self.scope_id, table.c.node_key == values["node_key"])
            )
            await connection.execute(statement.values(**values))

    async def _save_all(self, nodes: dict[str, _Node]) -> None:
        table = SOURCE_LIFECYCLE_NODES_TABLE
        async with self._database_owner().transaction() as connection:
            for node in nodes.values():
                values = self._values(node)
                await connection.execute(
                    update(table)
                    .where(table.c.scope_id == self.scope_id, table.c.node_key == values["node_key"])
                    .values(**values)
                )


def _node_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourceLifecycleError("node_id must be a nonblank string")  # noqa: TRY003
    return value
