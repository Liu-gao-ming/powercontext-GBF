"""Embedded seekdb's native, single-statement hybrid candidate search."""

from __future__ import annotations

from pyobvector import VECTOR
from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncConnection
from sqlalchemy.sql.elements import TextClause

from powercontext.builtin.artifacts.memory import (
    CapabilityNotSupportedError,
    EmbeddingProfile,
    MemorySearchChannels,
    MemorySearchRequest,
)
from powercontext.builtin.artifacts.memory.canonical import validate_embedding
from powercontext.builtin.persistence.memory_index import CompositeMemoryIndex, memory_channel_hits
from powercontext.builtin.persistence.oceanbase.memory_index import OceanBaseMemoryFTSIndex, OceanBaseMemoryVectorIndex
from powercontext.builtin.persistence.tags import memory_tag_parameters, memory_tag_sql

_HYBRID_SQL = """
WITH lexical AS (
    SELECT f.memory_artifact_id, f.head_revision, f.entry_id, f.entry_version_id, v.text,
           MATCH(f.searchable_text) AGAINST (:query) AS relevance
    FROM pc_memory_entry_heads AS f
    JOIN pc_memory_entry_versions AS v
      ON v.scope_id = f.scope_id AND v.memory_artifact_id = f.memory_artifact_id
     AND v.entry_version_id = f.entry_version_id
    WHERE f.scope_id = :scope_id AND f.memory_artifact_id IN :memory_artifact_ids
      AND MATCH(f.searchable_text) AGAINST (:query)
      {fts_tags}
    ORDER BY relevance DESC, f.memory_artifact_id, f.entry_id, f.entry_version_id
    LIMIT :candidate_limit
), semantic AS (
    SELECT m.memory_artifact_id, m.head_revision, m.entry_id, m.entry_version_id, v.text,
           l2_distance(m.embedding, :query_vector) AS distance
    FROM pc_memory_vector_entries AS m
    JOIN pc_memory_entry_versions AS v
      ON v.scope_id = m.scope_id AND v.memory_artifact_id = m.memory_artifact_id
     AND v.entry_version_id = m.entry_version_id
    WHERE m.scope_id = :scope_id AND m.memory_artifact_id IN :memory_artifact_ids
      {vector_tags}
    ORDER BY l2_distance(m.embedding, :query_vector) APPROXIMATE
    LIMIT :candidate_limit
)
SELECT 'fts' AS channel, memory_artifact_id, head_revision, entry_id, entry_version_id, text,
       NULL AS distance, -relevance AS channel_order FROM lexical
UNION ALL
SELECT 'vector' AS channel, memory_artifact_id, head_revision, entry_id, entry_version_id, text,
       distance, distance AS channel_order FROM semantic
ORDER BY channel, channel_order, memory_artifact_id, entry_id, entry_version_id
"""


class SeekDBMemoryIndex(CompositeMemoryIndex):
    """Keep single-channel behavior and projection updates; combine only hybrid SQL."""

    def __init__(self, profile: EmbeddingProfile) -> None:
        self._vector = OceanBaseMemoryVectorIndex(profile)
        super().__init__(OceanBaseMemoryFTSIndex(), self._vector)
        self._hybrid = self._statement(tagged=False)
        self._tagged_hybrid = self._statement(tagged=True)

    def _statement(self, *, tagged: bool) -> TextClause:
        # Only static internal clauses are interpolated. All request values are bound.
        sql = _HYBRID_SQL.format(
            fts_tags=memory_tag_sql("f") if tagged else "",
            vector_tags=memory_tag_sql("m") if tagged else "",
        )
        parameters = [
            bindparam("memory_artifact_ids", expanding=True),
            bindparam("query_vector", type_=VECTOR(self._vector.profile.dimension)),
        ]
        if tagged:
            parameters.extend((bindparam("tag_keys", expanding=True), bindparam("tag_hashes", expanding=True)))
        return text(sql).bindparams(*parameters)

    async def search(
        self, connection: AsyncConnection, scope_id: str, request: MemorySearchRequest, /
    ) -> MemorySearchChannels:
        if request.mode != "hybrid":
            return await super().search(connection, scope_id, request)
        profile = self._vector.profile
        if request.embedding_profile != profile or request.query_vector is None:
            raise CapabilityNotSupportedError("embedding-profile")
        if not await self._vector.vector_complete(connection, scope_id, request.memories, profile):
            raise CapabilityNotSupportedError("vector")
        rows = (
            (
                await connection.execute(
                    self._hybrid if request.tag_filter is None else self._tagged_hybrid,
                    {
                        "scope_id": scope_id,
                        "memory_artifact_ids": tuple(memory.artifact_id for memory in request.memories),
                        "query": request.analyzed_query,
                        "query_vector": validate_embedding(request.query_vector, dimension=profile.dimension),
                        "candidate_limit": request.candidate_limit,
                        **memory_tag_parameters(request.tag_filter),
                    },
                )
            )
            .mappings()
            .all()
        )
        return MemorySearchChannels(
            fts=memory_channel_hits((row for row in rows if row["channel"] == "fts"), request.memories),
            vector=memory_channel_hits((row for row in rows if row["channel"] == "vector"), request.memories),
        )
