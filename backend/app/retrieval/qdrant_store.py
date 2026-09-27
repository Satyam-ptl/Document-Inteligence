"""
Qdrant vector store wrapper (Phase 4).

Unlike the OCR (Phase 3) and embedding-model (this phase, `embedder.py`)
boundaries, this module's real dependency is NOT blocked in this sandbox:
`qdrant-client` ships an embedded, on-disk/in-memory mode
(`QdrantClient(path=...)` or `QdrantClient(":memory:")`) that needs no
server, no Docker daemon, and no network at all. So everything in this file
is genuinely exercised end-to-end in `tests/test_phase4_embeddings.py` — no
mocking needed here, only for the embedding model that feeds it vectors.

`QDRANT_MODE=local` (default, used by dev/tests) opens an on-disk collection
at `settings.QDRANT_LOCAL_PATH` via this local embedded mode.
`QDRANT_MODE=server` connects to a real Qdrant instance at `QDRANT_URL`
(docker-compose's `qdrant` service) — same client, same collection/point
API either way, so no other code needs to know which mode is active.
Switching modes has NOT been re-verified against a real server in this
sandbox (no network/Docker access — see PROJECT_STATE.md); the client
library is identical either way, but `Phase 11` (Docker Compose) should do
one real round-trip against the `server` mode before trusting it in
production.

Phase 5 update (hybrid retrieval): the collection now uses **named
vectors** instead of Phase 4's single unnamed dense vector — a `"dense"`
vector (unchanged 1024-dim bge-m3 dense embedding) plus a `"sparse"`
vector (bge-m3's lexical/sparse weights, stored as a Qdrant sparse
vector). This is a breaking schema change from Phase 4's collection shape;
nothing in this sandbox has real persisted Phase-4 data worth migrating
(dev-only local Qdrant, wiped freely in tests), but a real deployment
upgrading from a Phase-4 collection would need to recreate/reindex it —
noted in PROJECT_STATE.md's Phase 5 section. `search()` (dense-only, one
unnamed vector) is kept below for backward compatibility with any direct
caller, but now searches the `"dense"` named vector; `search_dense()` /
`search_sparse()` are the two legs `app/retrieval/fusion.py` and
`app/retrieval/hybrid.py` actually call for hybrid retrieval.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import Settings

_client_singleton: QdrantClient | None = None


def get_qdrant_client(settings: Settings) -> QdrantClient:
    """Lazily create (and cache) the Qdrant client. Local embedded mode by
    default (see module docstring); pass QDRANT_MODE=server + QDRANT_URL to
    talk to a real Qdrant instance instead."""
    global _client_singleton
    if _client_singleton is not None:
        return _client_singleton

    if settings.QDRANT_MODE == "local":
        _client_singleton = QdrantClient(path=str(settings.qdrant_local_path))
    else:
        _client_singleton = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)

    return _client_singleton


def reset_qdrant_client_cache() -> None:
    """Test-only helper: local-mode QdrantClient locks its on-disk directory,
    so tests that use a fresh temp path per test need to close/drop the
    cached singleton between tests rather than reusing one client forever."""
    global _client_singleton
    if _client_singleton is not None:
        _client_singleton.close()
    _client_singleton = None


DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


def ensure_collection(client: QdrantClient, collection_name: str, dimension: int) -> None:
    """Create the collection if it doesn't already exist. Idempotent — safe
    to call on every startup/ingestion run.

    Phase 5: creates named vectors for both dense (`DENSE_VECTOR_NAME`,
    cosine similarity) and sparse (`SPARSE_VECTOR_NAME`, bge-m3 lexical
    weights) retrieval legs, instead of Phase 4's single unnamed dense
    vector — see this module's docstring for why.
    """
    if client.collection_exists(collection_name):
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: qmodels.VectorParams(size=dimension, distance=qmodels.Distance.COSINE),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: qmodels.SparseVectorParams(index=qmodels.SparseIndexParams()),
        },
    )
    logger.info(
        f"Created Qdrant collection '{collection_name}' (dense dim={dimension}/cosine, sparse='{SPARSE_VECTOR_NAME}')."
    )


def _to_sparse_vector(weights: dict[str, float] | None) -> qmodels.SparseVector | None:
    """Convert bge-m3's `{token_id_str: weight}` sparse output into a Qdrant
    `SparseVector(indices=[...], values=[...])`. Token ids from
    `lexical_weights` are numeric-vocab-index strings (confirmed against
    the `FlagEmbedding` source in Phase 4 — see `embedder.py`), so they
    convert to ints directly; any entry that doesn't (defensively handled,
    should never happen against the real model) is skipped rather than
    raising, since a partially-missing sparse leg should degrade to
    dense-only ranking for that chunk, not fail the whole batch.
    """
    if not weights:
        return qmodels.SparseVector(indices=[], values=[])
    indices: list[int] = []
    values: list[float] = []
    for token, weight in weights.items():
        try:
            indices.append(int(token))
        except (TypeError, ValueError):
            logger.warning(f"_to_sparse_vector: skipping non-integer sparse token id {token!r}.")
            continue
        values.append(float(weight))
    return qmodels.SparseVector(indices=indices, values=values)


@dataclass
class ChunkVectorPayload:
    """One chunk's worth of data to index — deliberately mirrors the `Chunk`
    DB row's fields that retrieval/generation (Phase 5/6) need back out of a
    search hit, without requiring a DB round-trip just to show a source
    citation."""

    chunk_id: str
    document_id: str
    text: str
    page_number: int | None
    chunk_index: int
    element_type: str | None
    source_filename: str


def upsert_chunks(
    client: QdrantClient,
    collection_name: str,
    chunks: list[ChunkVectorPayload],
    vectors: list[list[float]],
    sparse_vectors: list[dict[str, float]] | None = None,
) -> list[str]:
    """Index a batch of chunks. Returns the Qdrant point IDs, in the same
    order as `chunks`/`vectors`, so the caller can write them back onto
    `Chunk.qdrant_point_id`.

    Qdrant point IDs must be a UUID or unsigned int — chunk DB ids are
    already UUID strings (see `Chunk.id` / `_uuid()` in
    `app/database/models.py`), so they're reused directly as point IDs
    rather than generating a second, unrelated ID to keep track of. This was
    confirmed the hard way: local-mode Qdrant genuinely rejects a
    non-UUID/non-int string id with `ValueError: Point id ... is not a
    valid UUID` (reproduced in this session while writing
    `tests/test_phase4_embeddings.py`, whose fixtures originally used
    plain strings like `"id-1"` — fixed there, noted here for whoever next
    touches this function).

    Phase 5: `sparse_vectors`, when given, must be the same length/order as
    `chunks`/`vectors`; each point then carries both a `DENSE_VECTOR_NAME`
    and a `SPARSE_VECTOR_NAME` vector so hybrid retrieval can query either
    leg. Omitting it (or passing `None`) writes dense-only points, still
    valid against the named-vector collection schema — sparse search would
    simply return nothing for those points, which is the correct fallback
    if the embedding model ever comes back without lexical weights for
    some reason.
    """
    if len(chunks) != len(vectors):
        raise ValueError(f"chunks ({len(chunks)}) and vectors ({len(vectors)}) length mismatch.")
    if sparse_vectors is not None and len(sparse_vectors) != len(chunks):
        raise ValueError(f"chunks ({len(chunks)}) and sparse_vectors ({len(sparse_vectors)}) length mismatch.")
    if not chunks:
        return []

    point_ids = [c.chunk_id for c in chunks]
    points = []
    for i, (point_id, c, vector) in enumerate(zip(point_ids, chunks, vectors, strict=True)):
        vector_payload: dict[str, Any] = {DENSE_VECTOR_NAME: vector}
        if sparse_vectors is not None:
            vector_payload[SPARSE_VECTOR_NAME] = _to_sparse_vector(sparse_vectors[i])
        points.append(
            qmodels.PointStruct(
                id=point_id,
                vector=vector_payload,
                payload={
                    "document_id": c.document_id,
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "page_number": c.page_number,
                    "chunk_index": c.chunk_index,
                    "element_type": c.element_type,
                    "source_filename": c.source_filename,
                },
            )
        )
    client.upsert(collection_name=collection_name, points=points)
    return point_ids


def delete_document_vectors(client: QdrantClient, collection_name: str, document_id: str) -> None:
    """Remove every indexed chunk belonging to a document — called from the
    document-delete endpoint so Qdrant never holds stale points for a
    document the user has removed."""
    if not client.collection_exists(collection_name):
        return
    client.delete(
        collection_name=collection_name,
        points_selector=qmodels.FilterSelector(
            filter=qmodels.Filter(
                must=[qmodels.FieldCondition(key="document_id", match=qmodels.MatchValue(value=document_id))]
            )
        ),
    )


def _document_filter(document_id: str | None) -> qmodels.Filter | None:
    if document_id is None:
        return None
    return qmodels.Filter(
        must=[qmodels.FieldCondition(key="document_id", match=qmodels.MatchValue(value=document_id))]
    )


def search(
    client: QdrantClient,
    collection_name: str,
    query_vector: list[float],
    top_k: int,
    document_id: str | None = None,
) -> list[dict[str, Any]]:
    """Dense-vector similarity search against the `DENSE_VECTOR_NAME` named
    vector. Returns a list of `{"score": float, **payload}` dicts, highest
    score first. `document_id` optionally scopes the search to one
    document — not used yet, but Phase 8 (conversation memory / follow-up
    resolution) will likely want it for "look only in the document we were
    just discussing" follow-ups.

    Kept as a plain dense-only search for any caller that just wants that
    (e.g. a quick smoke test); `search_dense`/`search_sparse` below are the
    two legs `app/retrieval/fusion.py` combines for real hybrid retrieval.
    """
    return search_dense(client, collection_name, query_vector, top_k, document_id=document_id)


def search_dense(
    client: QdrantClient,
    collection_name: str,
    query_vector: list[float],
    top_k: int,
    document_id: str | None = None,
) -> list[dict[str, Any]]:
    """Dense-only leg of hybrid retrieval — semantic similarity search
    against the `DENSE_VECTOR_NAME` named vector. Returns `[]` (rather than
    raising) if the collection doesn't exist yet — a real, reachable state
    (e.g. a fresh deployment, or `/api/chat` called before any document has
    ever been uploaded/indexed), not an error: `ensure_collection()` is only
    ever called from the ingestion path, so a query-only caller must not
    assume it's already been created. Found by actually exercising this
    path in `tests/test_phase6_generation.py`'s zero-chunks-indexed
    scenario (see PROJECT_STATE.md's Phase 6 -> Phase 7 handoff verification
    log) — local-mode Qdrant genuinely raises `ValueError: Collection ...
    not found` here otherwise."""
    if not client.collection_exists(collection_name):
        return []
    hits = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        using=DENSE_VECTOR_NAME,
        limit=top_k,
        query_filter=_document_filter(document_id),
    ).points
    return [{"score": hit.score, **(hit.payload or {})} for hit in hits]


def search_sparse(
    client: QdrantClient,
    collection_name: str,
    query_sparse: dict[str, float],
    top_k: int,
    document_id: str | None = None,
) -> list[dict[str, Any]]:
    """Sparse-only leg of hybrid retrieval — lexical/keyword-style search
    against the `SPARSE_VECTOR_NAME` named vector, using bge-m3's own
    lexical weights rather than a separate BM25 index
    (see `app/embeddings/embedder.py`'s Phase 5 update). Returns `[]`
    (rather than raising) for an empty/all-zero query, since a query with
    no lexical weight overlap is a legitimate outcome, not an error — the
    dense leg still covers that query in fusion.
    """
    sparse_vector = _to_sparse_vector(query_sparse)
    if not sparse_vector or not sparse_vector.indices:
        return []
    if not client.collection_exists(collection_name):
        return []
    hits = client.query_points(
        collection_name=collection_name,
        query=sparse_vector,
        using=SPARSE_VECTOR_NAME,
        limit=top_k,
        query_filter=_document_filter(document_id),
    ).points
    return [{"score": hit.score, **(hit.payload or {})} for hit in hits]


def new_point_id() -> str:
    """Not currently used by `upsert_chunks` (which reuses the chunk's own
    DB id as the point id — see its docstring), kept as a small utility for
    any future call site that needs an unrelated, freestanding point id."""
    return str(uuid.uuid4())
