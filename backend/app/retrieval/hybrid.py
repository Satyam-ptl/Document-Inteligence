"""
Hybrid retrieval orchestrator (Phase 5): query embedding -> dense + sparse
Qdrant search -> Reciprocal Rank Fusion -> cross-encoder reranking.

This is the one function later phases (Phase 6 generation, the `/api/search`
debug endpoint, Phase 8 follow-up resolution) should actually call —
`app/retrieval/{qdrant_store,fusion,reranker}.py` are the pieces it wires
together, but none of them alone represents "search for this query" the
way `hybrid_search()` does.

Degrades gracefully rather than failing outright:
  - If the embedding model is unavailable (`EmbeddingUnavailableError` —
    see `app/embeddings/embedder.py`, still true in this sandbox per
    PROJECT_STATE.md), there is no way to search at all — this re-raises,
    since the caller (an API endpoint) needs to return a clear 503, not a
    silently empty result that looks like "no matches."
  - If the reranker is unavailable (`RerankerUnavailableError`), search
    still works — falls back to the fused (dense+sparse RRF) order,
    logging a warning. Reranking is a quality improvement, not a
    correctness requirement (see `reranker.py`'s docstring).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from loguru import logger
from qdrant_client import QdrantClient

from app.config import Settings
from app.embeddings.embedder import EmbeddingModel, embed_query
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.qdrant_store import search_dense, search_sparse
from app.retrieval.reranker import RerankedCandidate, RerankerModel, RerankerUnavailableError, rerank


@dataclass
class HybridSearchResult:
    chunk_id: str
    score: float  # rerank score if reranking ran, else the fused RRF score
    reranked: bool  # whether this result's score/order reflects a real reranker pass
    dense_rank: int | None
    sparse_rank: int | None
    text: str
    document_id: str
    page_number: int | None
    chunk_index: int
    element_type: str | None
    source_filename: str


def hybrid_search(
    query: str,
    qdrant_client: QdrantClient,
    embedding_model: EmbeddingModel,
    settings: Settings,
    document_id: str | None = None,
    reranker_model: RerankerModel | None = None,
) -> list[HybridSearchResult]:
    """Run one hybrid dense+sparse+fusion(+optional rerank) search.

    `reranker_model` is passed in (rather than looked up internally via
    `get_reranker_model()`) so callers control whether/how reranking is
    attempted — e.g. the router catches `RerankerUnavailableError` itself
    and calls this with `reranker_model=None` on a retry, and tests can
    inject a fake reranker the same way `test_phase4_embeddings.py`
    injects a fake embedding model.
    """
    dense_vector, sparse_weights = embed_query(query, model=embedding_model)

    dense_hits = search_dense(
        qdrant_client,
        settings.QDRANT_COLLECTION,
        dense_vector,
        top_k=settings.RETRIEVAL_TOP_K_CANDIDATES,
        document_id=document_id,
    )
    sparse_hits = search_sparse(
        qdrant_client,
        settings.QDRANT_COLLECTION,
        sparse_weights,
        top_k=settings.RETRIEVAL_TOP_K_CANDIDATES,
        document_id=document_id,
    )

    fused = reciprocal_rank_fusion(dense_hits, sparse_hits, k=settings.RRF_K)
    overview_query = _is_overview_query(query)
    if overview_query:
        # A generic overview has no meaningful query terms for the offline
        # lexical fallback. Preserve source order so the introduction wins
        # over an arbitrary technical chunk with hash-vector overlap.
        fused.sort(key=lambda result: (
            result.payload.get("page_number") if result.payload.get("page_number") is not None else 10**9,
            result.payload.get("chunk_index", 10**9),
        ))
    pool = fused[: settings.RETRIEVAL_RERANK_POOL]

    if not pool:
        return []

    if reranker_model is not None and settings.RERANK_ENABLED and not overview_query:
        try:
            reranked_candidates: list[RerankedCandidate] = rerank(
                query,
                [r.payload for r in pool],
                model=reranker_model,
                top_k=settings.RETRIEVAL_TOP_K_FINAL,
            )
            rank_by_chunk = {r.chunk_id: r for r in pool}
            return [
                _to_result(rank_by_chunk[c.chunk_id], score=c.rerank_score, reranked=True)
                for c in reranked_candidates
            ]
        except RerankerUnavailableError as exc:
            logger.warning(f"hybrid_search: reranker unavailable, falling back to fused order — {exc}")

    return [_to_result(r, score=r.fused_score, reranked=False) for r in pool[: settings.RETRIEVAL_TOP_K_FINAL]]


def _is_overview_query(query: str) -> bool:
    words = set(re.findall(r"[a-z0-9]+", query.lower()))
    meaningful = words - {"a", "an", "and", "are", "as", "about", "does", "document", "is", "me", "of", "tell", "the", "this", "what"}
    return not meaningful or bool({"summarize", "summary", "overview"} & words)


def _to_result(fused: Any, score: float, reranked: bool) -> HybridSearchResult:
    payload = fused.payload
    return HybridSearchResult(
        chunk_id=fused.chunk_id,
        score=score,
        reranked=reranked,
        dense_rank=fused.dense_rank,
        sparse_rank=fused.sparse_rank,
        text=payload.get("text", ""),
        document_id=payload.get("document_id", ""),
        page_number=payload.get("page_number"),
        chunk_index=payload.get("chunk_index", 0),
        element_type=payload.get("element_type"),
        source_filename=payload.get("source_filename", ""),
    )
