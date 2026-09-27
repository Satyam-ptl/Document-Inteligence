"""
Search endpoint (Phase 5): a thin HTTP wrapper around
`app/retrieval/hybrid.py`'s `hybrid_search()`, so hybrid retrieval + fusion
+ reranking can be exercised and inspected directly — the same role Phase
2's `/pages` and `/chunks` debug endpoints played for ingestion — before
Phase 6 wires an LLM on top of it for actual grounded answer generation.

Mirrors Phase 3/4's error-boundary pattern: `EmbeddingUnavailableError`
(there's no way to search at all without a query vector) becomes a clear
503, not a silently-empty result list that looks like "no matches found."
A `RerankerUnavailableError` does NOT fail the request — `hybrid_search()`
already catches that internally and falls back to the fused order (see its
docstring) — but this endpoint still needs to know whether that fallback
happened (`reranker_model=None` vs `RerankerUnavailableError` caught before
`hybrid_search` is even called), which is why the reranker is looked up
once here, defensively, before the search call.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from app.config import Settings, get_settings
from app.embeddings.embedder import EmbeddingUnavailableError, get_embedding_model
from app.retrieval.hybrid import hybrid_search
from app.retrieval.qdrant_store import get_qdrant_client
from app.retrieval.reranker import RerankerUnavailableError, get_reranker_model
from app.schemas.schemas import SearchRequest, SearchResponseOut, SearchResultOut

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponseOut)
def search(
    body: SearchRequest,
    settings: Settings = Depends(get_settings),
) -> SearchResponseOut:
    if not body.query or not body.query.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="query must not be empty.")

    try:
        embedding_model = get_embedding_model(
            model_name=settings.EMBEDDING_MODEL,
            device=settings.EMBEDDING_DEVICE,
            fallback_enabled=settings.EMBEDDING_FALLBACK_ENABLED,
        )
    except EmbeddingUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Search is unavailable: {exc}"
        ) from exc

    reranker_model = None
    if settings.RERANK_ENABLED:
        try:
            reranker_model = get_reranker_model(model_name=settings.RERANKER_MODEL, device=settings.RERANKER_DEVICE)
        except RerankerUnavailableError as exc:
            logger.warning(f"search endpoint: reranker unavailable, continuing with fused-only order — {exc}")

    qdrant_client = get_qdrant_client(settings)

    results = hybrid_search(
        body.query,
        qdrant_client=qdrant_client,
        embedding_model=embedding_model,
        settings=settings,
        document_id=body.document_id,
        reranker_model=reranker_model,
    )

    return SearchResponseOut(
        query=body.query,
        reranker_used=reranker_model is not None and any(r.reranked for r in results),
        results=[
            SearchResultOut(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                score=r.score,
                reranked=r.reranked,
                dense_rank=r.dense_rank,
                sparse_rank=r.sparse_rank,
                text=r.text,
                page_number=r.page_number,
                chunk_index=r.chunk_index,
                element_type=r.element_type,
                source_filename=r.source_filename,
            )
            for r in results
        ],
    )
