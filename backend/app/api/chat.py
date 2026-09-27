"""
Chat endpoint (Phase 6, extended in Phase 7 and Phase 8): a thin HTTP
wrapper around `app/generation/generator.py`'s `generate_answer()` — the
same role `app/api/search.py` plays for `hybrid_search()` alone. Phase 7
added the `confidence`/`conflicts` fields to the response — both computed
inside `generate_answer()` itself, so this endpoint only has to serialize
them.

Phase 8 added opt-in multi-turn memory: if `body.conversation_id` is set,
this endpoint loads recent history (`app/conversations/memory.py`),
resolves the raw query into a standalone one if it looks like a follow-up,
runs generation against the *resolved* query, and records the turn. A
request that omits `conversation_id` skips all of this — no DB writes, no
extra LLM call — so single-turn callers keep Phase 6/7's exact original
cost and behavior.

Error-boundary pattern, extended one step past Phase 5's search endpoint:
`EmbeddingUnavailableError` and `LLMUnavailableError` both become a clear
503 (there is no way to answer at all without either a query vector or a
working LLM call — see `generator.py`'s docstring on why the LLM call has
no fallback the way the reranker does), while `RerankerUnavailableError`
is still handled exactly like Phase 5: soft-fail, log a warning, continue
without it, since reranking remains a quality improvement, not a
correctness requirement. Phase 8's rewrite call is even softer than that:
its own failure never raises at all, it just falls back to the raw query
(see `resolve_followup()`'s docstring).
"""

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.conversations.memory import (
    ResolvedQuery,
    get_or_create_conversation,
    get_recent_turns,
    record_turn,
    resolve_followup,
)
from app.database.session import get_db
from app.embeddings.embedder import EmbeddingUnavailableError, get_embedding_model
from app.generation.generator import generate_answer
from app.generation.llm_client import LLMUnavailableError, get_llm_client
from app.retrieval.qdrant_store import get_qdrant_client
from app.retrieval.reranker import RerankerUnavailableError, get_reranker_model
from app.schemas.schemas import (
    ChatRequest,
    ChatResponseOut,
    ConfidenceOut,
    ConflictPairOut,
    ConflictsOut,
    SourcePassageOut,
)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponseOut)
def chat(
    body: ChatRequest,
    settings: Settings = Depends(get_settings),
    db: Session = Depends(get_db),
) -> ChatResponseOut:
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
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Chat is unavailable: {exc}"
        ) from exc

    try:
        llm_client = get_llm_client(settings)
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Chat is unavailable: {exc}"
        ) from exc

    reranker_model = None
    if settings.RERANK_ENABLED:
        try:
            reranker_model = get_reranker_model(model_name=settings.RERANKER_MODEL, device=settings.RERANKER_DEVICE)
        except RerankerUnavailableError as exc:
            logger.warning(f"chat endpoint: reranker unavailable, continuing with fused-only order — {exc}")

    qdrant_client = get_qdrant_client(settings)

    # Phase 8: opt-in conversation memory. See this module's docstring —
    # a request without conversation_id does none of this.
    conversation = None
    resolved = ResolvedQuery(raw_query=body.query, resolved_query=body.query.strip(), rewritten=False)
    if body.conversation_id:
        conversation = get_or_create_conversation(db, body.conversation_id, document_id=body.document_id)
        history_turns = get_recent_turns(db, conversation.id, limit=settings.CONVERSATION_HISTORY_TURNS)
        resolved = resolve_followup(body.query, history_turns, llm_client=llm_client, settings=settings)

    try:
        result = generate_answer(
            resolved.resolved_query,
            qdrant_client=qdrant_client,
            embedding_model=embedding_model,
            llm_client=llm_client,
            settings=settings,
            document_id=body.document_id,
            reranker_model=reranker_model,
        )
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Chat is unavailable: {exc}"
        ) from exc

    if conversation is not None:
        record_turn(
            db, conversation, raw_query=body.query, resolved_query=resolved.resolved_query, answer=result.answer
        )

    return ChatResponseOut(
        query=body.query,
        answer=result.answer,
        insufficient_evidence=result.citations.is_insufficient_evidence,
        fully_cited=result.citations.fully_cited,
        invalid_citation_indices=result.citations.invalid_indices,
        sources=[
            SourcePassageOut(
                index=p.index,
                chunk_id=p.chunk_id,
                document_id=p.document_id,
                source_filename=p.source_filename,
                page_number=p.page_number,
                text=p.text,
                cited=p.index in result.citations.valid_indices,
            )
            for p in result.passages
        ],
        confidence=ConfidenceOut(
            level=result.confidence.level,
            score=result.confidence.score,
            reasons=result.confidence.reasons,
        ),
        conflicts=ConflictsOut(
            has_conflict=result.conflicts.has_conflict,
            conflicts=[
                ConflictPairOut(
                    passage_index_a=c.passage_index_a,
                    passage_index_b=c.passage_index_b,
                    text_a=c.text_a,
                    text_b=c.text_b,
                    sentence_a=c.sentence_a,
                    sentence_b=c.sentence_b,
                    shared_context=c.shared_context,
                )
                for c in result.conflicts.conflicts
            ],
        ),
        conversation_id=conversation.id if conversation is not None else None,
        resolved_query=resolved.resolved_query if resolved.rewritten else None,
    )
