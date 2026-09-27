"""
Grounded answer generation orchestrator (Phase 6, extended in Phase 7): the
one function later phases (Phase 8 conversation memory, the `/api/chat`
endpoint) should actually call — the same role `app/retrieval/hybrid.py`'s
`hybrid_search()` plays for retrieval alone.

Pipeline: `hybrid_search()` -> number the results into `ContextPassage`s
and build the prompt (`app/generation/prompt.py`) -> call the configured
LLM (`app/generation/llm_client.py`) -> validate citations against the
real passage count (`app/generation/citation.py`) -> (Phase 7) score
confidence (`app/reliability/confidence.py`) and detect cross-passage
numeric conflicts (`app/reliability/conflict.py`) -> return one
`GeneratedAnswer` carrying the answer text, every passage that was
actually offered to the LLM (so Phase 9's evidence viewer can show sources
regardless of which ones the LLM chose to cite), the citation validation
result, a confidence score/level, and any detected conflicts.

One deliberate short-circuit, to avoid ever sending a context-free,
hallucination-inviting prompt to the LLM: zero retrieved chunks means
there is nothing to answer from, so this returns a fixed
`INSUFFICIENT_EVIDENCE:` answer directly, without calling the LLM at all.
This is not just an optimization (saving an API call) — it removes any
chance of the model quietly answering from its own outside knowledge
when `hybrid_search()` found nothing, which would silently break the
project's "grounded answer" requirement. `document_id`-scoped searches
that come up empty get a message that names the scoping, since "nothing
in this one document" is a different, more actionable outcome for the
user than "nothing in the whole corpus."
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger
from qdrant_client import QdrantClient

from app.config import Settings
from app.embeddings.embedder import EmbeddingModel
from app.generation.citation import CitationValidationResult, validate_citations
from app.generation.llm_client import LLMClient
from app.generation.prompt import SYSTEM_PROMPT, ContextPassage, build_user_prompt, to_context_passages
from app.reliability.confidence import ConfidenceResult, compute_confidence
from app.reliability.conflict import ConflictDetectionResult, detect_conflicts
from app.retrieval.hybrid import HybridSearchResult, hybrid_search
from app.retrieval.reranker import RerankerModel

_NO_EVIDENCE_MESSAGE = (
    "INSUFFICIENT_EVIDENCE: No relevant passages were found in the document corpus for this question."
)
_NO_EVIDENCE_MESSAGE_SCOPED = (
    "INSUFFICIENT_EVIDENCE: No relevant passages were found in the specified document for this question."
)


@dataclass
class GeneratedAnswer:
    query: str
    answer: str
    passages: list[ContextPassage]  # every passage actually offered to the LLM, 1-indexed; [] on the no-evidence path
    citations: CitationValidationResult
    llm_called: bool  # False only for the zero-evidence short-circuit — no LLM request was made
    confidence: ConfidenceResult  # Phase 7: combined retrieval+citation heuristic — see app/reliability/confidence.py
    conflicts: ConflictDetectionResult  # Phase 7: cross-passage numeric contradictions — see app/reliability/conflict.py


def generate_answer(
    query: str,
    qdrant_client: QdrantClient,
    embedding_model: EmbeddingModel,
    llm_client: LLMClient,
    settings: Settings,
    document_id: str | None = None,
    reranker_model: RerankerModel | None = None,
) -> GeneratedAnswer:
    results: list[HybridSearchResult] = hybrid_search(
        query,
        qdrant_client=qdrant_client,
        embedding_model=embedding_model,
        settings=settings,
        document_id=document_id,
        reranker_model=reranker_model,
    )

    if not results:
        message = _NO_EVIDENCE_MESSAGE_SCOPED if document_id else _NO_EVIDENCE_MESSAGE
        logger.info(f"generate_answer: no retrieved chunks for query={query!r} document_id={document_id!r}")
        no_evidence_citations = validate_citations(message, num_passages=0)
        return GeneratedAnswer(
            query=query,
            answer=message,
            passages=[],
            citations=no_evidence_citations,
            llm_called=False,
            confidence=compute_confidence(results, no_evidence_citations, rrf_k=settings.RRF_K),
            conflicts=ConflictDetectionResult(has_conflict=False, conflicts=[]),
        )

    passages = to_context_passages(results)
    user_prompt = build_user_prompt(query, passages)

    # No try/except around this call: LLMUnavailableError is deliberately
    # left to propagate to the caller (app/api/chat.py), which turns it
    # into a 503 — unlike the reranker, there is no fallback path for a
    # missing LLM (see this module's docstring).
    answer = llm_client.complete(
        system=SYSTEM_PROMPT,
        user=user_prompt,
        max_tokens=settings.LLM_MAX_TOKENS,
        temperature=settings.LLM_TEMPERATURE,
    )

    citations = validate_citations(answer, num_passages=len(passages))
    if citations.invalid_indices:
        logger.warning(
            f"generate_answer: LLM cited passage number(s) {citations.invalid_indices} that don't exist "
            f"among the {len(passages)} passages given for query={query!r} — this answer's citations "
            "include at least one hallucinated reference; downstream consumers should not treat every "
            "cited claim in it as attributed."
        )
    if citations.uncited_sentence_count and not citations.is_insufficient_evidence:
        logger.warning(
            f"generate_answer: {citations.uncited_sentence_count}/{citations.total_sentence_count} "
            f"sentence(s) in the answer for query={query!r} have no citation marker at all."
        )

    conflicts = detect_conflicts(passages)
    if conflicts.has_conflict:
        logger.warning(
            f"generate_answer: {len(conflicts.conflicts)} potential cross-passage numeric conflict(s) "
            f"detected among retrieved passages for query={query!r} — the answer may be built on "
            "contradictory evidence; see ConflictDetectionResult for details."
        )

    return GeneratedAnswer(
        query=query,
        answer=answer,
        passages=passages,
        citations=citations,
        llm_called=True,
        confidence=compute_confidence(results, citations, rrf_k=settings.RRF_K),
        conflicts=conflicts,
    )
