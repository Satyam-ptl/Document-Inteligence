"""
Confidence scoring (Phase 7).

Combines two independent reliability signals that already exist from
earlier phases into one number + level the frontend (Phase 9) and API
consumers can show without having to understand retrieval scores or
citation validation internals themselves:

  1. **Retrieval quality** — how strong was the match between the query and
     the top retrieved passage (`app/retrieval/hybrid.py`'s
     `HybridSearchResult.score`)? A weak top match means the answer is
     built on shaky evidence even if the LLM cited it correctly.
  2. **Citation quality** — did `app/generation/citation.py`'s validation
     find hallucinated citations or uncited sentences? An answer built on
     strong evidence but full of hallucinated `[7]` references is not
     trustworthy either.

Deliberately a **heuristic, not a calibrated probability**: there is no
labeled dataset in this project to fit thresholds against, and no
uncertainty-quantification model available in this sandbox (same
FlagEmbedding/torch gap documented for Phase 4/5 — see PROJECT_STATE.md).
The weights and thresholds below are reasonable, documented defaults, not
derived from any formal calibration — tune them once real usage data
exists (see "KNOWN GAPS" in PROJECT_STATE.md's Phase 7 section).

One subtlety this module exists specifically to handle correctly: the two
retrieval-scoring paths in `hybrid_search()` are on *completely different
numeric scales*, and treating them the same would silently under- or
over-report confidence:
  - **Reranked** (`HybridSearchResult.reranked=True`): `rerank()` calls
    `compute_score(pairs, normalize=True)`, and bge-reranker-v2-m3's
    `normalize=True` applies a sigmoid, so this score is already a
    genuine 0..1 relevance probability — usable directly.
  - **Fused-only** (reranker unavailable/disabled — see `reranker.py`'s
    graceful-degradation docstring): the score is a raw Reciprocal Rank
    Fusion score, `sum(1/(RRF_K + rank))` across the dense/sparse legs —
    with the project's default `RRF_K=60`, even a rank-1-in-both-legs hit
    scores only ~0.033, nowhere near "1.0 = certain." Comparing that
    against the same 0..1 thresholds as a reranked score would call almost
    every fused-only result "low confidence" regardless of true quality.
    `_normalize_retrieval_score()` rescales it against the theoretical max
    for a rank-1/rank-1 hit at the configured `RRF_K` so both paths feed
    roughly comparable numbers into the same thresholds — still a proxy,
    not a probability.

Pure Python, zero external dependency — same rigor as `citation.py`/
`prompt.py`: run directly against real scenarios this session, not just
written and assumed correct (see PROJECT_STATE.md's Phase 7 verification
log).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from app.generation.citation import CitationValidationResult

ConfidenceLevel = Literal["high", "medium", "low"]

# Heuristic weights: retrieval quality and citation quality are treated as
# equally important. Neither alone is sufficient — see module docstring.
_RETRIEVAL_WEIGHT = 0.5
_CITATION_WEIGHT = 0.5

# Heuristic level thresholds against the combined 0..1 score.
_HIGH_THRESHOLD = 0.66
_MEDIUM_THRESHOLD = 0.33


@dataclass
class ConfidenceResult:
    level: ConfidenceLevel
    score: float  # 0.0-1.0, higher is more confident. Heuristic, not a calibrated probability.
    reasons: list[str] = field(default_factory=list)  # short, human-readable explanations, for API/UI display


class _ScoredResult(Protocol):
    """Structural type for whatever carries a retrieval score — satisfied
    by `app/retrieval/hybrid.py`'s real `HybridSearchResult` and any test
    fake with the same two attributes."""

    score: float
    reranked: bool


def _normalize_retrieval_score(top_score: float, reranked: bool, rrf_k: int) -> float:
    """Rescale a top retrieval score onto a comparable 0..1 range — see
    module docstring for why reranked and fused-only scores can't be
    compared against the same thresholds directly."""
    if reranked:
        return max(0.0, min(1.0, top_score))
    max_possible = 2.0 / (rrf_k + 1)  # theoretical max: rank 1 in both the dense and sparse legs
    if max_possible <= 0:
        return 0.0
    return max(0.0, min(1.0, top_score / max_possible))


def _level_for(score: float) -> ConfidenceLevel:
    if score >= _HIGH_THRESHOLD:
        return "high"
    if score >= _MEDIUM_THRESHOLD:
        return "medium"
    return "low"


def compute_confidence(
    results: list[Any],
    citations: CitationValidationResult,
    rrf_k: int,
) -> ConfidenceResult:
    """Combine retrieval score quality + citation validation into one
    `ConfidenceResult`.

    `results` is `app/retrieval/hybrid.py`'s `hybrid_search()` output (or
    any list of objects with `.score`/`.reranked`) — the same list
    `generate_answer()` already has in hand before building the prompt.
    Only the top (index-0, i.e. best-ranked) result's score is used: it
    represents the strongest evidence the answer could have been grounded
    in, which is what a reader actually cares about when deciding whether
    to trust the answer.
    """
    if citations.is_insufficient_evidence:
        return ConfidenceResult(
            level="low",
            score=0.0,
            reasons=["the model reported the retrieved passages don't contain enough information to answer"],
        )

    if not results:
        return ConfidenceResult(level="low", score=0.0, reasons=["no passages were retrieved for this question"])

    reasons: list[str] = []
    top = results[0]
    retrieval_component = _normalize_retrieval_score(top.score, reranked=top.reranked, rrf_k=rrf_k)
    if not top.reranked:
        reasons.append("reranker was unavailable for this query; retrieval confidence uses fused-only scores")

    if citations.total_sentence_count == 0:
        citation_component = 0.0
        reasons.append("the answer had no sentences to validate citations against")
    else:
        invalid_fraction = (
            len(citations.invalid_indices) / len(citations.cited_indices) if citations.cited_indices else 0.0
        )
        uncited_fraction = citations.uncited_sentence_count / citations.total_sentence_count
        citation_component = max(0.0, 1.0 - invalid_fraction - uncited_fraction)
        if citations.invalid_indices:
            reasons.append(
                f"{len(citations.invalid_indices)} citation(s) referenced a passage number "
                "that was never given to the model"
            )
        if citations.uncited_sentence_count:
            reasons.append(
                f"{citations.uncited_sentence_count}/{citations.total_sentence_count} sentence(s) "
                "in the answer have no citation at all"
            )

    score = max(0.0, min(1.0, _RETRIEVAL_WEIGHT * retrieval_component + _CITATION_WEIGHT * citation_component))

    if not reasons:
        reasons.append("strong retrieval match and a fully-cited answer")

    return ConfidenceResult(level=_level_for(score), score=round(score, 4), reasons=reasons)
