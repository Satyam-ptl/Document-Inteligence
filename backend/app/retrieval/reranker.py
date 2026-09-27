"""
BAAI/bge-reranker-v2-m3 cross-encoder reranking wrapper (Phase 5).

Design notes — mirrors `app/ocr/paddle_ocr.py` (Phase 3) and
`app/embeddings/embedder.py` (Phase 4)'s pattern deliberately, per
PROJECT_STATE.md's instruction to follow the same boundary pattern rather
than re-deriving it: a single lazy-singleton factory
(`get_reranker_model`), a plain function that takes the model as a
parameter (`rerank`), and one exception type
(`RerankerUnavailableError`) raised on any external-boundary failure so the
caller can degrade cleanly instead of crashing.

**Important gap, different from Phase 3/4's:** Phase 4's docstring
recommended confirming a heavy package's real API shape via `pip download
<package> --no-deps` (fetch the wheel, read the source, no install/network-
heavy-download needed) *without* running the model — and that trick worked
for `FlagEmbedding`/`BGEM3FlagModel` because network access existed at all
in that session, just not enough disk. **This session's sandbox has no
outbound network access whatsoever** (not a disk-space gap this time — see
this repo's network configuration), so even `pip download` was not
possible here. The API shape below is therefore written from documented,
widely-published `FlagEmbedding` API knowledge (its public README / PyPI
page describe `FlagReranker`/`FlagAutoReranker` with a
`compute_score(pairs, normalize=...)` method), **not** confirmed against
the actual installed package source the way every prior phase's external
API shape was. Treat this as the one part of Phase 5 that still needs the
Phase 4-style verification pass (`pip download FlagEmbedding --no-deps -d
somewhere && unzip` and read the reranker source directly, then a real
`.compute_score()` call) before fully trusting it — everything else in
this module (the boundary pattern, error handling, and the code that calls
this module) is written and tested the same rigorous way as every other
phase.

Documented real API (subject to the above caveat):
  ```python
  from FlagEmbedding import FlagAutoReranker
  reranker = FlagAutoReranker.from_finetuned(
      model_name_or_path, use_fp16=True, devices=device,
  )
  scores = reranker.compute_score(pairs, normalize=True)  # pairs: list[[query, passage]]
  ```
`compute_score` returns a single float for one pair or a `list[float]` for
several — normalized into a `list[float]` here either way so callers never
special-case the single-candidate case.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from loguru import logger


class RerankerUnavailableError(RuntimeError):
    """Raised when the reranker model cannot be created or cannot run
    (package not installed, model init failure, a `compute_score()` crash
    mid-batch). Callers must catch this and degrade to the pre-rerank
    fused order rather than failing the whole search — reranking is a
    quality improvement, not a correctness requirement, so an unavailable
    reranker should never turn a working search into a broken one."""


@dataclass
class RerankedCandidate:
    chunk_id: str
    rerank_score: float
    payload: dict[str, Any]


class RerankerModel(Protocol):
    """Structural type for whatever exposes `.compute_score(pairs, **kwargs)`.
    Both the real `FlagAutoReranker`/`FlagReranker` and the fakes used in
    tests satisfy this without inheritance."""

    def compute_score(self, pairs: list[list[str]], **kwargs: Any) -> Any: ...


_reranker_singleton: RerankerModel | None = None


def get_reranker_model(model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cpu") -> RerankerModel:
    """Lazily create (and cache) the real bge-reranker-v2-m3 cross-encoder.

    Raises RerankerUnavailableError if the `FlagEmbedding` package (or its
    `torch` dependency) isn't installed, or the model can't be constructed
    — the exact same environment gap Phase 4 hit for the embedding model
    (`FlagEmbedding` reuses the identical underlying dependency, so
    whatever fixes one fixes the other; see this module's docstring for
    what's different about *confirming* this one's API shape).
    """
    global _reranker_singleton
    if _reranker_singleton is not None:
        return _reranker_singleton

    try:
        from FlagEmbedding import FlagAutoReranker
    except ImportError as exc:
        raise RerankerUnavailableError(
            "The 'FlagEmbedding' package (and its 'torch' dependency) is not "
            "installed. Run: pip install FlagEmbedding torch"
        ) from exc

    try:
        _reranker_singleton = FlagAutoReranker.from_finetuned(model_name, use_fp16=False, devices=device)
    except Exception as exc:  # noqa: BLE001 — hard external boundary
        raise RerankerUnavailableError(f"Could not initialize reranker model '{model_name}': {exc}") from exc

    return _reranker_singleton


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    model: RerankerModel,
    top_k: int | None = None,
) -> list[RerankedCandidate]:
    """Re-score a pool of fused candidates against the query with a
    cross-encoder, returning them sorted by rerank score descending
    (highest-quality match first), optionally truncated to `top_k`.

    Each candidate dict is expected to have at least `"chunk_id"` and
    `"text"` keys (the shape `app/retrieval/fusion.py`'s `FusedResult.payload`
    carries). Raises RerankerUnavailableError if `model.compute_score()`
    itself raises — callers (see `app/retrieval/hybrid.py`) are expected to
    catch this and fall back to the pre-rerank fused order rather than
    failing the search outright.
    """
    if not candidates:
        return []

    pairs = [[query, c["text"]] for c in candidates]
    try:
        raw_scores = model.compute_score(pairs, normalize=True)
    except Exception as exc:  # noqa: BLE001 — external model call
        raise RerankerUnavailableError(f"Reranking failed for {len(candidates)} candidates: {exc}") from exc

    scores = [float(raw_scores)] if isinstance(raw_scores, (int, float)) else [float(s) for s in raw_scores]
    if len(scores) != len(candidates):
        logger.warning(
            f"rerank: expected {len(candidates)} scores back from compute_score, got {len(scores)} — "
            "truncating/padding defensively rather than crashing the search."
        )
        scores = (scores + [0.0] * len(candidates))[: len(candidates)]

    reranked = [
        RerankedCandidate(chunk_id=c["chunk_id"], rerank_score=score, payload=c)
        for c, score in zip(candidates, scores, strict=True)
    ]
    reranked.sort(key=lambda r: -r.rerank_score)
    return reranked[:top_k] if top_k is not None else reranked
