"""
Reciprocal Rank Fusion (RRF) for combining dense + sparse retrieval legs
(Phase 5).

Deliberately implemented in plain Python rather than reached for via
Qdrant's native server-side fusion query (`qdrant_client.models.FusionQuery`
/ `Prefetch`). Two reasons:
  1. `search_dense()`/`search_sparse()` (`app/retrieval/qdrant_store.py`)
     already run as two independent, fully real, already-verified queries —
     fusing their results here is pure Python with no new external
     dependency or unverified client API surface to trust.
  2. RRF itself is a simple, well-understood formula
     (`score = sum(1 / (k + rank))` over every list a document appears in,
     1-indexed rank), so implementing it directly is both easier to test in
     isolation and easier to reason about than trusting a specific
     client-library version's fusion-query wiring sight-unseen.
Standard reference: Cormack, Clarke & Buettcher, "Reciprocal Rank Fusion
Outperforms Condorcet and Individual Rank Learning Methods" (SIGIR 2009).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FusedResult:
    chunk_id: str
    fused_score: float
    dense_rank: int | None  # 1-indexed rank in the dense leg, None if absent from it
    sparse_rank: int | None  # 1-indexed rank in the sparse leg, None if absent from it
    payload: dict[str, Any]  # the winning hit's full payload (text, page_number, etc.)


def reciprocal_rank_fusion(
    dense_hits: list[dict[str, Any]],
    sparse_hits: list[dict[str, Any]],
    k: int = 60,
) -> list[FusedResult]:
    """Combine two ranked hit lists (each already sorted best-first, as
    `search_dense`/`search_sparse` return them) into one fused ranking.

    Each hit dict is expected to be `{"score": float, "chunk_id": str, ...
    rest of the Qdrant payload}` (the shape `qdrant_store.search_dense` /
    `search_sparse` return). A chunk appearing in both lists accumulates
    both contributions, which is exactly RRF's point: a candidate that's
    merely decent on both dense and sparse similarity usually beats one
    that's #1 on only one of them — consistent with hybrid retrieval's
    purpose of catching both semantic and exact-keyword matches (e.g. a
    specific model number or code that a purely semantic embedding might
    blur past).

    Returns results sorted by fused score descending. Ties are broken by
    preferring the hit seen first in `dense_hits` (arbitrary but stable —
    fine, since an exact tie this deep in a fused ranking is not
    meaningfully distinguishable anyway).
    """
    scores: dict[str, float] = {}
    dense_ranks: dict[str, int] = {}
    sparse_ranks: dict[str, int] = {}
    payloads: dict[str, dict[str, Any]] = {}
    first_seen_order: dict[str, int] = {}
    order_counter = 0

    for rank, hit in enumerate(dense_hits, start=1):
        chunk_id = hit["chunk_id"]
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
        dense_ranks[chunk_id] = rank
        if chunk_id not in payloads:
            payloads[chunk_id] = hit
            first_seen_order[chunk_id] = order_counter
            order_counter += 1

    for rank, hit in enumerate(sparse_hits, start=1):
        chunk_id = hit["chunk_id"]
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
        sparse_ranks[chunk_id] = rank
        if chunk_id not in payloads:
            payloads[chunk_id] = hit
            first_seen_order[chunk_id] = order_counter
            order_counter += 1

    fused = [
        FusedResult(
            chunk_id=chunk_id,
            fused_score=score,
            dense_rank=dense_ranks.get(chunk_id),
            sparse_rank=sparse_ranks.get(chunk_id),
            payload=payloads[chunk_id],
        )
        for chunk_id, score in scores.items()
    ]
    fused.sort(key=lambda r: (-r.fused_score, first_seen_order[r.chunk_id]))
    return fused
