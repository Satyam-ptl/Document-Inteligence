"""
BAAI/bge-m3 embedding wrapper (Phase 4).

Design notes (read before touching this file) — deliberately mirrors
`app/ocr/paddle_ocr.py`'s Phase 3 pattern:

- The real `BGEM3FlagModel` (from the `FlagEmbedding` package) downloads
  model weights from HuggingFace Hub on first use, and `FlagEmbedding`
  itself depends on `torch`. In *this* build/dev sandbox neither works:
  `huggingface.co` is not on the outbound network allow-list (same gap
  documented for Phase 3's PaddleOCR-VL weights — see PROJECT_STATE.md),
  AND, separately, plain `pip install torch` could not even complete here
  for lack of disk space (`torch` plus its bundled NVIDIA CUDA libraries
  need >10 GB; this sandbox only had a few GB free). Both are environment
  limitations, reproduced directly in this session, not assumed — see
  PROJECT_STATE.md's Phase 4 verification log for the exact commands and
  errors.
- Because installing the real dependency wasn't even possible here, the
  real API shape below was NOT confirmed by importing/running the package
  (unlike Phase 3, which could at least install and import `paddleocr`).
  Instead it was confirmed by downloading the `FlagEmbedding` wheel with
  `pip download --no-deps` (no install, no torch needed) and reading
  `FlagEmbedding/inference/embedder/encoder_only/m3.py` directly. Confirmed
  from that source:
    * `from FlagEmbedding import BGEM3FlagModel` (aliases the real
      `M3Embedder` class — see that package's
      `inference/embedder/encoder_only/__init__.py`).
    * Constructor is `BGEM3FlagModel(model_name_or_path, use_fp16=bool,
      devices=str|list[str]|None, ...)` — note the parameter is `devices`
      (plural), not `device`; this was confirmed from source specifically
      to avoid shipping a wrong-kwarg bug that could only be caught by
      actually running it, which isn't possible in this sandbox.
    * `.encode(texts, batch_size=..., return_dense=True, return_sparse=...,
      return_colbert_vecs=...)` returns a dict with keys `"dense_vecs"`
      (np.ndarray, shape (n, 1024)), `"lexical_weights"` (list[dict[str,
      float]] — sparse/lexical weights, token -> weight), and
      `"colbert_vecs"` (list[np.ndarray]) — whichever of these were
      requested via the `return_*` flags.
- Because of that, exactly like Phase 3: the model object is created by a
  single factory function (`get_embedding_model`) and every call site goes
  through `embed_texts`, which takes the model as a parameter. Tests inject
  a fake model that mimics the real `.encode()` return shape (see
  `tests/test_phase4_embeddings.py`), so the chunk-batching /
  Qdrant-upsert / DB-bookkeeping logic is fully exercised without the real
  weights. Phase 5 (hybrid retrieval) should request `return_sparse=True`
  here too and thread `lexical_weights` through `EmbeddingResult.sparse` —
  the field already exists on this dataclass for that reason, just unused
  until Phase 5 wires a sparse index.
- Whoever runs this next in an environment with open network access AND
  enough disk should: `pip install FlagEmbedding torch`, confirm
  `get_embedding_model()` actually downloads weights, and re-verify
  `embed_texts()` against real output before trusting embedding quality in
  production — then delete this caveat.

Phase 5 update (hybrid retrieval): `embed_texts()` now requests
`return_sparse=True` as well and threads bge-m3's `lexical_weights` output
through `EmbeddingResult.sparse` (previously always `None`) — this is the
field the Phase 4 docstring above said would be wired here. Per-text sparse
output is a `dict[str, float]` of *token id* (as a string, per the
source-confirmed shape) -> lexical weight; `embed_query()` below is a thin
convenience wrapper for the common one-text case (embedding a user's
question at retrieval time) so callers don't have to unwrap a
single-element batch by hand. Nothing about the boundary/error-handling
pattern changes — same `EmbeddingUnavailableError`, same fake-at-the-model
boundary in tests, same reason (the real weights still aren't
installable/downloadable in this sandbox; see above).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Protocol

from loguru import logger

# bge-m3's dense embedding dimensionality (fixed by the model architecture,
# not configurable) — used to size the Qdrant collection in
# app/retrieval/qdrant_store.py. If EMBEDDING_MODEL is ever swapped for a
# different model with a different dimension, update
# Settings.EMBEDDING_DIMENSION in app/config.py to match.
BGE_M3_DIMENSION = 1024


class EmbeddingUnavailableError(RuntimeError):
    """Raised when the embedding model cannot be created or cannot run
    (e.g. model weights unreachable, package/torch not installed, an
    `encode()` crash mid-batch). Callers must catch this and mark the
    document 'failed' with a clear message rather than letting a
    background task die silently — exactly the `OCRUnavailableError`
    pattern from Phase 3."""


@dataclass
class EmbeddingResult:
    dense: list[list[float]]
    # Populated once Phase 5 wires hybrid (dense+sparse) retrieval by
    # requesting return_sparse=True from the same bge-m3 model call above —
    # left unused (None) here so Phase 4's Qdrant collection only needs a
    # single dense vector per point.
    sparse: list[dict[str, float]] | None = None


class EmbeddingModel(Protocol):
    """Structural type for whatever `.encode()` returns a dict with at
    least a `'dense_vecs'` key. Both the real `BGEM3FlagModel` and the
    fakes used in tests satisfy this without inheritance."""

    def encode(self, texts: list[str], **kwargs: Any) -> Any: ...


class _LocalHashEmbeddingModel:
    """Dependency-free fallback for development and offline deployments.

    This is a lexical hashing model, not a replacement for BGE-M3 semantic
    quality. It keeps ingestion and keyword retrieval usable when torch or
    model weights are unavailable, with the same dense dimension and sparse
    token contract expected by Qdrant.
    """

    _TOKEN_RE = re.compile(r"\w+", re.UNICODE)

    def __init__(self, dimension: int = BGE_M3_DIMENSION):
        self.dimension = dimension

    def encode(self, texts: list[str], **kwargs: Any) -> dict[str, Any]:
        dense_vectors: list[list[float]] = []
        lexical_weights: list[dict[str, float]] = []
        for text in texts:
            tokens = self._TOKEN_RE.findall(text.lower())
            counts: dict[int, float] = {}
            for token in tokens:
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest, "big") % self.dimension
                counts[index] = counts.get(index, 0.0) + 1.0
            norm = sum(value * value for value in counts.values()) ** 0.5 or 1.0
            dense = [0.0] * self.dimension
            for index, value in counts.items():
                dense[index] = value / norm
            dense_vectors.append(dense)
            lexical_weights.append({str(index): value / norm for index, value in counts.items()})
        return {"dense_vecs": dense_vectors, "lexical_weights": lexical_weights}


_model_singleton: EmbeddingModel | None = None


def get_embedding_model(
    model_name: str = "BAAI/bge-m3", device: str = "cpu", fallback_enabled: bool = True
) -> EmbeddingModel:
    """Lazily create (and cache) the real bge-m3 embedding model.

    Raises EmbeddingUnavailableError if the `FlagEmbedding` package (or its
    `torch` dependency) isn't installed, or the model can't be constructed
    (most commonly: model weights can't be downloaded, or torch itself
    couldn't be installed for lack of disk — see this module's docstring).
    """
    global _model_singleton
    if _model_singleton is not None:
        return _model_singleton

    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as exc:
        if fallback_enabled:
            logger.warning("FlagEmbedding is unavailable; using local hashing embeddings for this process.")
            _model_singleton = _LocalHashEmbeddingModel()
            return _model_singleton
        raise EmbeddingUnavailableError(
            "The 'FlagEmbedding' package (and its 'torch' dependency) is not "
            "installed. Run: pip install FlagEmbedding torch"
        ) from exc

    try:
        _model_singleton = BGEM3FlagModel(model_name, use_fp16=False, devices=device)
    except Exception as exc:  # noqa: BLE001 — this is a hard external boundary
        if fallback_enabled:
            logger.warning(f"Could not initialize {model_name}; using local hashing embeddings: {exc}")
            _model_singleton = _LocalHashEmbeddingModel()
            return _model_singleton
        raise EmbeddingUnavailableError(f"Could not initialize embedding model '{model_name}': {exc}") from exc

    return _model_singleton


def embed_texts(
    texts: list[str], model: EmbeddingModel, batch_size: int = 12, return_sparse: bool = True
) -> EmbeddingResult:
    """Embed a batch of chunk texts into dense (and, by default since Phase
    5, sparse/lexical) vectors.

    Raises EmbeddingUnavailableError if `model.encode()` itself raises (e.g.
    an OOM mid-batch, or the model losing its loaded state).
    """
    if not texts:
        return EmbeddingResult(dense=[], sparse=[] if return_sparse else None)

    try:
        output = model.encode(
            texts,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=return_sparse,
            return_colbert_vecs=False,
        )
    except Exception as exc:  # noqa: BLE001 — external model call
        raise EmbeddingUnavailableError(f"Embedding failed for a batch of {len(texts)} texts: {exc}") from exc

    dense_raw = output["dense_vecs"] if isinstance(output, dict) and "dense_vecs" in output else output
    dense = [[float(x) for x in vec] for vec in dense_raw]

    if any(len(vec) != BGE_M3_DIMENSION for vec in dense):
        logger.warning(
            f"embed_texts: expected {BGE_M3_DIMENSION}-dim vectors from bge-m3, "
            f"got a different shape — check EMBEDDING_MODEL / EMBEDDING_DIMENSION are in sync."
        )

    sparse: list[dict[str, float]] | None = None
    if return_sparse:
        lexical_raw = output.get("lexical_weights") if isinstance(output, dict) else None
        sparse = [
            {str(token): float(weight) for token, weight in (entry or {}).items()} for entry in (lexical_raw or [])
        ]
        if len(sparse) != len(texts):
            logger.warning(
                f"embed_texts: expected {len(texts)} sparse entries, got {len(sparse)} — "
                "hybrid search will fall back to dense-only ranking for any missing ones."
            )

    return EmbeddingResult(dense=dense, sparse=sparse)


def embed_query(text: str, model: EmbeddingModel) -> tuple[list[float], dict[str, float]]:
    """Convenience wrapper for the single-text query case (embedding a
    user's question at retrieval time). Returns `(dense_vector,
    sparse_weights)`. Always requests sparse output, since hybrid retrieval
    (Phase 5) needs both legs for every query, not just for chunk indexing.
    """
    result = embed_texts([text], model=model, batch_size=1, return_sparse=True)
    dense = result.dense[0] if result.dense else []
    sparse = result.sparse[0] if result.sparse else {}
    return dense, sparse
