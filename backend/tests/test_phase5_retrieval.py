"""
Phase 5 tests: hybrid (dense+sparse) retrieval, Reciprocal Rank Fusion, and
reranking.

Three different boundaries here, tested differently on purpose (same
approach Phase 4's test file used, see its own module docstring):

- `app/retrieval/fusion.py`: pure Python, no external dependency at all —
  tested directly, for real, no mocking needed.
- `app/retrieval/qdrant_store.py`'s named dense+sparse vectors: the real
  dependency (`qdrant-client`'s embedded/local mode) is NOT blocked in this
  sandbox (see Phase 4's test file) — tested against a real, local, on-disk
  Qdrant collection, no mocking.
- `app/embeddings/embedder.py` (sparse output) and `app/retrieval/reranker.py`:
  both ultimately depend on `FlagEmbedding`/`torch`, still not installable
  in this sandbox (disk + network — see PROJECT_STATE.md and
  `reranker.py`'s docstring for the one added wrinkle: even the Phase
  4-style `pip download --no-deps` source-reading trick wasn't possible
  this session, since this sandbox has no outbound network access at all).
  So `get_reranker_model()` is exercised for real against the genuine
  `ImportError` (confirming the boundary raises `RerankerUnavailableError`
  correctly with no mocking needed for that one test), and everything else
  touching a reranker call is tested against a fake reranker that mimics
  the documented (not source-confirmed — see reranker.py) `.compute_score()`
  shape.
"""

from __future__ import annotations

import shutil
import tempfile
import uuid

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.embeddings.embedder import EmbeddingUnavailableError
from app.main import app
from app.retrieval import qdrant_store as qs
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.hybrid import hybrid_search
from app.retrieval.reranker import RerankerUnavailableError, get_reranker_model, rerank

# --------------------------------------------------------------------------
# app/retrieval/fusion.py — pure logic, no mocking needed.
# --------------------------------------------------------------------------


def _hit(chunk_id: str, **extra) -> dict:
    return {"chunk_id": chunk_id, "score": 1.0, **extra}


def test_rrf_favors_a_chunk_present_in_both_lists() -> None:
    dense_hits = [_hit("a"), _hit("b"), _hit("c")]
    sparse_hits = [_hit("d"), _hit("b"), _hit("e")]

    fused = reciprocal_rank_fusion(dense_hits, sparse_hits, k=60)

    assert fused[0].chunk_id == "b"
    assert fused[0].dense_rank == 2
    assert fused[0].sparse_rank == 2


def test_rrf_disjoint_lists_orders_by_best_single_rank() -> None:
    dense_hits = [_hit("a"), _hit("b")]
    sparse_hits = [_hit("c"), _hit("d")]

    fused = reciprocal_rank_fusion(dense_hits, sparse_hits, k=60)

    assert [r.chunk_id for r in fused] == ["a", "c", "b", "d"]
    assert fused[0].sparse_rank is None
    assert fused[1].dense_rank is None


def test_rrf_empty_lists_returns_empty() -> None:
    assert reciprocal_rank_fusion([], [], k=60) == []


def test_rrf_preserves_payload_of_first_seen_hit() -> None:
    dense_hits = [_hit("a", text="from dense leg")]
    sparse_hits = [_hit("a", text="from sparse leg")]

    fused = reciprocal_rank_fusion(dense_hits, sparse_hits, k=60)

    assert fused[0].payload["text"] == "from dense leg"


# --------------------------------------------------------------------------
# app/retrieval/qdrant_store.py — named dense+sparse vectors, against a
# REAL local Qdrant instance (no mocking).
# --------------------------------------------------------------------------


@pytest.fixture
def qdrant_tmp_client():
    tmp_dir = tempfile.mkdtemp()
    client = qs.QdrantClient(path=tmp_dir)
    yield client
    client.close()
    shutil.rmtree(tmp_dir, ignore_errors=True)


def _uuid() -> str:
    return str(uuid.uuid4())


def test_ensure_collection_creates_dense_and_sparse_vectors(qdrant_tmp_client) -> None:
    assert qdrant_tmp_client.collection_exists("hybrid_chunks") is False
    qs.ensure_collection(qdrant_tmp_client, "hybrid_chunks", dimension=4)
    assert qdrant_tmp_client.collection_exists("hybrid_chunks") is True
    # Idempotent, same as Phase 4's plain dense-only collection.
    qs.ensure_collection(qdrant_tmp_client, "hybrid_chunks", dimension=4)


def test_upsert_with_sparse_and_search_sparse_finds_best_lexical_match(qdrant_tmp_client) -> None:
    qs.ensure_collection(qdrant_tmp_client, "hybrid_chunks", dimension=4)
    id_a, id_b = _uuid(), _uuid()
    chunks = [
        qs.ChunkVectorPayload(id_a, "doc-a", "chunk about apples", 1, 0, "paragraph", "a.pdf"),
        qs.ChunkVectorPayload(id_b, "doc-a", "chunk about oranges", 1, 1, "paragraph", "a.pdf"),
    ]
    dense_vectors = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    sparse_vectors = [{"7": 1.0, "12": 0.5}, {"7": 0.2}]

    qs.upsert_chunks(qdrant_tmp_client, "hybrid_chunks", chunks, dense_vectors, sparse_vectors)

    hits = qs.search_sparse(qdrant_tmp_client, "hybrid_chunks", {"7": 1.0, "12": 1.0}, top_k=2)
    assert len(hits) == 2
    assert hits[0]["chunk_id"] == id_a  # stronger overlap on both tokens
    assert hits[0]["score"] > hits[1]["score"]


def test_search_sparse_with_no_overlap_returns_empty_query_result(qdrant_tmp_client) -> None:
    qs.ensure_collection(qdrant_tmp_client, "hybrid_chunks", dimension=2)
    assert qs.search_sparse(qdrant_tmp_client, "hybrid_chunks", {}, top_k=5) == []


def test_search_dense_works_against_named_vector_collection(qdrant_tmp_client) -> None:
    qs.ensure_collection(qdrant_tmp_client, "hybrid_chunks", dimension=2)
    id_a = _uuid()
    chunks = [qs.ChunkVectorPayload(id_a, "doc-a", "text a", 1, 0, "paragraph", "a.pdf")]
    qs.upsert_chunks(qdrant_tmp_client, "hybrid_chunks", chunks, [[1.0, 0.0]], [{"1": 1.0}])

    hits = qs.search_dense(qdrant_tmp_client, "hybrid_chunks", [1.0, 0.0], top_k=5)
    assert len(hits) == 1
    assert hits[0]["chunk_id"] == id_a


def test_upsert_without_sparse_vectors_is_still_valid(qdrant_tmp_client) -> None:
    """Backward-compat: omitting sparse_vectors (None) must still produce a
    valid point against the named-vector collection — dense search should
    find it, sparse search simply won't (no sparse vector was written)."""
    qs.ensure_collection(qdrant_tmp_client, "hybrid_chunks", dimension=2)
    id_a = _uuid()
    chunks = [qs.ChunkVectorPayload(id_a, "doc-a", "text a", 1, 0, "paragraph", "a.pdf")]

    qs.upsert_chunks(qdrant_tmp_client, "hybrid_chunks", chunks, [[1.0, 0.0]])

    assert qs.search_dense(qdrant_tmp_client, "hybrid_chunks", [1.0, 0.0], top_k=5)[0]["chunk_id"] == id_a
    assert qs.search_sparse(qdrant_tmp_client, "hybrid_chunks", {"1": 1.0}, top_k=5) == []


def test_upsert_chunks_rejects_mismatched_sparse_length(qdrant_tmp_client) -> None:
    qs.ensure_collection(qdrant_tmp_client, "hybrid_chunks", dimension=2)
    chunks = [qs.ChunkVectorPayload(_uuid(), "doc-a", "t", 1, 0, "paragraph", "a.pdf")]
    with pytest.raises(ValueError):
        qs.upsert_chunks(qdrant_tmp_client, "hybrid_chunks", chunks, [[1.0, 0.0]], sparse_vectors=[{"1": 1.0}, {}])


# --------------------------------------------------------------------------
# app/retrieval/reranker.py
# --------------------------------------------------------------------------


class _FakeRerankerModel:
    """Mimics the documented (not source-confirmed this session — see
    reranker.py's docstring) `.compute_score()` shape: takes `[query,
    passage]` pairs, returns a list of floats. Scores by how many words of
    the query appear in the passage, purely so tests can assert a
    deterministic, sensible ordering without a real cross-encoder."""

    def compute_score(self, pairs: list[list[str]], **kwargs):
        scores = []
        for query, passage in pairs:
            q_words = set(query.lower().split())
            p_words = set(passage.lower().split())
            scores.append(float(len(q_words & p_words)))
        return scores


class _CrashingRerankerModel:
    def compute_score(self, pairs, **kwargs):
        raise RuntimeError("simulated reranker backend crash")


def test_rerank_orders_candidates_by_score() -> None:
    candidates = [
        {"chunk_id": "a", "text": "totally unrelated content"},
        {"chunk_id": "b", "text": "the revenue grew significantly this quarter"},
    ]
    results = rerank("revenue grew this quarter", candidates, model=_FakeRerankerModel())
    assert results[0].chunk_id == "b"
    assert results[0].rerank_score > results[1].rerank_score


def test_rerank_respects_top_k() -> None:
    candidates = [{"chunk_id": str(i), "text": f"word{i}"} for i in range(5)]
    results = rerank("query", candidates, model=_FakeRerankerModel(), top_k=2)
    assert len(results) == 2


def test_rerank_with_empty_candidates_returns_empty() -> None:
    assert rerank("query", [], model=_FakeRerankerModel()) == []


def test_rerank_wraps_model_crash_as_reranker_unavailable() -> None:
    with pytest.raises(RerankerUnavailableError):
        rerank("query", [{"chunk_id": "a", "text": "t"}], model=_CrashingRerankerModel())


def test_get_reranker_model_raises_when_flagembedding_not_installed() -> None:
    """Real, reproduced boundary failure — FlagEmbedding genuinely isn't
    installed in this sandbox (see requirements.txt), no mocking needed."""
    with pytest.raises(RerankerUnavailableError):
        get_reranker_model()


# --------------------------------------------------------------------------
# app/retrieval/hybrid.py — the orchestrator, against a real local Qdrant
# collection seeded directly (fast, no need to go through the full
# ingestion pipeline just to test fusion/rerank wiring).
# --------------------------------------------------------------------------


class _FixedEmbeddingModel:
    """Always returns the same pre-baked dense vector + sparse weights for
    any input text — lets tests control exactly what hybrid_search's query
    embedding looks like without depending on real bge-m3 output."""

    def __init__(self, dense: list[float], sparse: dict[str, float]):
        self.dense = dense
        self.sparse = sparse

    def encode(self, texts: list[str], **kwargs):
        return {"dense_vecs": [self.dense for _ in texts], "lexical_weights": [self.sparse for _ in texts]}


@pytest.fixture
def hybrid_settings():
    return Settings(
        QDRANT_MODE="local",
        QDRANT_COLLECTION="hybrid_search_test",
        RETRIEVAL_TOP_K_CANDIDATES=10,
        RETRIEVAL_RERANK_POOL=10,
        RETRIEVAL_TOP_K_FINAL=3,
        RRF_K=60,
    )


@pytest.fixture
def seeded_hybrid_client(qdrant_tmp_client):
    """Seeds three chunks with deliberately different dense/sparse profiles
    so fusion behavior is distinguishable:
      - dense_only: strong dense match to the query, no sparse overlap.
      - sparse_only: strong sparse match, weak/orthogonal dense match.
      - both: decent match on both legs.
    """
    qs.ensure_collection(qdrant_tmp_client, "hybrid_search_test", dimension=4)
    id_dense, id_sparse, id_both = _uuid(), _uuid(), _uuid()
    chunks = [
        qs.ChunkVectorPayload(id_dense, "doc-x", "dense only chunk", 1, 0, "paragraph", "x.pdf"),
        qs.ChunkVectorPayload(id_sparse, "doc-x", "sparse only chunk", 1, 1, "paragraph", "x.pdf"),
        qs.ChunkVectorPayload(id_both, "doc-x", "both legs chunk", 1, 2, "paragraph", "x.pdf"),
    ]
    dense_vectors = [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.9, 0.1, 0.0, 0.0]]
    sparse_vectors = [{}, {"7": 1.0}, {"7": 0.8}]
    qs.upsert_chunks(qdrant_tmp_client, "hybrid_search_test", chunks, dense_vectors, sparse_vectors)
    return qdrant_tmp_client, {"dense": id_dense, "sparse": id_sparse, "both": id_both}


def test_hybrid_search_deprioritizes_chunk_matching_only_one_leg(seeded_hybrid_client, hybrid_settings) -> None:
    client, ids = seeded_hybrid_client
    embedding_model = _FixedEmbeddingModel(dense=[1.0, 0.0, 0.0, 0.0], sparse={"7": 1.0})

    results = hybrid_search(
        "query", qdrant_client=client, embedding_model=embedding_model, settings=hybrid_settings
    )

    result_ids = [r.chunk_id for r in results]
    assert ids["dense"] in result_ids
    assert ids["sparse"] in result_ids
    assert ids["both"] in result_ids
    # The chunk matching only the dense leg must not outrank both chunks
    # that also have sparse support — RRF's whole point.
    assert results[0].chunk_id != ids["dense"]
    assert all(r.reranked is False for r in results)


def test_hybrid_search_with_reranker_overrides_fused_order(seeded_hybrid_client, hybrid_settings) -> None:
    client, ids = seeded_hybrid_client
    embedding_model = _FixedEmbeddingModel(dense=[1.0, 0.0, 0.0, 0.0], sparse={"7": 1.0})

    class _ForceRerankToDenseOnly:
        def compute_score(self, pairs, **kwargs):
            return [10.0 if "dense only" in passage else 0.0 for _, passage in pairs]

    results = hybrid_search(
        "query",
        qdrant_client=client,
        embedding_model=embedding_model,
        settings=hybrid_settings,
        reranker_model=_ForceRerankToDenseOnly(),
    )

    assert results[0].chunk_id == ids["dense"]
    assert results[0].reranked is True


def test_hybrid_search_falls_back_to_fused_order_when_reranker_unavailable(
    seeded_hybrid_client, hybrid_settings
) -> None:
    client, ids = seeded_hybrid_client
    embedding_model = _FixedEmbeddingModel(dense=[1.0, 0.0, 0.0, 0.0], sparse={"7": 1.0})

    results = hybrid_search(
        "query",
        qdrant_client=client,
        embedding_model=embedding_model,
        settings=hybrid_settings,
        reranker_model=_CrashingRerankerModel(),
    )

    # Must not raise, and must degrade to the fused (non-reranked) order.
    assert len(results) > 0
    assert all(r.reranked is False for r in results)


def test_hybrid_search_scoped_to_document_id_with_no_matches_returns_empty(
    seeded_hybrid_client, hybrid_settings
) -> None:
    client, _ids = seeded_hybrid_client
    embedding_model = _FixedEmbeddingModel(dense=[1.0, 0.0, 0.0, 0.0], sparse={"7": 1.0})

    results = hybrid_search(
        "query",
        qdrant_client=client,
        embedding_model=embedding_model,
        settings=hybrid_settings,
        document_id="doc-that-does-not-exist",
    )
    assert results == []


# --------------------------------------------------------------------------
# POST /api/search — end-to-end through the real FastAPI app, real local
# Qdrant, embedding + reranker faked at the same boundary the router-level
# tests use.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


class _DeterministicEmbeddingModel:
    """Same scheme as test_phase4's fake: dense vector derived from text
    length, so embedding identical text at index-time and query-time
    produces an identical vector (guaranteed top hit at cosine sim 1.0)."""

    def __init__(self, dimension: int):
        self.dimension = dimension

    def encode(self, texts: list[str], **kwargs):
        dense, lexical = [], []
        for t in texts:
            vec = [0.0] * self.dimension
            vec[len(t) % self.dimension] = 1.0
            dense.append(vec)
            lexical.append({str(len(t) % 50): 1.0})
        return {"dense_vecs": dense, "lexical_weights": lexical}


def test_search_endpoint_finds_indexed_chunk(client, monkeypatch) -> None:
    fake_model = _DeterministicEmbeddingModel(dimension=get_settings().EMBEDDING_DIMENSION)
    monkeypatch.setattr("app.ingestion.router.get_embedding_model", lambda **kw: fake_model)

    import io

    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    rect = pymupdf.Rect(72, 72, page.rect.width - 72, page.rect.height - 72)
    unique_text = "The quarterly gross margin improved to thirty eight percent."
    page.insert_textbox(rect, unique_text, fontsize=11)
    pdf_bytes = doc.tobytes()
    doc.close()

    resp = client.post("/api/documents", files={"file": ("margin.pdf", io.BytesIO(pdf_bytes), "application/pdf")})
    doc_id = resp.json()["id"]
    assert client.get(f"/api/documents/{doc_id}").json()["processing_status"] == "completed"

    monkeypatch.setattr("app.api.search.get_embedding_model", lambda **kw: fake_model)
    monkeypatch.setattr(
        "app.api.search.get_reranker_model",
        lambda **kw: (_ for _ in ()).throw(RerankerUnavailableError("not installed in this sandbox")),
    )

    search_resp = client.post("/api/search", json={"query": unique_text, "document_id": doc_id})
    assert search_resp.status_code == 200
    body = search_resp.json()
    assert body["reranker_used"] is False
    assert len(body["results"]) >= 1
    assert body["results"][0]["document_id"] == doc_id


def test_search_endpoint_rejects_empty_query(client) -> None:
    resp = client.post("/api/search", json={"query": "   "})
    assert resp.status_code == 400


def test_search_endpoint_returns_503_when_embedding_unavailable(client, monkeypatch) -> None:
    def _raise(**kwargs):
        raise EmbeddingUnavailableError("simulated: model weights unreachable")

    monkeypatch.setattr("app.api.search.get_embedding_model", _raise)

    resp = client.post("/api/search", json={"query": "anything"})
    assert resp.status_code == 503
