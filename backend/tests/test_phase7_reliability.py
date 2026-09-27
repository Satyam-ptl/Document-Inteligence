"""
Phase 7 tests: confidence scoring, conflict detection, and their wiring
into `generate_answer()` / `POST /api/chat`.

Same tiered approach as every earlier phase's test file:

- `app/reliability/confidence.py` and `app/reliability/conflict.py`: pure
  Python, zero external dependency. Tested directly, for real, no mocking.
- `app/generation/generator.py`'s Phase 7 wiring and the `/api/chat`
  response fields: needs the full Phase 4/5/6 dependency chain (real in
  this environment — see PROJECT_STATE.md's Phase 6->7 verification log),
  so tested against a real embedded/local Qdrant collection + a fake
  embedding model, same pattern test_phase5/6 established.
"""

from __future__ import annotations

import shutil
import tempfile
import uuid

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.generation.citation import validate_citations
from app.generation.prompt import ContextPassage
from app.main import app
from app.reliability.confidence import ConfidenceResult, compute_confidence
from app.reliability.conflict import detect_conflicts
from app.retrieval import qdrant_store as qs
from app.retrieval.qdrant_store import reset_qdrant_client_cache

# ---------------------------------------------------------------------------
# app/reliability/confidence.py — pure Python
# ---------------------------------------------------------------------------


class _FakeScoredResult:
    def __init__(self, score: float, reranked: bool):
        self.score = score
        self.reranked = reranked


def test_compute_confidence_insufficient_evidence_short_circuits_to_low():
    citations = validate_citations("INSUFFICIENT_EVIDENCE: nothing relevant found.", num_passages=5)
    result = compute_confidence([_FakeScoredResult(0.9, True)], citations, rrf_k=60)
    assert result.level == "low"
    assert result.score == 0.0
    assert "enough information" in result.reasons[0].lower()


def test_compute_confidence_no_results_is_low():
    citations = validate_citations("Some answer [1].", num_passages=1)
    result = compute_confidence([], citations, rrf_k=60)
    assert result.level == "low"
    assert result.score == 0.0


def test_compute_confidence_high_for_reranked_score_and_fully_cited_answer():
    answer = "Revenue was $5M in 2025 [1]."
    citations = validate_citations(answer, num_passages=1)
    results = [_FakeScoredResult(score=0.95, reranked=True)]
    result = compute_confidence(results, citations, rrf_k=60)
    assert result.level == "high"
    assert result.score > 0.66
    assert result.reasons == ["strong retrieval match and a fully-cited answer"]


def test_compute_confidence_weak_reranked_score_pulls_level_down_even_if_fully_cited():
    # Retrieval and citation quality are weighted equally (see module
    # docstring): a very weak top match (0.05) combined with a perfectly
    # cited answer (1.0) averages to ~0.53 — "medium," not "high," since a
    # weak match alone should be enough to keep the level off the top tier.
    answer = "Revenue was $5M in 2025 [1]."
    citations = validate_citations(answer, num_passages=1)
    results = [_FakeScoredResult(score=0.05, reranked=True)]
    result = compute_confidence(results, citations, rrf_k=60)
    assert result.level == "medium"
    assert result.score < 0.66


def test_compute_confidence_fused_only_score_is_rescaled_not_treated_as_raw_rrf():
    # A rank-1-in-both-legs fused score at RRF_K=60 is ~0.0328, which would
    # look "low" against a raw 0..1 threshold but is actually the best
    # possible fused-only outcome — normalization should treat it that way.
    rrf_k = 60
    best_possible_fused_score = 2.0 / (rrf_k + 1)
    answer = "Revenue was $5M in 2025 [1]."
    citations = validate_citations(answer, num_passages=1)
    results = [_FakeScoredResult(score=best_possible_fused_score, reranked=False)]
    result = compute_confidence(results, citations, rrf_k=rrf_k)
    assert result.level == "high"
    assert any("reranker was unavailable" in r for r in result.reasons)


def test_compute_confidence_penalizes_hallucinated_citations():
    answer = "The company was founded in 1998 [7]."  # passage 7 doesn't exist
    citations = validate_citations(answer, num_passages=2)
    results = [_FakeScoredResult(score=0.95, reranked=True)]
    result = compute_confidence(results, citations, rrf_k=60)
    assert result.level in ("low", "medium")
    assert any("never given to the model" in r for r in result.reasons)


def test_compute_confidence_penalizes_uncited_sentences():
    answer = "The company was founded in 1998. It grew rapidly afterward [1]."
    citations = validate_citations(answer, num_passages=1)
    results = [_FakeScoredResult(score=0.95, reranked=True)]
    result = compute_confidence(results, citations, rrf_k=60)
    assert any("no citation at all" in r for r in result.reasons)


def test_compute_confidence_score_is_clamped_and_rounded():
    citations = validate_citations("An answer [1].", num_passages=1)
    result = compute_confidence([_FakeScoredResult(score=1.0, reranked=True)], citations, rrf_k=60)
    assert 0.0 <= result.score <= 1.0
    assert result.score == round(result.score, 4)


# ---------------------------------------------------------------------------
# app/reliability/conflict.py — pure Python
# ---------------------------------------------------------------------------


def _passage(index: int, text: str) -> ContextPassage:
    return ContextPassage(index=index, chunk_id=f"c{index}", document_id="d1", source_filename="f.pdf",
                           page_number=1, text=text)


def test_detect_conflicts_flags_different_revenue_figures_for_same_quarter():
    passages = [
        _passage(1, "Total revenue for Q3 2025 was $5.2 million."),
        _passage(2, "Total revenue for Q3 2025 was $6.1 million."),
    ]
    result = detect_conflicts(passages)
    assert result.has_conflict is True
    assert len(result.conflicts) == 1
    conflict = result.conflicts[0]
    assert {conflict.passage_index_a, conflict.passage_index_b} == {1, 2}


def test_detect_conflicts_no_conflict_when_values_agree_within_tolerance():
    passages = [
        _passage(1, "Total revenue for Q3 2025 was $5.00 million."),
        _passage(2, "Total revenue for Q3 2025 was $5.02 million."),
    ]
    result = detect_conflicts(passages)
    assert result.has_conflict is False


def test_detect_conflicts_ignores_same_passage_numbers():
    # Two different numbers in the *same* passage are not a cross-source conflict.
    passages = [_passage(1, "Revenue was $5 million in Q1 and $6 million in Q2.")]
    result = detect_conflicts(passages)
    assert result.has_conflict is False


def test_detect_conflicts_ignores_unrelated_numbers_with_no_context_overlap():
    passages = [
        _passage(1, "The building has 12 floors."),
        _passage(2, "The company employs 4500 people worldwide."),
    ]
    result = detect_conflicts(passages)
    assert result.has_conflict is False


def test_detect_conflicts_ignores_different_unit_types():
    passages = [
        _passage(1, "Market share grew to 12% this quarter."),
        _passage(2, "Market share reached $12 million in value this quarter."),
    ]
    result = detect_conflicts(passages)
    assert result.has_conflict is False  # percent vs. magnitude — never compared


def test_detect_conflicts_empty_passages_returns_no_conflict():
    result = detect_conflicts([])
    assert result.has_conflict is False
    assert result.conflicts == []


def test_detect_conflicts_shared_context_is_populated_for_explainability():
    passages = [
        _passage(1, "Total revenue for Q3 2025 was $5.2 million."),
        _passage(2, "Total revenue for Q3 2025 was $6.1 million."),
    ]
    result = detect_conflicts(passages)
    assert result.has_conflict is True
    assert "revenue" in result.conflicts[0].shared_context


# ---------------------------------------------------------------------------
# app/generation/generator.py + POST /api/chat — Phase 7 wiring, against a
# real embedded/local Qdrant collection (same pattern as test_phase5/6).
# ---------------------------------------------------------------------------


class _FakeEmbeddingModel:
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


class _FakeLLMClient:
    def __init__(self, answer: str):
        self.answer = answer
        self.last_user: str | None = None

    def complete(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        self.last_user = user
        return self.answer


@pytest.fixture
def client(monkeypatch, tmp_path):
    # `get_qdrant_client()` caches a process-wide singleton (see
    # qdrant_store.py's own docstring on why local mode needs this) — reset
    # it before AND after this test so a fresh client opens against THIS
    # test's tmp_path rather than silently reusing a previous test's client
    # (and its now-deleted directory). Missing this was a real bug this
    # test file had, caught by actually running it — see PROJECT_STATE.md's
    # Phase 7 verification log.
    reset_qdrant_client_cache()
    db_path = tmp_path / "test.db"
    qdrant_path = tmp_path / "qdrant"
    test_settings = Settings(
        DATABASE_URL=f"sqlite:///{db_path}",
        QDRANT_MODE="local",
        QDRANT_LOCAL_PATH=str(qdrant_path),
        QDRANT_COLLECTION=f"test_{uuid.uuid4().hex}",
        LLM_PROVIDER="deepseek",
        LLM_API_KEY="test-key",
        RERANK_ENABLED=False,  # keep this test about confidence/conflict wiring, not reranker behavior
    )

    # NB: `get_settings()` called directly (not through FastAPI's DI) does
    # NOT go through `app.dependency_overrides` below — that override only
    # fires for endpoint parameters the framework resolves. Tests that need
    # the *same* settings the app is using (e.g. to seed Qdrant at the
    # right path/collection before hitting the endpoint) must use
    # `test_settings` directly instead of calling `get_settings()` again.
    app.dependency_overrides[get_settings] = lambda: test_settings
    with TestClient(app) as c:
        c.test_settings = test_settings  # stash for tests that need it
        yield c
    app.dependency_overrides.clear()
    reset_qdrant_client_cache()
    shutil.rmtree(qdrant_path, ignore_errors=True)


def _seed_conflicting_chunks(settings: Settings) -> None:
    """Index two chunks with contradictory revenue figures directly via
    qdrant_store, bypassing the ingestion pipeline (same shortcut
    test_phase5's seeded_hybrid_client fixture uses)."""
    qclient = qs.QdrantClient(path=settings.QDRANT_LOCAL_PATH)
    qs.ensure_collection(qclient, settings.QDRANT_COLLECTION, dimension=settings.EMBEDDING_DIMENSION)
    chunks = [
        qs.ChunkVectorPayload(
            str(uuid.uuid4()), "doc-1", "Total revenue for Q3 2025 was $5.2 million.", 1, 0, "paragraph", "a.pdf"
        ),
        qs.ChunkVectorPayload(
            str(uuid.uuid4()), "doc-1", "Total revenue for Q3 2025 was $6.1 million.", 1, 1, "paragraph", "b.pdf"
        ),
    ]
    dim = settings.EMBEDDING_DIMENSION
    dense_vectors = [[1.0 if i == 0 else 0.0 for i in range(dim)], [1.0 if i == 0 else 0.0 for i in range(dim)]]
    sparse_vectors = [{"1": 1.0}, {"1": 1.0}]
    qs.upsert_chunks(qclient, settings.QDRANT_COLLECTION, chunks, dense_vectors, sparse_vectors)
    qclient.close()


def test_chat_endpoint_includes_confidence_and_no_conflict_for_single_source_answer(client, monkeypatch):
    settings = client.test_settings
    fake_embedding_model = _FakeEmbeddingModel(dimension=settings.EMBEDDING_DIMENSION)
    monkeypatch.setattr("app.api.chat.get_embedding_model", lambda **kw: fake_embedding_model)
    fake_llm = _FakeLLMClient(answer="Revenue was $5.2 million [1].")
    monkeypatch.setattr("app.api.chat.get_llm_client", lambda settings: fake_llm)

    resp = client.post("/api/chat", json={"query": "What was the revenue?"})
    assert resp.status_code == 200
    body = resp.json()
    assert "confidence" in body
    assert body["confidence"]["level"] in ("high", "medium", "low")
    assert isinstance(body["confidence"]["score"], float)
    assert isinstance(body["confidence"]["reasons"], list)
    assert "conflicts" in body
    assert body["conflicts"]["has_conflict"] is False


def test_chat_endpoint_detects_conflict_across_two_indexed_chunks(client, monkeypatch):
    settings = client.test_settings
    _seed_conflicting_chunks(settings)

    fake_embedding_model = _FakeEmbeddingModel(dimension=settings.EMBEDDING_DIMENSION)
    monkeypatch.setattr("app.api.chat.get_embedding_model", lambda **kw: fake_embedding_model)
    fake_llm = _FakeLLMClient(answer="Total revenue for Q3 2025 was $5.2 million [1].")
    monkeypatch.setattr("app.api.chat.get_llm_client", lambda settings: fake_llm)

    resp = client.post("/api/chat", json={"query": "What was Q3 revenue?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["conflicts"]["has_conflict"] is True
    assert len(body["conflicts"]["conflicts"]) >= 1
    pair = body["conflicts"]["conflicts"][0]
    assert {pair["passage_index_a"], pair["passage_index_b"]} == {1, 2}


def test_chat_endpoint_zero_evidence_path_still_returns_low_confidence_and_no_conflict(client, monkeypatch):
    settings = client.test_settings
    fake_embedding_model = _FakeEmbeddingModel(dimension=settings.EMBEDDING_DIMENSION)
    monkeypatch.setattr("app.api.chat.get_embedding_model", lambda **kw: fake_embedding_model)
    fake_llm = _FakeLLMClient(answer="should never be returned")
    monkeypatch.setattr("app.api.chat.get_llm_client", lambda settings: fake_llm)

    resp = client.post("/api/chat", json={"query": "What was the revenue?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["insufficient_evidence"] is True
    assert body["confidence"]["level"] == "low"
    assert body["confidence"]["score"] == 0.0
    assert body["conflicts"]["has_conflict"] is False
    assert fake_llm.last_user is None
