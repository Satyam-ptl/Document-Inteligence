"""
Phase 6 tests: grounded answer generation, prompt construction, citation
validation, and the /api/chat endpoint.

Same tiered approach as every earlier phase's test file (see their own
docstrings) — different boundaries tested differently on purpose:

- `app/generation/prompt.py` and `app/generation/citation.py`: pure
  Python, zero external dependency. Tested directly, for real, no mocking
  needed — and, per PROJECT_STATE.md's Phase 6 verification log, these
  exact scenarios were already run by hand with a bare `python3` script
  this session (no pytest available at all — see that log for why), so
  this file mostly formalizes checks that are already known to pass
  rather than being a leap of faith.
- `app/generation/generator.py`: orchestrates `hybrid_search()` (Phase 5)
  plus a real LLM call, so it needs the full Phase 4/5 dependency chain
  (pydantic-settings, numpy, qdrant-client, and either a real or embedded
  Qdrant collection) to even import `app.config`, let alone run — none of
  that was installable in this sandbox (see PROJECT_STATE.md). Tested here
  against a fake `llm_client` and the same local/embedded Qdrant + fake
  embedding model pattern `test_phase5_retrieval.py` established, NOT run
  this session.
- `app/generation/llm_client.py`: the one piece of Phase 6 that talks to a
  real external HTTP API. `get_llm_client()`'s config-validation boundary
  (missing `LLM_API_KEY` -> `LLMUnavailableError`, before any HTTP call is
  even attempted) needs no network and no `httpx` import to test at all —
  but even that could not be run this session because `httpx` itself
  isn't installed here (see llm_client.py's docstring: this sandbox has no
  outbound network access at all, so not even `pip install httpx`
  succeeded). The actual HTTP call shapes are UNVERIFIED — the next
  session with real network access should add an `httpx.MockTransport`-
  based test per provider (deepseek/openai share `_OpenAICompatibleClient`,
  so one test each is enough) before trusting them, per the Phase 4-style
  source-reading pass this module's docstring calls for.
"""

from __future__ import annotations

import shutil
import tempfile
import uuid

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.embeddings.embedder import EmbeddingUnavailableError
from app.generation.citation import (
    extract_citation_indices,
    is_insufficient_evidence_answer,
    split_sentences,
    validate_citations,
)
from app.generation.llm_client import (
    LLMUnavailableError,
    _LocalExtractiveClient,
    get_llm_client,
    reset_llm_client_singleton,
)
from app.generation.prompt import (
    SYSTEM_PROMPT,
    ContextPassage,
    build_context_block,
    build_user_prompt,
    to_context_passages,
)
from app.main import app
from app.retrieval import qdrant_store as qs


# ---------------------------------------------------------------------------
# app/generation/citation.py — pure Python, real-verified (see module docstring)
# ---------------------------------------------------------------------------


def test_extract_citation_indices_dedupes_and_preserves_first_seen_order():
    assert extract_citation_indices("A [1]. B [2][1]. C.") == [1, 2]


def test_extract_citation_indices_empty_when_no_markers():
    assert extract_citation_indices("No markers here.") == []


def test_validate_citations_fully_cited_answer():
    answer = "Revenue grew 12% in 2025 [2]. The two reports agree on this [1][2]."
    result = validate_citations(answer, num_passages=3)
    assert result.is_insufficient_evidence is False
    assert result.invalid_indices == []
    assert result.uncited_sentence_count == 0
    assert result.fully_cited is True


def test_validate_citations_flags_hallucinated_citation():
    answer = "The company was founded in 1998 [7]."
    result = validate_citations(answer, num_passages=3)
    assert result.invalid_indices == [7]
    assert result.valid_indices == []
    assert result.fully_cited is False


def test_validate_citations_flags_uncited_sentence():
    answer = "The company was founded in 1998. It grew rapidly afterward [1]."
    result = validate_citations(answer, num_passages=2)
    assert result.total_sentence_count == 2
    assert result.uncited_sentence_count == 1
    assert result.fully_cited is False


def test_validate_citations_insufficient_evidence_short_circuit():
    answer = "INSUFFICIENT_EVIDENCE: The passages do not mention pricing."
    assert is_insufficient_evidence_answer(answer) is True
    result = validate_citations(answer, num_passages=5)
    assert result.is_insufficient_evidence is True
    assert result.fully_cited is True
    assert result.cited_indices == []


def test_is_insufficient_evidence_answer_requires_prefix_not_substring():
    assert is_insufficient_evidence_answer("  insufficient_evidence: no match") is True
    assert is_insufficient_evidence_answer("This is insufficient_evidence in the middle") is False


def test_split_sentences_keeps_citation_markers_glued_to_their_sentence():
    sentences = split_sentences("First fact [1]. Second fact [2][3]. Plain closer.")
    assert sentences == ["First fact [1].", "Second fact [2][3].", "Plain closer."]


# ---------------------------------------------------------------------------
# app/generation/prompt.py — pure Python, real-verified (see module docstring)
# ---------------------------------------------------------------------------


class _FakeResult:
    def __init__(self, chunk_id, document_id, source_filename, page_number, text):
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.source_filename = source_filename
        self.page_number = page_number
        self.text = text


def test_to_context_passages_numbers_from_one_and_preserves_order():
    results = [
        _FakeResult("c1", "d1", "report.pdf", 3, "Revenue was $5M in 2025."),
        _FakeResult("c2", "d1", "report.pdf", 7, "Costs decreased by 10%."),
    ]
    passages = to_context_passages(results)
    assert [p.index for p in passages] == [1, 2]
    assert passages[0].chunk_id == "c1"
    assert passages[1].page_number == 7


def test_build_context_block_includes_source_and_page():
    passages = [ContextPassage(1, "c1", "d1", "report.pdf", 3, "Revenue was $5M.")]
    block = build_context_block(passages)
    assert "[1] (source: report.pdf, page 3)" in block
    assert "Revenue was $5M." in block


def test_build_context_block_empty_passages():
    assert build_context_block([]) == "(no passages retrieved)"


def test_build_user_prompt_contains_question_and_passage_markers():
    passages = [ContextPassage(1, "c1", "d1", "report.pdf", 3, "Revenue was $5M.")]
    prompt = build_user_prompt("What was the revenue?", passages)
    assert "What was the revenue?" in prompt
    assert "[1]" in prompt


def test_system_prompt_instructs_insufficient_evidence_format():
    assert "INSUFFICIENT_EVIDENCE" in SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# app/generation/llm_client.py — offline fallback relevance guard
# ---------------------------------------------------------------------------


def test_local_fallback_does_not_present_unrelated_prompt_as_answer():
    client = _LocalExtractiveClient()
    user = (
        "Context passages:\n\n"
        "[1] (source: speaking-prompts.pdf, page 6)\n"
        "Don't you think we could... Asking for opinion What is your opinion on...\n\n"
        "---\n\n"
        "Question: What is the capital of Mongolia?"
    )

    answer = client.complete("", user)

    assert answer.startswith("INSUFFICIENT_EVIDENCE:")
    assert "Asking for opinion" not in answer


def test_local_fallback_matches_split_textbook_title():
    client = _LocalExtractiveClient()
    user = (
        "Context passages:\n\n"
        "[1] (source: XII-english.pdf, page 6)\n"
        "It is a pleasure to hand over this textbook English Yuvakbharati "
        "for Standard XII which will be helpful in shaping the course of your life.\n\n"
        "[2] (source: XII-english.pdf, page 7)\n"
        "The traveler brings back is an ineffable compound of himself and the place.\n\n"
        "---\n\n"
        "Question: What is Yuvak Bharati?"
    )

    answer = client.complete("", user)

    assert "Yuvakbharati" in answer
    assert "traveler" not in answer


def test_local_fallback_rejects_general_opinion_question_without_document_evidence():
    client = _LocalExtractiveClient()
    user = (
        "Context passages:\n\n"
        "[1] (source: XII-english.pdf, page 6)\n"
        "This textbook introduces English literature for Standard XII.\n\n"
        "---\n\n"
        "Question: If you could redesign the education system, what would be your first change?"
    )

    answer = client.complete("", user)

    assert answer.startswith("INSUFFICIENT_EVIDENCE:")
    assert "personalized" not in answer


def test_local_fallback_returns_relevant_rows_from_table_chunk():
    client = _LocalExtractiveClient()
    user = (
        "Context passages:\n\n"
        "[1] (source: revenue.csv, page 1)\n"
        "Table: revenue\n"
        "Headers: Year | Revenue\n"
        "2024 | 100\n"
        "2025 | 150\n\n"
        "---\n\n"
        "Question: What was the revenue in 2025?"
    )

    answer = client.complete("", user)

    assert "2025 | 150" in answer
    assert answer.endswith("[1].")


# ---------------------------------------------------------------------------
# app/generation/llm_client.py — config-validation boundary only (no HTTP)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_llm_singleton():
    reset_llm_client_singleton()
    yield
    reset_llm_client_singleton()


def test_get_llm_client_raises_when_api_key_missing():
    settings = Settings(LLM_PROVIDER="deepseek", LLM_API_KEY=None)
    with pytest.raises(LLMUnavailableError):
        get_llm_client(settings)


def test_get_llm_client_caches_per_provider_model_and_key():
    settings = Settings(LLM_PROVIDER="deepseek", LLM_API_KEY="test-key", LLM_MODEL="deepseek-chat")
    client1 = get_llm_client(settings)
    client2 = get_llm_client(settings)
    assert client1 is client2  # same cache key -> same cached instance

    settings_changed = Settings(LLM_PROVIDER="openai", LLM_API_KEY="test-key", LLM_MODEL="gpt-4o")
    client3 = get_llm_client(settings_changed)
    assert client3 is not client1  # different provider -> new client, not a stale cached one


# ---------------------------------------------------------------------------
# app/generation/llm_client.py — real HTTP call shapes, now verifiable:
# `httpx` is installed in this environment (unlike the Phase 5/6 sessions
# that wrote this module - see its docstring). `_OpenAICompatibleClient` and
# `_GeminiClient` both call the module-level `httpx.post(...)` function
# directly rather than an injectable `httpx.Client` instance, so an
# `httpx.MockTransport` can't be wired in without changing production code
# just for testability; monkeypatching `httpx.post` itself is the
# equivalent, no-production-code-change way to exercise the exact same
# parsing logic against a fully-controlled fake response — these are real
# assertions against the real `complete()` method, not a description of
# intended behavior.
# ---------------------------------------------------------------------------


class _FakeHTTPResponse:
    def __init__(self, json_body: dict, status_code: int = 200):
        self._json_body = json_body
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError("simulated non-2xx", request=None, response=self)

    def json(self) -> dict:
        return self._json_body


def test_openai_compatible_client_extracts_content_on_happy_path(monkeypatch):
    from app.generation.llm_client import _OpenAICompatibleClient

    captured = {}

    def _fake_post(url, *, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return _FakeHTTPResponse({"choices": [{"message": {"content": "The revenue was $5M."}}]})

    monkeypatch.setattr("httpx.post", _fake_post)

    client = _OpenAICompatibleClient(
        api_key="sk-test", base_url="https://api.deepseek.com/v1", model="deepseek-chat", timeout_seconds=30.0
    )
    result = client.complete(system="You are helpful.", user="What was the revenue?", max_tokens=512, temperature=0.0)

    assert result == "The revenue was $5M."
    assert captured["url"] == "https://api.deepseek.com/v1/chat/completions"
    assert captured["json"]["messages"] == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What was the revenue?"},
    ]
    assert captured["headers"]["Authorization"] == "Bearer sk-test"


def test_openai_compatible_client_raises_on_malformed_response(monkeypatch):
    from app.generation.llm_client import _OpenAICompatibleClient

    monkeypatch.setattr("httpx.post", lambda *a, **kw: _FakeHTTPResponse({"unexpected": "shape"}))

    client = _OpenAICompatibleClient(
        api_key="sk-test", base_url="https://api.openai.com/v1", model="gpt-4o", timeout_seconds=30.0
    )
    with pytest.raises(LLMUnavailableError, match="unexpected shape"):
        client.complete(system="sys", user="usr", max_tokens=512, temperature=0.0)


def test_openai_compatible_client_raises_on_http_error(monkeypatch):
    from app.generation.llm_client import _OpenAICompatibleClient

    monkeypatch.setattr("httpx.post", lambda *a, **kw: _FakeHTTPResponse({}, status_code=500))

    client = _OpenAICompatibleClient(
        api_key="sk-test", base_url="https://api.openai.com/v1", model="gpt-4o", timeout_seconds=30.0
    )
    with pytest.raises(LLMUnavailableError, match="failed"):
        client.complete(system="sys", user="usr", max_tokens=512, temperature=0.0)


def test_gemini_client_extracts_and_joins_text_parts_on_happy_path(monkeypatch):
    from app.generation.llm_client import _GeminiClient

    captured = {}

    def _fake_post(url, *, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return _FakeHTTPResponse(
            {"candidates": [{"content": {"parts": [{"text": "The revenue "}, {"text": "was $5M."}]}}]}
        )

    monkeypatch.setattr("httpx.post", _fake_post)

    client = _GeminiClient(
        api_key="gm-test",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        model="gemini-2.0-flash",
        timeout_seconds=30.0,
    )
    result = client.complete(system="sys", user="What was the revenue?", max_tokens=512, temperature=0.0)

    assert result == "The revenue was $5M."
    assert captured["url"] == (
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    )
    assert captured["json"]["system_instruction"] == {"parts": [{"text": "sys"}]}
    assert captured["headers"]["x-goog-api-key"] == "gm-test"


def test_gemini_client_raises_on_malformed_response(monkeypatch):
    from app.generation.llm_client import _GeminiClient

    monkeypatch.setattr("httpx.post", lambda *a, **kw: _FakeHTTPResponse({"unexpected": "shape"}))

    client = _GeminiClient(
        api_key="gm-test", base_url="https://generativelanguage.googleapis.com/v1beta",
        model="gemini-2.0-flash", timeout_seconds=30.0,
    )
    with pytest.raises(LLMUnavailableError, match="unexpected shape"):
        client.complete(system="sys", user="usr", max_tokens=512, temperature=0.0)


# ---------------------------------------------------------------------------
# app/generation/generator.py + POST /api/chat — needs the full Phase 4/5
# dependency chain (pydantic-settings, numpy, qdrant-client). NOT run this
# session — see this file's module docstring and PROJECT_STATE.md.
# ---------------------------------------------------------------------------


class _FakeEmbeddingModel:
    """Same fake shape as test_phase4_embeddings.py / test_phase5_retrieval.py's
    fakes: mimics BGEM3FlagModel.encode()'s real, source-confirmed return
    dict (`dense_vecs` + `lexical_weights`). Only used here to satisfy
    `app.api.chat.get_embedding_model`'s call for a query embedding when a
    test doesn't care what that embedding actually is (e.g. this file's
    zero-chunks-indexed short-circuit test) - real weights still aren't
    reachable in this sandbox (no `huggingface.co` access)."""

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
    """Records the prompt it was given and returns a fixed, pre-scripted
    answer — lets these tests assert generator.py's orchestration logic
    (short-circuit on empty results, citation validation wiring) without
    depending on a real LLM call at all."""

    def __init__(self, answer: str):
        self.answer = answer
        self.last_system: str | None = None
        self.last_user: str | None = None

    def complete(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        self.last_system = system
        self.last_user = user
        return self.answer


@pytest.fixture
def client(monkeypatch, tmp_path):
    """Same embedded-Qdrant-plus-fake-embedding-model fixture pattern as
    `test_phase4_embeddings.py` / `test_phase5_retrieval.py` — a fresh
    temp-dir Qdrant collection and DB per test, and the real embedding
    model call replaced with a fake at the same seam those files use."""
    db_path = tmp_path / "test.db"
    qdrant_path = tmp_path / "qdrant"

    def _get_settings_override():
        return Settings(
            DATABASE_URL=f"sqlite:///{db_path}",
            QDRANT_MODE="local",
            QDRANT_LOCAL_PATH=str(qdrant_path),
            QDRANT_COLLECTION=f"test_{uuid.uuid4().hex}",
            LLM_PROVIDER="deepseek",
            LLM_API_KEY="test-key",
        )

    app.dependency_overrides[get_settings] = _get_settings_override
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    shutil.rmtree(qdrant_path, ignore_errors=True)


def test_chat_endpoint_rejects_empty_query(client):
    resp = client.post("/api/chat", json={"query": "   "})
    assert resp.status_code == 400


def test_chat_endpoint_503_when_embedding_model_unavailable(client, monkeypatch):
    def _raise(*args, **kwargs):
        raise EmbeddingUnavailableError("model weights unreachable")

    monkeypatch.setattr("app.api.chat.get_embedding_model", _raise)
    resp = client.post("/api/chat", json={"query": "What was the revenue?"})
    assert resp.status_code == 503


def test_chat_endpoint_503_when_llm_unavailable(client, monkeypatch):
    def _raise(*args, **kwargs):
        raise LLMUnavailableError("LLM_API_KEY is not set")

    monkeypatch.setattr("app.api.chat.get_llm_client", _raise)
    resp = client.post("/api/chat", json={"query": "What was the revenue?"})
    assert resp.status_code == 503


def test_chat_endpoint_returns_insufficient_evidence_without_calling_llm(client, monkeypatch):
    """No chunks indexed at all -> generator.py's zero-results short-circuit
    should fire, and the (fake) LLM should never even be called.

    Real embedding model is still not installable in this sandbox (no
    `huggingface.co` access even where FlagEmbedding/torch themselves can
    be installed - see embedder.py's docstring), so this needs the same
    `get_embedding_model` fake every other endpoint test in
    test_phase4/5 uses. Missing this monkeypatch was a real bug this test
    file had - discovered by actually running it (see PROJECT_STATE.md's
    Phase 6 -> Phase 7 handoff verification log), not a pre-existing
    design decision."""
    fake_embedding_model = _FakeEmbeddingModel(dimension=get_settings().EMBEDDING_DIMENSION)
    monkeypatch.setattr("app.api.chat.get_embedding_model", lambda **kw: fake_embedding_model)
    fake_llm = _FakeLLMClient(answer="should never be returned")
    monkeypatch.setattr("app.api.chat.get_llm_client", lambda settings: fake_llm)

    resp = client.post("/api/chat", json={"query": "What was the revenue?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["insufficient_evidence"] is True
    assert body["sources"] == []
    assert fake_llm.last_user is None  # confirms the LLM was never actually invoked
