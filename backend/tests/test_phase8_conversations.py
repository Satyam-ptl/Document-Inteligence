"""
Phase 8 tests: conversation memory / multi-turn follow-up resolution.

Same tiered approach as every earlier phase's test file:

- `app/conversations/memory.py`'s pure logic (`resolve_followup`,
  `build_history_block`) tested directly with a fake LLM client — no
  network, no real DB needed for these.
- `get_or_create_conversation` / `get_recent_turns` / `record_turn`
  tested against a real (temp-file) SQLite DB session, same tier
  Phase 1/2's tests use for the DB layer.
- The full `/api/chat` + `GET /api/conversations/{id}` wiring tested
  against a real embedded/local Qdrant collection + fake embedding/LLM
  models, same pattern test_phase5/6/7 established.
"""

from __future__ import annotations

import shutil
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import Settings, get_settings
from app.conversations.memory import (
    REWRITE_SYSTEM_PROMPT,
    build_history_block,
    get_or_create_conversation,
    get_recent_turns,
    record_turn,
    resolve_followup,
)
from app.database.models import Base, Conversation, ConversationTurn
from app.generation.llm_client import LLMUnavailableError
from app.main import app
from app.retrieval import qdrant_store as qs
from app.retrieval.qdrant_store import reset_qdrant_client_cache

# ---------------------------------------------------------------------------
# app/conversations/memory.py — pure logic, fake LLM client
# ---------------------------------------------------------------------------


class _FakeLLMClient:
    """Returns `rewrite_answer` for the rewrite call (identified by its
    system prompt) and would raise on anything else — these unit tests
    only ever exercise `resolve_followup()`, which only ever makes the
    rewrite call."""

    def __init__(self, rewrite_answer: str | None = None, raise_unavailable: bool = False):
        self.rewrite_answer = rewrite_answer
        self.raise_unavailable = raise_unavailable
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        self.calls.append((system, user))
        if self.raise_unavailable:
            raise LLMUnavailableError("simulated provider outage")
        assert system == REWRITE_SYSTEM_PROMPT
        return self.rewrite_answer or ""


def _fake_turn(index: int, raw_query: str, answer: str) -> ConversationTurn:
    # Not persisted — resolve_followup only reads .raw_query/.answer, so a
    # plain in-memory instance (no DB session) is enough for these tests.
    return ConversationTurn(
        id=str(uuid.uuid4()),
        conversation_id="c1",
        turn_index=index,
        raw_query=raw_query,
        resolved_query=raw_query,
        answer=answer,
    )


def test_resolve_followup_skips_rewrite_when_no_history():
    llm = _FakeLLMClient(rewrite_answer="should never be used")
    settings = Settings(CONVERSATION_REWRITE_ENABLED=True)
    result = resolve_followup("What was the revenue in 2025?", [], llm, settings)
    assert result.rewritten is False
    assert result.resolved_query == "What was the revenue in 2025?"
    assert llm.calls == []  # no LLM call at all on a first turn


def test_resolve_followup_skips_rewrite_when_disabled():
    llm = _FakeLLMClient(rewrite_answer="should never be used")
    settings = Settings(CONVERSATION_REWRITE_ENABLED=False)
    history = [_fake_turn(0, "What was 2025 revenue?", "Revenue was $5M in 2025 [1].")]
    result = resolve_followup("what about 2024?", history, llm, settings)
    assert result.rewritten is False
    assert result.resolved_query == "what about 2024?"
    assert llm.calls == []


def test_resolve_followup_rewrites_elliptical_followup():
    llm = _FakeLLMClient(rewrite_answer="What was the revenue in 2024?")
    settings = Settings(CONVERSATION_REWRITE_ENABLED=True)
    history = [_fake_turn(0, "What was 2025 revenue?", "Revenue was $5M in 2025 [1].")]
    result = resolve_followup("what about 2024?", history, llm, settings)
    assert result.rewritten is True
    assert result.resolved_query == "What was the revenue in 2024?"
    assert len(llm.calls) == 1
    system, user = llm.calls[0]
    assert system == REWRITE_SYSTEM_PROMPT
    assert "What was 2025 revenue?" in user
    assert "what about 2024?" in user


def test_resolve_followup_passes_through_already_standalone_query():
    llm = _FakeLLMClient(rewrite_answer="What is the capital of France?")
    settings = Settings(CONVERSATION_REWRITE_ENABLED=True)
    history = [_fake_turn(0, "What was 2025 revenue?", "Revenue was $5M in 2025 [1].")]
    result = resolve_followup("What is the capital of France?", history, llm, settings)
    assert result.rewritten is False
    assert result.resolved_query == "What is the capital of France?"


def test_resolve_followup_strips_quotes_from_rewrite():
    llm = _FakeLLMClient(rewrite_answer='"What was the revenue in 2024?"')
    settings = Settings(CONVERSATION_REWRITE_ENABLED=True)
    history = [_fake_turn(0, "What was 2025 revenue?", "Revenue was $5M in 2025 [1].")]
    result = resolve_followup("what about 2024?", history, llm, settings)
    assert result.resolved_query == "What was the revenue in 2024?"


def test_resolve_followup_falls_back_to_raw_query_on_llm_failure():
    llm = _FakeLLMClient(raise_unavailable=True)
    settings = Settings(CONVERSATION_REWRITE_ENABLED=True)
    history = [_fake_turn(0, "What was 2025 revenue?", "Revenue was $5M in 2025 [1].")]
    result = resolve_followup("what about 2024?", history, llm, settings)
    assert result.rewritten is False
    assert result.resolved_query == "what about 2024?"


def test_resolve_followup_falls_back_to_raw_query_on_empty_rewrite():
    llm = _FakeLLMClient(rewrite_answer="   ")
    settings = Settings(CONVERSATION_REWRITE_ENABLED=True)
    history = [_fake_turn(0, "What was 2025 revenue?", "Revenue was $5M in 2025 [1].")]
    result = resolve_followup("what about 2024?", history, llm, settings)
    assert result.rewritten is False
    assert result.resolved_query == "what about 2024?"


def test_build_history_block_is_oldest_first_user_assistant_lines():
    history = [
        _fake_turn(0, "Q1?", "A1"),
        _fake_turn(1, "Q2?", "A2"),
    ]
    block = build_history_block(history)
    assert block == "User: Q1?\nAssistant: A1\nUser: Q2?\nAssistant: A2"


# ---------------------------------------------------------------------------
# get_or_create_conversation / get_recent_turns / record_turn — real DB
# ---------------------------------------------------------------------------


@pytest.fixture
def db_session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'phase8_test.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_get_or_create_conversation_creates_new_with_given_id(db_session):
    conversation = get_or_create_conversation(db_session, "my-conv-id")
    assert conversation.id == "my-conv-id"
    assert conversation.turns == []


def test_get_or_create_conversation_creates_new_with_generated_id_when_none(db_session):
    conversation = get_or_create_conversation(db_session, None)
    assert conversation.id  # a uuid was generated
    assert db_session.get(Conversation, conversation.id) is not None


def test_get_or_create_conversation_returns_existing(db_session):
    first = get_or_create_conversation(db_session, "reused-id")
    record_turn(db_session, first, raw_query="q", resolved_query="q", answer="a")
    second = get_or_create_conversation(db_session, "reused-id")
    assert second.id == first.id
    assert len(second.turns) == 1


def test_record_turn_assigns_incrementing_turn_index(db_session):
    conversation = get_or_create_conversation(db_session, "conv-idx")
    t0 = record_turn(db_session, conversation, raw_query="q0", resolved_query="q0", answer="a0")
    t1 = record_turn(db_session, conversation, raw_query="q1", resolved_query="q1", answer="a1")
    assert t0.turn_index == 0
    assert t1.turn_index == 1


def test_get_recent_turns_respects_limit_and_order(db_session):
    conversation = get_or_create_conversation(db_session, "conv-limit")
    for i in range(5):
        record_turn(db_session, conversation, raw_query=f"q{i}", resolved_query=f"q{i}", answer=f"a{i}")
    recent = get_recent_turns(db_session, conversation.id, limit=3)
    assert [t.raw_query for t in recent] == ["q2", "q3", "q4"]  # oldest-first among the 3 most recent


# ---------------------------------------------------------------------------
# POST /api/chat + GET /api/conversations/{id} — full wiring, real Qdrant
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


class _ScriptedLLMClient:
    """Distinguishes the Phase 8 rewrite call from the Phase 6 generation
    call by system prompt, so a single fake can serve both in one
    integration test — same trick as `_FakeLLMClient` above."""

    def __init__(self, rewrite_answer: str, generation_answer: str):
        self.rewrite_answer = rewrite_answer
        self.generation_answer = generation_answer
        self.calls: list[str] = []  # "rewrite" | "generation", in call order

    def complete(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        if system == REWRITE_SYSTEM_PROMPT:
            self.calls.append("rewrite")
            return self.rewrite_answer
        self.calls.append("generation")
        return self.generation_answer


@pytest.fixture
def client(tmp_path):
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
        RERANK_ENABLED=False,
    )
    app.dependency_overrides[get_settings] = lambda: test_settings
    with TestClient(app) as c:
        c.test_settings = test_settings
        yield c
    app.dependency_overrides.clear()
    reset_qdrant_client_cache()
    shutil.rmtree(qdrant_path, ignore_errors=True)


def _seed_chunk(settings: Settings, text: str) -> None:
    qclient = qs.QdrantClient(path=settings.QDRANT_LOCAL_PATH)
    qs.ensure_collection(qclient, settings.QDRANT_COLLECTION, dimension=settings.EMBEDDING_DIMENSION)
    chunk = qs.ChunkVectorPayload(str(uuid.uuid4()), "doc-1", text, 1, 0, "paragraph", "a.pdf")
    dim = settings.EMBEDDING_DIMENSION
    qs.upsert_chunks(qclient, settings.QDRANT_COLLECTION, [chunk], [[1.0] + [0.0] * (dim - 1)], [{"1": 1.0}])
    qclient.close()


def test_chat_without_conversation_id_creates_no_conversation(client, monkeypatch):
    settings = client.test_settings
    _seed_chunk(settings, "Total revenue for 2025 was $5 million.")
    monkeypatch.setattr(
        "app.api.chat.get_embedding_model", lambda **kw: _FakeEmbeddingModel(settings.EMBEDDING_DIMENSION)
    )
    fake_llm = _ScriptedLLMClient(rewrite_answer="unused", generation_answer="Revenue was $5M in 2025 [1].")
    monkeypatch.setattr("app.api.chat.get_llm_client", lambda settings: fake_llm)

    resp = client.post("/api/chat", json={"query": "What was 2025 revenue?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["conversation_id"] is None
    assert body["resolved_query"] is None
    assert fake_llm.calls == ["generation"]  # no rewrite call at all


def test_chat_first_turn_with_conversation_id_creates_conversation_no_rewrite(client, monkeypatch):
    settings = client.test_settings
    _seed_chunk(settings, "Total revenue for 2025 was $5 million.")
    monkeypatch.setattr(
        "app.api.chat.get_embedding_model", lambda **kw: _FakeEmbeddingModel(settings.EMBEDDING_DIMENSION)
    )
    fake_llm = _ScriptedLLMClient(rewrite_answer="unused", generation_answer="Revenue was $5M in 2025 [1].")
    monkeypatch.setattr("app.api.chat.get_llm_client", lambda settings: fake_llm)

    resp = client.post("/api/chat", json={"query": "What was 2025 revenue?", "conversation_id": "conv-a"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["conversation_id"] == "conv-a"
    assert body["resolved_query"] is None  # first turn: nothing to rewrite against
    assert fake_llm.calls == ["generation"]


def test_chat_followup_turn_resolves_and_records_history(client, monkeypatch):
    settings = client.test_settings
    _seed_chunk(settings, "Total revenue for 2025 was $5 million.")
    _seed_chunk(settings, "Total revenue for 2024 was $4 million.")
    monkeypatch.setattr(
        "app.api.chat.get_embedding_model", lambda **kw: _FakeEmbeddingModel(settings.EMBEDDING_DIMENSION)
    )
    fake_llm = _ScriptedLLMClient(
        rewrite_answer="What was the revenue in 2024?",
        generation_answer="Revenue was $4M in 2024 [1].",
    )
    monkeypatch.setattr("app.api.chat.get_llm_client", lambda settings: fake_llm)

    first = client.post("/api/chat", json={"query": "What was 2025 revenue?", "conversation_id": "conv-b"})
    assert first.status_code == 200

    second = client.post("/api/chat", json={"query": "what about 2024?", "conversation_id": "conv-b"})
    assert second.status_code == 200
    body = second.json()
    assert body["conversation_id"] == "conv-b"
    assert body["resolved_query"] == "What was the revenue in 2024?"
    assert fake_llm.calls == ["generation", "rewrite", "generation"]

    history = client.get("/api/conversations/conv-b")
    assert history.status_code == 200
    hbody = history.json()
    assert hbody["id"] == "conv-b"
    assert len(hbody["turns"]) == 2
    assert hbody["turns"][0]["raw_query"] == "What was 2025 revenue?"
    assert hbody["turns"][1]["raw_query"] == "what about 2024?"
    assert hbody["turns"][1]["resolved_query"] == "What was the revenue in 2024?"


def test_get_conversation_404_for_unknown_id(client):
    resp = client.get("/api/conversations/does-not-exist")
    assert resp.status_code == 404
