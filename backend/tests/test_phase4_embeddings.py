"""
Phase 4 tests: embeddings + Qdrant indexing.

Two very different boundaries here, tested differently on purpose:

- `app/retrieval/qdrant_store.py`: the real dependency (`qdrant-client`'s
  embedded/local mode) is NOT blocked in this sandbox — no server, no
  Docker, no network needed. So this module's tests hit real Qdrant, for
  real, with no mocking at all (`test_qdrant_store_*` below).
- `app/embeddings/embedder.py`: the real dependency (`FlagEmbedding` +
  `torch`) could NOT be installed here — `torch` alone failed with "No
  space left on device" (see PROJECT_STATE.md's Phase 4 verification log
  for the exact reproduced error and `requirements.txt`'s comment for
  what's needed to actually run it). So router-level tests monkeypatch
  `app.ingestion.router.get_embedding_model` to a fake model whose
  `.encode()` mimics bge-m3's real, source-confirmed return shape
  (`{"dense_vecs": ...}`) — everything downstream of that one call
  (chunk-batching, Qdrant upsert, `Chunk.embedded`/`qdrant_point_id`
  bookkeeping, status transitions, error handling) is real code, genuinely
  exercised.
"""

from __future__ import annotations

import io
import shutil
import tempfile
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.embeddings.embedder import EmbeddingUnavailableError, embed_texts
from app.main import app
from app.retrieval import qdrant_store as qs


# --------------------------------------------------------------------------
# Pure unit tests: app/retrieval/qdrant_store.py against a REAL, local,
# on-disk Qdrant instance (no server, no network, no mocking).
# --------------------------------------------------------------------------


@pytest.fixture
def qdrant_tmp_client():
    """A real QdrantClient in local/embedded mode, pointed at a throwaway
    temp directory so this test suite never touches the app's own
    `data/qdrant/` dev directory or collides with other tests' points."""
    tmp_dir = tempfile.mkdtemp()
    client = qs.QdrantClient(path=tmp_dir)
    yield client
    client.close()
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_ensure_collection_is_idempotent(qdrant_tmp_client) -> None:
    assert qdrant_tmp_client.collection_exists("test_collection") is False
    qs.ensure_collection(qdrant_tmp_client, "test_collection", dimension=4)
    assert qdrant_tmp_client.collection_exists("test_collection") is True
    # Calling it again must not raise (idempotent, safe on every ingestion run).
    qs.ensure_collection(qdrant_tmp_client, "test_collection", dimension=4)


def test_upsert_and_search_roundtrip(qdrant_tmp_client) -> None:
    qs.ensure_collection(qdrant_tmp_client, "chunks", dimension=4)

    chunks = [
        qs.ChunkVectorPayload(
            chunk_id="11111111-1111-1111-1111-111111111111",
            document_id="doc-a",
            text="revenue grew 12% in fiscal year 2025",
            page_number=1,
            chunk_index=0,
            element_type="paragraph",
            source_filename="report.pdf",
        ),
        qs.ChunkVectorPayload(
            chunk_id="22222222-2222-2222-2222-222222222222",
            document_id="doc-b",
            text="unrelated chunk from a different document",
            page_number=3,
            chunk_index=5,
            element_type="paragraph",
            source_filename="other.pdf",
        ),
    ]
    vectors = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]

    point_ids = qs.upsert_chunks(qdrant_tmp_client, "chunks", chunks, vectors)
    assert point_ids == [c.chunk_id for c in chunks]

    hits = qs.search(qdrant_tmp_client, "chunks", query_vector=[1.0, 0.0, 0.0, 0.0], top_k=2)
    assert len(hits) == 2
    assert hits[0]["chunk_id"] == "11111111-1111-1111-1111-111111111111"
    assert hits[0]["document_id"] == "doc-a"
    assert hits[0]["text"] == "revenue grew 12% in fiscal year 2025"
    assert hits[0]["score"] > hits[1]["score"]


def test_search_scoped_to_document_id(qdrant_tmp_client) -> None:
    qs.ensure_collection(qdrant_tmp_client, "chunks", dimension=2)
    chunks = [
        qs.ChunkVectorPayload("33333333-3333-3333-3333-333333333333", "doc-a", "text a", 1, 0, "paragraph", "a.pdf"),
        qs.ChunkVectorPayload("44444444-4444-4444-4444-444444444444", "doc-b", "text b", 1, 0, "paragraph", "b.pdf"),
    ]
    qs.upsert_chunks(qdrant_tmp_client, "chunks", chunks, [[1.0, 0.0], [1.0, 0.0]])

    hits = qs.search(qdrant_tmp_client, "chunks", query_vector=[1.0, 0.0], top_k=10, document_id="doc-b")
    assert len(hits) == 1
    assert hits[0]["document_id"] == "doc-b"


def test_delete_document_vectors_removes_only_that_document(qdrant_tmp_client) -> None:
    qs.ensure_collection(qdrant_tmp_client, "chunks", dimension=2)
    chunks = [
        qs.ChunkVectorPayload("33333333-3333-3333-3333-333333333333", "doc-a", "text a", 1, 0, "paragraph", "a.pdf"),
        qs.ChunkVectorPayload("44444444-4444-4444-4444-444444444444", "doc-b", "text b", 1, 0, "paragraph", "b.pdf"),
    ]
    qs.upsert_chunks(qdrant_tmp_client, "chunks", chunks, [[1.0, 0.0], [0.0, 1.0]])

    qs.delete_document_vectors(qdrant_tmp_client, "chunks", "doc-a")

    remaining = qs.search(qdrant_tmp_client, "chunks", query_vector=[1.0, 0.0], top_k=10)
    assert [h["document_id"] for h in remaining] == ["doc-b"]


def test_delete_document_vectors_on_missing_collection_is_a_noop(qdrant_tmp_client) -> None:
    # Must not raise even if the collection was never created (e.g. a
    # document that failed before ever reaching the embedding step).
    qs.delete_document_vectors(qdrant_tmp_client, "never_created", "doc-x")


def test_upsert_chunks_rejects_mismatched_lengths(qdrant_tmp_client) -> None:
    qs.ensure_collection(qdrant_tmp_client, "chunks", dimension=2)
    chunks = [qs.ChunkVectorPayload("33333333-3333-3333-3333-333333333333", "doc-a", "t", 1, 0, "paragraph", "a.pdf")]
    with pytest.raises(ValueError):
        qs.upsert_chunks(qdrant_tmp_client, "chunks", chunks, vectors=[[1.0, 0.0], [0.0, 1.0]])


def test_upsert_chunks_with_empty_list_returns_empty(qdrant_tmp_client) -> None:
    assert qs.upsert_chunks(qdrant_tmp_client, "chunks", [], []) == []


# --------------------------------------------------------------------------
# app/embeddings/embedder.py: the real model can't be installed here (see
# module docstring), so these test the boundary itself plus embed_texts()
# against a fake model that mimics bge-m3's real, source-confirmed return
# shape.
# --------------------------------------------------------------------------


class _FakeEmbeddingModel:
    """Mimics BGEM3FlagModel.encode()'s real return shape: a dict with a
    'dense_vecs' key (confirmed by reading FlagEmbedding's source directly —
    see embedder.py's docstring for how, since the package itself couldn't
    be installed here to run for real). Returns a fixed-direction vector
    derived from each text's length so different texts are distinguishable
    in tests without needing a real model."""

    def __init__(self, dimension: int = 1024):
        self.dimension = dimension
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str], **kwargs):
        self.calls.append(texts)
        dense = []
        for t in texts:
            vec = [0.0] * self.dimension
            vec[len(t) % self.dimension] = 1.0
            dense.append(vec)
        # Phase 5 update: embed_texts() now requests return_sparse=True by
        # default, so this fake must also return a same-length
        # "lexical_weights" list (see FlagEmbedding's real, source-confirmed
        # shape in embedder.py's docstring) for the full-pipeline tests
        # below, which now index both a dense and a sparse vector per chunk.
        lexical_weights = [{str(i % 50): 1.0} for i in range(len(texts))]
        return {"dense_vecs": dense, "lexical_weights": lexical_weights}


class _CrashingEmbeddingModel:
    def encode(self, texts: list[str], **kwargs):
        raise RuntimeError("simulated embedding backend crash")


def test_embed_texts_returns_dense_vectors_from_fake_model() -> None:
    model = _FakeEmbeddingModel(dimension=8)
    result = embed_texts(["hello", "a longer chunk of text"], model=model, batch_size=2)
    assert len(result.dense) == 2
    assert all(len(vec) == 8 for vec in result.dense)
    assert model.calls == [["hello", "a longer chunk of text"]]


def test_embed_texts_with_empty_list_does_not_call_model() -> None:
    model = _FakeEmbeddingModel()
    result = embed_texts([], model=model)
    assert result.dense == []
    assert model.calls == []


def test_embed_texts_wraps_model_crash_as_embedding_unavailable() -> None:
    with pytest.raises(EmbeddingUnavailableError):
        embed_texts(["some text"], model=_CrashingEmbeddingModel())


def test_get_embedding_model_uses_local_fallback_when_flagembedding_missing() -> None:
    """Offline development remains usable when the heavyweight model is absent."""
    from app.embeddings.embedder import get_embedding_model

    model = get_embedding_model()
    output = model.encode(["revenue increased in 2025"])
    assert len(output["dense_vecs"]) == 1
    assert len(output["dense_vecs"][0]) == 1024
    assert output["lexical_weights"]


# --------------------------------------------------------------------------
# End-to-end: real upload -> real PyMuPDF extraction -> real chunking ->
# [faked] embedding call -> real Qdrant indexing -> real DB bookkeeping,
# through the actual FastAPI background-task path.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    with TestClient(app_module()) as c:
        yield c


def app_module():
    from app.main import app

    return app


def _make_text_pdf(paragraphs: list[str]) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    rect = pymupdf.Rect(72, 72, page.rect.width - 72, page.rect.height - 72)
    page.insert_textbox(rect, "\n".join(paragraphs), fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def test_full_pipeline_embeds_and_indexes_with_fake_model(client, monkeypatch) -> None:
    fake_model = _FakeEmbeddingModel(dimension=get_settings().EMBEDDING_DIMENSION)
    monkeypatch.setattr("app.ingestion.router.get_embedding_model", lambda **kw: fake_model)

    pdf_bytes = _make_text_pdf(["Q3 revenue grew 15% year over year, reaching $4.2 million."])
    resp = client.post(
        "/api/documents",
        files={"file": ("q3.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert resp.status_code == 201
    doc_id = resp.json()["id"]

    doc = client.get(f"/api/documents/{doc_id}").json()
    assert doc["processing_status"] == "completed", doc.get("processing_error")

    chunks = client.get(f"/api/documents/{doc_id}/chunks").json()
    assert len(chunks) >= 1
    assert all(c["embedded"] is True for c in chunks)

    # Real Qdrant search (same client/collection app code uses) should
    # actually find this document's chunk by its embedded vector.
    settings = get_settings()
    qclient = qs.get_qdrant_client(settings)
    hits = qs.search(
        qclient,
        settings.QDRANT_COLLECTION,
        query_vector=fake_model.encode(["Q3 revenue grew 15% year over year, reaching $4.2 million."])[
            "dense_vecs"
        ][0],
        top_k=5,
        document_id=doc_id,
    )
    assert len(hits) >= 1
    assert hits[0]["document_id"] == doc_id


def test_embedding_unavailable_marks_document_failed_not_crash(client, monkeypatch) -> None:
    def _raise(**kwargs):
        raise EmbeddingUnavailableError("simulated: model weights unreachable")

    monkeypatch.setattr("app.ingestion.router.get_embedding_model", _raise)

    pdf_bytes = _make_text_pdf(["Some perfectly normal extractable text."])
    resp = client.post(
        "/api/documents",
        files={"file": ("fails.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    doc_id = resp.json()["id"]

    doc = client.get(f"/api/documents/{doc_id}").json()
    assert doc["processing_status"] == "failed"
    assert "Embedding unavailable" in doc["processing_error"]

    # Chunks must have been written (chunking happens before embedding) but
    # never marked embedded, since the embedding call never succeeded.
    chunks = client.get(f"/api/documents/{doc_id}/chunks").json()
    assert len(chunks) >= 1
    assert all(c["embedded"] is False for c in chunks)


def test_delete_document_cleans_up_qdrant_vectors(client, monkeypatch) -> None:
    fake_model = _FakeEmbeddingModel(dimension=get_settings().EMBEDDING_DIMENSION)
    monkeypatch.setattr("app.ingestion.router.get_embedding_model", lambda **kw: fake_model)

    pdf_bytes = _make_text_pdf(["Content that will shortly be deleted entirely."])
    resp = client.post(
        "/api/documents",
        files={"file": ("temp.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    doc_id = resp.json()["id"]
    assert client.get(f"/api/documents/{doc_id}").json()["processing_status"] == "completed"

    settings = get_settings()
    qclient = qs.get_qdrant_client(settings)
    probe_vector = [1.0] + [0.0] * (settings.EMBEDDING_DIMENSION - 1)
    before = qs.search(qclient, settings.QDRANT_COLLECTION, query_vector=probe_vector, top_k=50, document_id=doc_id)
    assert len(before) >= 1

    del_resp = client.delete(f"/api/documents/{doc_id}")
    assert del_resp.status_code == 204

    after = qs.search(qclient, settings.QDRANT_COLLECTION, query_vector=probe_vector, top_k=50, document_id=doc_id)
    assert after == []
