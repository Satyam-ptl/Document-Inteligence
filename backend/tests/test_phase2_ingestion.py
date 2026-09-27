"""
Phase 2 tests: real PyMuPDF extraction + naive chunking, driven end-to-end
through the upload endpoint (BackgroundTasks run synchronously inside
Starlette's TestClient, so by the time `client.post(...)` returns, ingestion
has already completed or failed).

We build real, valid PDFs in-memory with pymupdf itself rather than faking
PDF bytes, since Phase 1's fake "%PDF-1.4 ..." bytes aren't parseable and
would only ever exercise the failure path.
"""

import io

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.database.session import SessionLocal
from app.main import app


class _FakeEmbeddingModel:
    """Phase 2's own concern is PyMuPDF extraction + chunking, not
    embedding — but since Phase 4, every document that finishes chunking
    also goes through an embedding step before reaching "completed", and
    the real embedding model (FlagEmbedding/bge-m3) isn't installable in
    this sandbox (see app/embeddings/embedder.py's docstring). This fake
    keeps these tests' original job (confirming Phase 2's own logic) intact
    without needing to know anything about Phase 4's embedding output
    beyond returning a vector of the right dimension."""

    def encode(self, texts: list[str], **kwargs):
        from app.config import get_settings

        dim = get_settings().EMBEDDING_DIMENSION
        # Phase 5 update: embed_texts() now requests return_sparse=True by
        # default, so a fake model must return a same-length "lexical_weights"
        # list too (an empty dict per text is enough — this file's concern
        # is PyMuPDF extraction/chunking, not hybrid retrieval quality).
        return {"dense_vecs": [[0.0] * dim for _ in texts], "lexical_weights": [{} for _ in texts]}


@pytest.fixture(autouse=True)
def _fake_embedding_model(monkeypatch):
    monkeypatch.setattr("app.ingestion.router.get_embedding_model", lambda **kw: _FakeEmbeddingModel())


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _make_text_pdf(paragraphs_per_page: list[list[str]]) -> bytes:
    """Build a real multi-page PDF with actual embedded text using pymupdf.

    Uses insert_textbox (wrapped, within a rect) rather than insert_text
    (single unwrapped line). insert_text draws one continuous line that
    runs off the right edge of the page and gets clipped there, so a long
    paragraph silently loses most of its extractable text — insert_textbox
    wraps within the page's margins so the full string is actually
    extractable via get_text().
    """
    doc = pymupdf.open()
    for paragraphs in paragraphs_per_page:
        page = doc.new_page()
        rect = pymupdf.Rect(72, 72, page.rect.width - 72, page.rect.height - 72)
        page.insert_textbox(rect, "\n".join(paragraphs), fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def _make_blank_pdf(num_pages: int = 1) -> bytes:
    """Build a real PDF with pages but no text at all (simulates a scanned PDF)."""
    doc = pymupdf.open()
    for _ in range(num_pages):
        doc.new_page()
    data = doc.tobytes()
    doc.close()
    return data


def test_text_pdf_is_extracted_and_chunked(client) -> None:
    long_paragraph = "The quick brown fox jumps over the lazy dog. " * 40  # >1000 chars
    pdf_bytes = _make_text_pdf(
        [
            ["Executive Summary", long_paragraph],
            ["Second page heading", "A short second page."],
        ]
    )

    resp = client.post(
        "/api/documents",
        files={"file": ("report.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert resp.status_code == 201
    doc_id = resp.json()["id"]

    get_resp = client.get(f"/api/documents/{doc_id}")
    assert get_resp.status_code == 200
    doc = get_resp.json()

    assert doc["processing_status"] == "completed", doc.get("processing_error")
    assert doc["source_type"] == "text_pdf"
    assert doc["processing_method"] == "pymupdf"
    assert doc["page_count"] == 2

    pages_resp = client.get(f"/api/documents/{doc_id}/pages")
    assert pages_resp.status_code == 200
    pages = pages_resp.json()
    assert len(pages) == 2
    assert pages[0]["page_number"] == 1
    assert "Executive Summary" in pages[0]["raw_text"]
    assert pages[1]["page_number"] == 2

    chunks_resp = client.get(f"/api/documents/{doc_id}/chunks")
    assert chunks_resp.status_code == 200
    chunks = chunks_resp.json()
    # Long first page (>1000 chars) must have been split into >1 chunk;
    # short second page should be exactly one chunk.
    page_1_chunks = [c for c in chunks if c["page_number"] == 1]
    page_2_chunks = [c for c in chunks if c["page_number"] == 2]
    assert len(page_1_chunks) > 1
    assert len(page_2_chunks) == 1
    # chunk_index must be strictly increasing across the whole document
    indices = [c["chunk_index"] for c in chunks]
    assert indices == sorted(indices)
    # Phase 4 note: every chunk now gets embedded+indexed as part of the
    # same pipeline run (with a fake embedding model in this test file —
    # see the module-level `_fake_embedding_model` fixture), so this is now
    # True rather than the pre-Phase-4 default of False.
    assert all(c["embedded"] is True for c in chunks)

    client.delete(f"/api/documents/{doc_id}")


def test_scanned_pdf_is_routed_to_ocr_pipeline(client, monkeypatch) -> None:
    """Phase 2 owns detection/routing (is this a scanned PDF, and does it
    reach the OCR code path at all?) — the actual OCR pipeline behavior
    (parsing, tables, error handling) is Phase 3's job and is covered in
    tests/test_phase3_ocr.py. Before Phase 3, a scanned PDF stopped
    permanently at a holding status; now it's actually routed into OCR, so
    this monkeypatches a trivial fake pipeline just to confirm routing +
    source_type detection without re-testing OCR internals here.
    """

    class _TrivialPipeline:
        def predict(self, input):  # noqa: A002
            return [{"width": 100, "height": 100, "parsing_res_list": []}]

    monkeypatch.setattr("app.ingestion.router.get_ocr_pipeline", lambda **kw: _TrivialPipeline())

    pdf_bytes = _make_blank_pdf(num_pages=3)

    resp = client.post(
        "/api/documents",
        files={"file": ("scan.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert resp.status_code == 201
    doc_id = resp.json()["id"]

    get_resp = client.get(f"/api/documents/{doc_id}")
    doc = get_resp.json()

    assert doc["processing_status"] == "completed"
    assert doc["source_type"] == "scanned_pdf"
    assert doc["processing_method"] == "paddleocr_vl"
    assert doc["page_count"] == 3

    # 3 pages went through the OCR path (empty layout -> empty pages/chunks,
    # but 3 DocumentPage rows should exist, one per rasterized page).
    assert len(client.get(f"/api/documents/{doc_id}/pages").json()) == 3

    client.delete(f"/api/documents/{doc_id}")


def test_corrupt_pdf_upload_marks_document_failed(client) -> None:
    # Passes the extension/size checks but isn't a real PDF, so extraction
    # must fail cleanly (not crash the background task or the server).
    resp = client.post(
        "/api/documents",
        files={"file": ("broken.pdf", io.BytesIO(b"%PDF-1.4 this is not really a pdf"), "application/pdf")},
    )
    assert resp.status_code == 201
    doc_id = resp.json()["id"]

    get_resp = client.get(f"/api/documents/{doc_id}")
    doc = get_resp.json()
    assert doc["processing_status"] == "failed"
    assert doc["processing_error"]

    client.delete(f"/api/documents/{doc_id}")


def test_image_upload_is_routed_to_ocr_pipeline(client, monkeypatch) -> None:
    class _TrivialPipeline:
        def predict(self, input):  # noqa: A002
            return [{"width": 1, "height": 1, "parsing_res_list": []}]

    monkeypatch.setattr("app.ingestion.router.get_ocr_pipeline", lambda **kw: _TrivialPipeline())

    # A minimal valid 1x1 red PNG (regenerated with zlib/struct; the
    # original hardcoded hex string had a typo and wasn't valid hex).
    png_bytes = bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de"
        "0000000c49444154789c63f8cfc0000003010100c9fe92ef0000000049454e44ae"
        "426082"
    )
    resp = client.post(
        "/api/documents",
        files={"file": ("photo.png", io.BytesIO(png_bytes), "image/png")},
    )
    assert resp.status_code == 201
    doc_id = resp.json()["id"]

    doc = client.get(f"/api/documents/{doc_id}").json()
    assert doc["processing_status"] == "completed"
    assert doc["source_type"] == "image"
    assert doc["processing_method"] == "paddleocr_vl"

    client.delete(f"/api/documents/{doc_id}")


def test_docx_upload_extracts_paragraphs_and_tables(client) -> None:
    from docx import Document

    document = Document()
    document.add_paragraph("Yuvak Bharati is the English textbook for Standard XII.")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Year"
    table.cell(0, 1).text = "Revenue"
    table.cell(1, 0).text = "2025"
    table.cell(1, 1).text = "150"
    output = io.BytesIO()
    document.save(output)

    resp = client.post(
        "/api/documents",
        files={
            "file": (
                "notes.docx",
                io.BytesIO(output.getvalue()),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )
    assert resp.status_code == 201
    doc_id = resp.json()["id"]

    doc = client.get(f"/api/documents/{doc_id}").json()
    assert doc["processing_status"] == "completed"
    assert doc["source_type"] == "docx"
    assert doc["processing_method"] == "python_docx"
    chunks = client.get(f"/api/documents/{doc_id}/chunks").json()
    assert "Yuvak Bharati" in chunks[0]["text"]
    assert "2025 | 150" in chunks[0]["text"]

    client.delete(f"/api/documents/{doc_id}")


def test_csv_upload_is_extracted_as_searchable_table(client) -> None:
    csv_bytes = b"Year,Revenue\n2024,100\n2025,150\n"
    resp = client.post(
        "/api/documents",
        files={"file": ("revenue.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert resp.status_code == 201, resp.text
    doc_id = resp.json()["id"]

    doc = client.get(f"/api/documents/{doc_id}").json()
    assert doc["processing_status"] == "completed"
    assert doc["source_type"] == "csv"
    assert doc["processing_method"] == "pandas"
    assert doc["page_count"] == 1

    chunks = client.get(f"/api/documents/{doc_id}/chunks").json()
    assert len(chunks) == 1
    assert chunks[0]["element_type"] == "table"
    assert "2025 | 150" in chunks[0]["text"]

    client.delete(f"/api/documents/{doc_id}")


def test_xlsx_upload_preserves_worksheet_tables(client) -> None:
    from openpyxl import Workbook

    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "Revenue"
    worksheet.append(["Year", "Revenue"])
    worksheet.append([2024, 100])
    worksheet.append([2025, 150])
    output = io.BytesIO()
    workbook.save(output)

    resp = client.post(
        "/api/documents",
        files={
            "file": (
                "revenue.xlsx",
                io.BytesIO(output.getvalue()),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )
    assert resp.status_code == 201, resp.text
    doc_id = resp.json()["id"]

    doc = client.get(f"/api/documents/{doc_id}").json()
    assert doc["processing_status"] == "completed"
    assert doc["source_type"] == "xlsx"
    assert doc["page_count"] == 1
    chunks = client.get(f"/api/documents/{doc_id}/chunks").json()
    assert chunks[0]["element_type"] == "table"
    assert "2024 | 100" in chunks[0]["text"]

    client.delete(f"/api/documents/{doc_id}")


def test_naive_chunker_respects_chunk_overlap_directly() -> None:
    """Unit-level check of the chunker itself, independent of the API."""
    from app.chunking.naive_chunker import chunk_pages
    from app.ingestion.pdf_extractor import PageExtraction

    text = "A" * 2500
    pages = [PageExtraction(page_number=1, width=612, height=792, text=text)]
    chunks = chunk_pages(pages, chunk_size=1000, chunk_overlap=100)

    assert len(chunks) == 3  # 1000, 1000, 500-ish with a 900-char step
    assert all(len(c.text) <= 1000 for c in chunks)
    assert [c.chunk_index for c in chunks] == [0, 1, 2]


def test_chunk_overlap_must_be_smaller_than_chunk_size() -> None:
    from app.chunking.naive_chunker import chunk_pages
    from app.ingestion.pdf_extractor import PageExtraction

    pages = [PageExtraction(page_number=1, width=612, height=792, text="hello")]
    with pytest.raises(ValueError):
        chunk_pages(pages, chunk_size=100, chunk_overlap=100)


def test_db_rows_are_actually_persisted_not_just_api_shaped(client) -> None:
    """Sanity check straight against the DB, bypassing the API response shaping."""
    from app.database.models import Chunk, DocumentPage

    pdf_bytes = _make_text_pdf([["Just one short page of text."]])
    resp = client.post(
        "/api/documents",
        files={"file": ("tiny.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    doc_id = resp.json()["id"]

    db = SessionLocal()
    try:
        pages = db.query(DocumentPage).filter(DocumentPage.document_id == doc_id).all()
        chunks = db.query(Chunk).filter(Chunk.document_id == doc_id).all()
        assert len(pages) == 1
        assert len(chunks) == 1
        assert chunks[0].source_filename == "tiny.pdf"
    finally:
        db.close()

    client.delete(f"/api/documents/{doc_id}")
