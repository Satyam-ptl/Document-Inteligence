"""
Phase 3 tests: OCR routing/parsing/DB-writing logic, end-to-end through the
upload endpoint, plus a couple of pure unit tests for the smaller pieces
(HTML table parsing, PDF rasterization, image preprocessing).

The real PaddleOCR-VL pipeline needs several GB of model weights that this
sandbox's network allow-list blocks (see PROJECT_STATE.md's Phase 3
verification log for the exact reproduced error: "No available model
hosting platforms detected"). So these tests monkeypatch
`app.ingestion.router.get_ocr_pipeline` to return a fake pipeline whose
`predict()` output mimics the real pipeline's documented shape (confirmed by
reading the installed `paddlex` package's own source — see
`app/ocr/paddle_ocr.py`'s docstring) rather than trying to talk to the real
one. Everything downstream of that boundary — parsing, table extraction,
DB writes, chunking, status transitions, error handling — is real code
exercised for real, only the model call itself is faked.
"""

from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.ocr.paddle_ocr import OCRUnavailableError


class _FakeEmbeddingModel:
    """Same rationale as test_phase2_ingestion.py's fake: Phase 3's own
    concern is OCR routing/parsing, not embedding, but every document now
    passes through an embedding step (Phase 4) before reaching "completed"."""

    def encode(self, texts: list[str], **kwargs):
        from app.config import get_settings

        dim = get_settings().EMBEDDING_DIMENSION
        # Phase 5 update: embed_texts() now requests return_sparse=True by
        # default, so a fake model must return a same-length "lexical_weights"
        # list too (an empty dict per text is enough — this file's concern
        # is OCR routing/parsing, not hybrid retrieval quality).
        return {"dense_vecs": [[0.0] * dim for _ in texts], "lexical_weights": [{} for _ in texts]}


@pytest.fixture(autouse=True)
def _fake_embedding_model(monkeypatch):
    monkeypatch.setattr("app.ingestion.router.get_embedding_model", lambda **kw: _FakeEmbeddingModel())


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _make_blank_pdf(num_pages: int = 1) -> bytes:
    doc = pymupdf.open()
    for _ in range(num_pages):
        doc.new_page()
    data = doc.tobytes()
    doc.close()
    return data


def _make_png_bytes() -> bytes:
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        import struct

        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    import struct

    width, height = 4, 4
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\xff\xff\xff" * width for _ in range(height))
    idat = zlib.compress(raw)
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


class _FakeBlock:
    def __init__(self, label: str, content: str, bbox=None, confidence=0.97, group_id=None):
        self.label = label
        self.content = content
        self.bbox = bbox or [10, 10, 200, 40]
        self.confidence = confidence
        self.group_id = group_id
        self.polygon_points = None
        self.image = None
        self.global_block_id = None
        self.global_group_id = None


class _FakePageResult(dict):
    """Mimics the dict-like PaddleOCRVLResult the real pipeline returns —
    supports `res["width"]`, `res["parsing_res_list"]`, etc., which is all
    `_parse_page_result` in app/ocr/paddle_ocr.py relies on."""


TABLE_HTML = (
    "<table><tr><th>Year</th><th>Revenue</th></tr>"
    "<tr><td>2024</td><td>100</td></tr>"
    "<tr><td>2025</td><td>150</td></tr></table>"
)


def _make_fake_pipeline(blocks_per_page: list[list[_FakeBlock]], width=800, height=1100):
    class _FakePipeline:
        def predict(self, input: str):  # noqa: A002
            idx = self.call_count if hasattr(self, "call_count") else 0
            self.call_count = idx + 1
            blocks = blocks_per_page[idx] if idx < len(blocks_per_page) else []
            res = _FakePageResult(width=width, height=height, parsing_res_list=blocks)
            return [res]

    return _FakePipeline()


def test_scanned_pdf_is_routed_through_ocr_and_completes(client, monkeypatch) -> None:
    blocks = [[
        _FakeBlock("paragraph_title", "Annual Report"),
        _FakeBlock("text", "Revenue grew significantly this year."),
        _FakeBlock("table", TABLE_HTML),
    ]]
    fake_pipeline = _make_fake_pipeline(blocks)
    monkeypatch.setattr("app.ingestion.router.get_ocr_pipeline", lambda **kw: fake_pipeline)

    pdf_bytes = _make_blank_pdf(num_pages=1)
    resp = client.post(
        "/api/documents",
        files={"file": ("scanned.pdf", pdf_bytes, "application/pdf")},
    )
    assert resp.status_code == 201, resp.text
    doc_id = resp.json()["id"]

    doc = client.get(f"/api/documents/{doc_id}").json()
    assert doc["processing_status"] == "completed", doc
    assert doc["source_type"] == "scanned_pdf"
    assert doc["processing_method"] == "paddleocr_vl"

    chunks = client.get(f"/api/documents/{doc_id}/chunks").json()
    assert any(c["element_type"] == "table" for c in chunks)
    table_chunk = next(c for c in chunks if c["element_type"] == "table")
    assert "2024" in table_chunk["text"] and "2025" in table_chunk["text"]
    assert any("Revenue grew" in c["text"] for c in chunks if c["element_type"] == "paragraph")


def test_image_upload_is_routed_through_ocr_and_completes(client, monkeypatch) -> None:
    blocks = [[_FakeBlock("text", "Invoice total: $4,200.00")]]
    fake_pipeline = _make_fake_pipeline(blocks)
    monkeypatch.setattr("app.ingestion.router.get_ocr_pipeline", lambda **kw: fake_pipeline)

    resp = client.post(
        "/api/documents",
        files={"file": ("scan.png", _make_png_bytes(), "image/png")},
    )
    assert resp.status_code == 201, resp.text
    doc_id = resp.json()["id"]

    doc = client.get(f"/api/documents/{doc_id}").json()
    assert doc["processing_status"] == "completed", doc
    assert doc["source_type"] == "image"
    assert doc["page_count"] == 1

    chunks = client.get(f"/api/documents/{doc_id}/chunks").json()
    assert any("Invoice total" in c["text"] for c in chunks)


def test_ocr_unavailable_marks_document_failed_not_crash(client, monkeypatch) -> None:
    def _raise(**kw):
        raise OCRUnavailableError("No available model hosting platforms detected.")

    monkeypatch.setattr("app.ingestion.router.get_ocr_pipeline", _raise)

    pdf_bytes = _make_blank_pdf(num_pages=1)
    resp = client.post(
        "/api/documents",
        files={"file": ("scanned2.pdf", pdf_bytes, "application/pdf")},
    )
    assert resp.status_code == 201, resp.text
    doc_id = resp.json()["id"]

    doc = client.get(f"/api/documents/{doc_id}").json()
    assert doc["processing_status"] == "failed"
    assert "OCR unavailable" in doc["processing_error"]


def test_ocr_predict_failure_mid_run_marks_failed_not_crash(client, monkeypatch) -> None:
    class _CrashingPipeline:
        def predict(self, input: str):  # noqa: A002
            raise RuntimeError("simulated model crash")

    monkeypatch.setattr("app.ingestion.router.get_ocr_pipeline", lambda **kw: _CrashingPipeline())

    pdf_bytes = _make_blank_pdf(num_pages=1)
    resp = client.post(
        "/api/documents",
        files={"file": ("scanned3.pdf", pdf_bytes, "application/pdf")},
    )
    doc_id = resp.json()["id"]
    doc = client.get(f"/api/documents/{doc_id}").json()
    assert doc["processing_status"] == "failed"
    assert "OCR unavailable" in doc["processing_error"]


# ---------------------------------------------------------------------------
# Pure unit tests (no HTTP, no fakes needed — real code, real libraries)
# ---------------------------------------------------------------------------


def test_parse_html_table_extracts_headers_and_rows() -> None:
    from app.ocr.paddle_ocr import _parse_html_table

    parsed = _parse_html_table(TABLE_HTML)
    assert parsed == {"headers": ["Year", "Revenue"], "rows": [["2024", "100"], ["2025", "150"]]}


def test_parse_html_table_returns_none_for_non_table_content() -> None:
    from app.ocr.paddle_ocr import _parse_html_table

    assert _parse_html_table("just some plain text") is None
    assert _parse_html_table("") is None


def test_classify_block_maps_labels_to_element_types() -> None:
    from app.ocr.paddle_ocr import _classify_block

    assert _classify_block("table", TABLE_HTML)[0] == "table"
    assert _classify_block("formula", "E=mc^2")[0] == "formula"
    assert _classify_block("paragraph_title", "Intro")[0] == "heading"
    assert _classify_block("chart", "")[0] == "image"
    assert _classify_block("text", "hello")[0] == "paragraph"


def test_rasterize_pdf_pages_writes_one_png_per_page(tmp_path: Path) -> None:
    from app.ocr.rasterizer import rasterize_pdf_pages

    pdf_path = tmp_path / "in.pdf"
    pdf_path.write_bytes(_make_blank_pdf(num_pages=3))

    out_dir = tmp_path / "out"
    paths = rasterize_pdf_pages(pdf_path, out_dir, dpi=72)

    assert len(paths) == 3
    for p in paths:
        assert p.exists()
        assert p.suffix == ".png"
        assert p.stat().st_size > 0


def test_preprocess_image_runs_without_crashing_on_real_image(tmp_path: Path) -> None:
    from app.ocr.preprocessing import preprocess_image

    img_path = tmp_path / "test.png"
    img_path.write_bytes(_make_png_bytes())

    result_path = preprocess_image(img_path)
    assert result_path == img_path
    assert img_path.exists()
    assert img_path.stat().st_size > 0


def test_preprocess_image_handles_unreadable_file_gracefully(tmp_path: Path) -> None:
    from app.ocr.preprocessing import preprocess_image

    bad_path = tmp_path / "not_an_image.png"
    bad_path.write_bytes(b"this is not image data")

    # Must not raise — preprocessing failures fall back to the original file.
    result_path = preprocess_image(bad_path)
    assert result_path == bad_path


def test_ocr_page_result_raw_text_flattens_table_to_readable_text() -> None:
    from app.ocr.paddle_ocr import OCRElement, OCRPageResult

    page = OCRPageResult(
        page_number=1,
        width=800,
        height=1100,
        elements=[
            OCRElement(element_type="heading", text="Report", bbox=None, confidence=0.9),
            OCRElement(
                element_type="table",
                text=TABLE_HTML,
                bbox=None,
                confidence=0.9,
                table_data={"headers": ["Year", "Revenue"], "rows": [["2024", "100"]]},
            ),
        ],
    )
    assert "Report" in page.raw_text
    assert "Year | Revenue" in page.raw_text
    assert "2024 | 100" in page.raw_text
