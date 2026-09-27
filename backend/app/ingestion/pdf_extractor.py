"""
PyMuPDF (pymupdf) based extraction for text-based PDFs.

Phase 2 scope: extract per-page raw text + page dimensions, and detect
whether a PDF is actually a "scanned" PDF (i.e. pages that are essentially
images with little or no embedded text layer) so it can be routed to the
PaddleOCR-VL pipeline in Phase 3 instead of being silently ingested as
near-empty text.
"""

from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass
class PageExtraction:
    page_number: int  # 1-indexed
    width: float
    height: float
    text: str


@dataclass
class PdfExtractionResult:
    pages: list[PageExtraction]
    is_scanned: bool


def extract_pdf(path: Path, min_text_chars_per_page: int) -> PdfExtractionResult:
    """
    Open a PDF and extract text page-by-page with PyMuPDF.

    Raises ValueError if the file cannot be opened as a PDF or has zero pages
    (the caller is expected to catch this and mark the document as failed).

    A PDF is flagged `is_scanned=True` when its average extractable text per
    page falls below `min_text_chars_per_page` — a strong signal the pages
    are scanned images with no embedded text layer, which PyMuPDF cannot
    read (that's Phase 3's job, via PaddleOCR-VL).
    """
    try:
        # Open from bytes so a failed parse cannot leave PyMuPDF holding a
        # Windows file handle when the upload is immediately deleted.
        doc = pymupdf.open(stream=path.read_bytes(), filetype="pdf")
    except Exception as exc:  # pymupdf raises its own exception types
        raise ValueError(f"Could not open file as a PDF: {exc}") from exc

    try:
        if doc.page_count == 0:
            raise ValueError("PDF has no pages.")

        pages: list[PageExtraction] = []
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = page.get_text("text")
            pages.append(
                PageExtraction(
                    page_number=i + 1,
                    width=page.rect.width,
                    height=page.rect.height,
                    text=text,
                )
            )
    finally:
        doc.close()

    total_chars = sum(len(p.text.strip()) for p in pages)
    avg_chars_per_page = total_chars / len(pages)
    is_scanned = avg_chars_per_page < min_text_chars_per_page

    return PdfExtractionResult(pages=pages, is_scanned=is_scanned)
