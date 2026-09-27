"""
PaddleOCR-VL-1.6 wrapper: scanned-PDF page images and standalone images go
through here to become structured layout elements (text, headings, tables,
formulas, figures) with bounding boxes and confidence, instead of raw text.

Design notes (read before touching this file):

- The real `PaddleOCRVL` pipeline downloads several GB of model weights from
  Paddle's/HuggingFace's model hosting on first use. In *this* build/dev
  sandbox, that download is blocked (see PROJECT_STATE.md's Phase 3
  verification log for the exact reproduced error) — outbound network is
  allow-listed and none of the model-hosting hosts are on the list. That is
  a sandbox limitation, not a code bug: `pip install paddleocr[doc-parser]`
  and `from paddleocr import PaddleOCRVL` both work fine, and the failure
  only happens when the pipeline tries to fetch weights.
- Because of that, this module is written so the pipeline object is created
  by a single factory function (`get_ocr_pipeline`) and every call site goes
  through `run_ocr_on_images`, which takes the pipeline as a parameter. Tests
  inject a fake pipeline that mimics the real output shape (see
  tests/test_phase3_ocr.py), so the parsing/routing/DB-writing logic is
  fully exercised without needing the real weights. Whoever runs this next
  in an environment with open network access should just delete this
  docstring's caveat once `get_ocr_pipeline()` has been confirmed to
  actually download weights and `run_ocr_on_images` has been re-verified
  against a real scanned PDF.
- Output shape this module depends on (confirmed by reading the installed
  `paddlex.inference.pipelines.paddleocr_vl.result` source, since the
  pipeline itself couldn't be run — see verification log): each page result
  is dict-like with a `parsing_res_list` key, a list of blocks, each having
  `.label` (e.g. "text", "paragraph_title", "table", "formula", "chart",
  "image", "seal", ...), `.content` (plain text, or an HTML table string
  when label == "table"), `.bbox` ([x0, y0, x1, y1] ints), and an
  auto-incrementing reading-order index across the page.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from loguru import logger

# Labels PaddleOCR-VL emits that we fold into our own DocumentElement types.
_TABLE_LABELS = {"table"}
_FORMULA_LABELS = {"formula"}
_HEADING_LABELS = {"paragraph_title", "doc_title", "title"}
_FIGURE_LABELS = {"image", "chart", "figure", "figure_title", "seal"}
# Everything else (text, list, abstract, reference, footnote, header, footer, ...)
# is treated as a plain paragraph.


class OCRUnavailableError(RuntimeError):
    """Raised when the OCR pipeline cannot be created or cannot run (e.g. model
    weights unreachable). Callers must catch this and mark the document
    'failed' with a clear message rather than letting a background task die
    silently."""


@dataclass
class OCRElement:
    element_type: str  # paragraph | heading | table | formula | image | list
    text: str | None
    bbox: dict[str, float] | None
    confidence: float | None
    table_data: dict[str, Any] | None = None
    order: int | None = None


@dataclass
class OCRPageResult:
    page_number: int  # 1-indexed
    width: float | None
    height: float | None
    elements: list[OCRElement] = field(default_factory=list)

    @property
    def raw_text(self) -> str:
        """Reading-order plain text assembled from this page's elements —
        this is what gets stored in DocumentPage.raw_text and fed to the
        Phase 2 chunker, exactly like PyMuPDF's page text is."""
        parts = []
        for el in self.elements:
            if not el.text:
                continue
            if el.element_type == "table":
                parts.append(_html_table_to_text(el.text))
            else:
                parts.append(el.text)
        return "\n\n".join(parts)


class OCRPipeline(Protocol):
    """Structural type for whatever `predict()` returns something iterable
    of page-result objects supporting `res["parsing_res_list"]`,
    `res["width"]`, `res["height"]`. Both the real `PaddleOCRVL` pipeline and
    the fakes used in tests satisfy this without inheritance."""

    def predict(self, input: str) -> Any: ...  # noqa: A002


_pipeline_singleton: OCRPipeline | None = None


def get_ocr_pipeline(pipeline_version: str = "v1.6") -> OCRPipeline:
    """Lazily create (and cache) the real PaddleOCR-VL pipeline.

    Raises OCRUnavailableError if the `paddleocr` package isn't installed or
    the pipeline can't be constructed (most commonly: model weights can't be
    downloaded — see this module's docstring).
    """
    global _pipeline_singleton
    if _pipeline_singleton is not None:
        return _pipeline_singleton

    try:
        from paddleocr import PaddleOCRVL
    except ImportError as exc:
        raise OCRUnavailableError(
            "The 'paddleocr' package is not installed. Run: "
            'pip install paddlepaddle "paddleocr[doc-parser]>=3.6.0"'
        ) from exc

    try:
        _pipeline_singleton = PaddleOCRVL(pipeline_version=pipeline_version)
    except Exception as exc:  # noqa: BLE001 — this is a hard external boundary
        raise OCRUnavailableError(
            f"Could not initialize PaddleOCR-VL ({pipeline_version}): {exc}"
        ) from exc

    return _pipeline_singleton


def run_ocr_on_images(
    image_paths: list[Path],
    pipeline: OCRPipeline,
    start_page_number: int = 1,
) -> list[OCRPageResult]:
    """Run the OCR pipeline over one image per page and return structured
    per-page results. `image_paths` must already be in page order.

    Raises OCRUnavailableError if `pipeline.predict()` itself raises (e.g. a
    corrupt image, or the pipeline losing its model state mid-run).
    """
    results: list[OCRPageResult] = []
    for offset, image_path in enumerate(image_paths):
        page_number = start_page_number + offset
        try:
            page_outputs = list(pipeline.predict(str(image_path)))
        except Exception as exc:  # noqa: BLE001 — external model call
            raise OCRUnavailableError(
                f"OCR failed on page {page_number} ({image_path.name}): {exc}"
            ) from exc

        if not page_outputs:
            results.append(OCRPageResult(page_number=page_number, width=None, height=None))
            continue

        # PaddleOCRVL.predict() yields one result per page; a single input
        # image is one page, so take the first (only) result.
        res = page_outputs[0]
        results.append(_parse_page_result(res, page_number))

    return results


def _parse_page_result(res: Any, page_number: int) -> OCRPageResult:
    width = _first(res.get("width"))
    height = _first(res.get("height"))
    blocks = res.get("parsing_res_list") or []

    elements: list[OCRElement] = []
    for order, block in enumerate(blocks):
        label = getattr(block, "label", "text")
        content = getattr(block, "content", "") or ""
        bbox_raw = getattr(block, "bbox", None)
        bbox = _bbox_to_dict(bbox_raw)

        element_type, table_data = _classify_block(label, content)
        elements.append(
            OCRElement(
                element_type=element_type,
                text=content,
                bbox=bbox,
                confidence=getattr(block, "confidence", None),
                table_data=table_data,
                order=order,
            )
        )

    return OCRPageResult(page_number=page_number, width=width, height=height, elements=elements)


def _classify_block(label: str, content: str) -> tuple[str, dict[str, Any] | None]:
    if label in _TABLE_LABELS:
        return "table", _parse_html_table(content)
    if label in _FORMULA_LABELS:
        return "formula", None
    if label in _HEADING_LABELS:
        return "heading", None
    if label in _FIGURE_LABELS:
        return "image", None
    if label == "list":
        return "list", None
    return "paragraph", None


def _bbox_to_dict(bbox_raw: Any) -> dict[str, float] | None:
    if not bbox_raw or len(bbox_raw) < 4:
        return None
    x0, y0, x1, y1 = bbox_raw[:4]
    return {"x0": float(x0), "y0": float(y0), "x1": float(x1), "y1": float(y1)}


def _first(value: Any) -> Any:
    """PaddleOCR-VL sometimes returns width/height as a bare number and
    sometimes as a one-element list (see result.py's `_page_image_width`
    doing the same unwrap) — normalize both shapes."""
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _parse_html_table(html: str) -> dict[str, Any] | None:
    """Best-effort HTML table -> {"headers": [...], "rows": [[...]]} so
    table_data is queryable without re-parsing HTML downstream. Falls back to
    None (table_data stays empty, raw HTML/text is still kept as the
    element's `text`) if the content isn't parseable HTML — a malformed
    table must never take down the whole ingestion run.
    """
    if not html or "<table" not in html.lower():
        return None
    try:
        rows_raw = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.IGNORECASE | re.DOTALL)
        rows: list[list[str]] = []
        for row_html in rows_raw:
            cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row_html, re.IGNORECASE | re.DOTALL)
            clean_cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
            if clean_cells:
                rows.append(clean_cells)
        if not rows:
            return None
        headers, *body = rows
        return {"headers": headers, "rows": body}
    except Exception:  # noqa: BLE001 — parsing is best-effort, never fatal
        logger.warning("Failed to parse OCR table HTML into structured rows; keeping raw HTML only.")
        return None


def _html_table_to_text(html: str) -> str:
    """Flatten a table's HTML/content into readable text for the page's
    raw_text (and therefore for chunking) even when a structured parse
    wasn't possible."""
    parsed = _parse_html_table(html)
    if parsed is None:
        return re.sub(r"<[^>]+>", " ", html).strip()
    lines = [" | ".join(parsed["headers"])]
    lines += [" | ".join(row) for row in parsed["rows"]]
    return "\n".join(lines)
