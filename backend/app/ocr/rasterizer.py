"""
Renders PDF pages to PNG image files so they can be fed into PaddleOCR-VL
(which takes an image path, not a PDF). Also the single entry point standalone
image uploads pass through on their way into the OCR pipeline, so both paths
(scanned PDF, image upload) share the same preprocessing step.
"""

from pathlib import Path

import pymupdf

# 200 DPI is a reasonable default: high enough for small print / table text,
# without producing enormous images that blow up OCR latency & memory.
DEFAULT_RENDER_DPI = 200


def rasterize_pdf_pages(pdf_path: Path, output_dir: Path, dpi: int = DEFAULT_RENDER_DPI) -> list[Path]:
    """Render every page of `pdf_path` to a PNG in `output_dir`, in page
    order. Returns the list of written image paths (1 per page)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    zoom = dpi / 72.0  # PDF points are 72/inch; pymupdf's matrix is a scale factor from that.
    matrix = pymupdf.Matrix(zoom, zoom)

    image_paths: list[Path] = []
    doc = pymupdf.open(pdf_path)
    try:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=matrix)
            out_path = output_dir / f"page_{i + 1:04d}.png"
            pix.save(out_path)
            image_paths.append(out_path)
    finally:
        doc.close()

    return image_paths
