"""
Naive, dependency-free text chunker.

Phase 2 scope: a straightforward fixed-size sliding-window chunker applied
to each page's text independently — chunks never cross a page boundary, so
every chunk can always cite a single, unambiguous page number for source
attribution later (Phase 6). This is intentionally simple; later phases may
swap in a smarter (semantic / sentence-aware) chunker without changing the
`Chunk` table shape.
"""

from dataclasses import dataclass

from app.ingestion.pdf_extractor import PageExtraction


@dataclass
class ChunkResult:
    page_number: int
    chunk_index: int
    text: str
    element_type: str = "paragraph"


def split_text_sliding_window(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """The actual fixed-size sliding-window splitter, factored out of
    `chunk_pages` so Phase 3's OCR chunking (`app/ingestion/router.py`,
    which needs to chunk assembled page text the same way but keep table
    elements as separate whole chunks) can reuse it instead of duplicating
    the windowing logic. Returns already-stripped, non-empty pieces (an
    empty/whitespace-only input returns an empty list).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must not be negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    text = text.strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    pieces: list[str] = []
    step = chunk_size - chunk_overlap
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end == len(text):
            break
        start += step

    return pieces


def chunk_pages(pages: list[PageExtraction], chunk_size: int, chunk_overlap: int) -> list[ChunkResult]:
    """
    Split each page's text into overlapping fixed-size chunks.

    `chunk_index` is a single monotonically increasing counter across the
    whole document (not reset per page), so chunks retain a stable overall
    order regardless of page number.
    """
    chunks: list[ChunkResult] = []
    global_index = 0

    for page in pages:
        for piece in split_text_sliding_window(page.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
            chunks.append(ChunkResult(page_number=page.page_number, chunk_index=global_index, text=piece))
            global_index += 1

    return chunks
