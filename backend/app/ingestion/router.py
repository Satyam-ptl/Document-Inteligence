"""
Ingestion router: given a Document row already persisted with status
"uploaded", figures out what kind of file it is and dispatches to the
right extraction pipeline, then chunks the result and writes
DocumentPage / Chunk rows.

Phase 2 wired the "text_pdf" path (PyMuPDF). Phase 3 wires the "scanned_pdf"
and "image" paths through PaddleOCR-VL: scanned PDFs are rasterized to
per-page images and images are used as-is, both go through OpenCV
preprocessing then the OCR pipeline, and the resulting layout elements
(paragraphs, headings, tables, formulas, figures) are written to
DocumentElement as well as DocumentPage/Chunk so retrieval has structured
data to work with, not just flattened text.
CSV and XLSX files are extracted as page-scoped tables, with one page per
CSV file or XLSX worksheet.

Runs as a FastAPI BackgroundTask (see app/api/documents.py), so it opens
its own DB session rather than reusing the request's — the request's
session is closed before this function ever runs.
"""

from pathlib import Path

from loguru import logger
from sqlalchemy.orm import Session

from app.chunking.naive_chunker import chunk_pages
from app.config import Settings, get_settings
from app.database.models import Chunk, Document, DocumentElement, DocumentPage
from app.database.session import SessionLocal
from app.embeddings.embedder import EmbeddingUnavailableError, embed_texts, get_embedding_model
from app.ingestion.docx_extractor import extract_docx
from app.ingestion.pdf_extractor import extract_pdf
from app.ingestion.table_extractor import extract_table_file
from app.ocr.paddle_ocr import OCRPageResult, OCRUnavailableError, get_ocr_pipeline, run_ocr_on_images
from app.ocr.preprocessing import preprocess_image
from app.ocr.rasterizer import rasterize_pdf_pages
from app.retrieval.qdrant_store import ChunkVectorPayload, ensure_collection, get_qdrant_client, upsert_chunks

_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "tiff", "bmp"}
_TABLE_EXTENSIONS = {"xlsx", "csv"}


def process_document(document_id: str) -> None:
    """Entry point invoked as a background task right after upload."""
    settings = get_settings()
    db: Session = SessionLocal()
    try:
        doc = db.get(Document, document_id)
        if doc is None:
            logger.warning(f"process_document: document {document_id} no longer exists, skipping.")
            return

        file_path = settings.upload_path / doc.filename
        ext = Path(doc.filename).suffix.lower().lstrip(".")

        doc.processing_status = "detecting"
        db.commit()

        if ext == "pdf":
            _process_pdf(doc, db, file_path, settings)
        elif ext in _IMAGE_EXTENSIONS:
            doc.source_type = "image"
            doc.page_count = 1
            doc.processing_status = "ocr"
            db.commit()
            _process_via_ocr(doc, db, image_paths=[file_path], settings=settings)
        elif ext in _TABLE_EXTENSIONS:
            _process_table_file(doc, db, file_path, ext, settings)
        elif ext == "docx":
            _process_docx(doc, db, file_path, settings)
        else:
            doc.processing_status = "failed"
            doc.processing_error = f"Unsupported file extension '.{ext}'."
            db.commit()

    except Exception as exc:  # noqa: BLE001 — background task boundary; must never raise
        logger.exception(f"process_document failed for document {document_id}")
        doc = db.get(Document, document_id)
        if doc is not None:
            doc.processing_status = "failed"
            doc.processing_error = str(exc)[:2000]
            db.commit()
    finally:
        db.close()


def _process_pdf(doc: Document, db: Session, file_path: Path, settings: Settings) -> None:
    doc.processing_status = "processing"
    db.commit()

    result = extract_pdf(file_path, min_text_chars_per_page=settings.SCANNED_PDF_MIN_CHARS_PER_PAGE)
    doc.page_count = len(result.pages)

    if result.is_scanned:
        doc.source_type = "scanned_pdf"
        doc.processing_status = "ocr"
        db.commit()

        render_dir = settings.processed_path / doc.id / "pages"
        image_paths = rasterize_pdf_pages(file_path, render_dir, dpi=settings.OCR_RENDER_DPI)
        _process_via_ocr(doc, db, image_paths=image_paths, settings=settings)
        return

    doc.source_type = "text_pdf"
    doc.processing_method = "pymupdf"
    doc.processing_status = "extracting"
    db.commit()

    for page in result.pages:
        db.add(
            DocumentPage(
                document_id=doc.id,
                page_number=page.page_number,
                width=page.width,
                height=page.height,
                raw_text=page.text,
                processing_method="pymupdf",
            )
        )
    db.commit()

    doc.processing_status = "chunking"
    db.commit()

    chunks = chunk_pages(
        result.pages,
        chunk_size=settings.CHUNK_SIZE_CHARS,
        chunk_overlap=settings.CHUNK_OVERLAP_CHARS,
    )
    for c in chunks:
        db.add(
            Chunk(
                document_id=doc.id,
                page_number=c.page_number,
                chunk_index=c.chunk_index,
                text=c.text,
                element_type=c.element_type,
                source_filename=doc.original_filename,
            )
        )
    db.commit()
    logger.info(f"Document {doc.id}: extracted {len(result.pages)} pages, {len(chunks)} chunks (pymupdf).")

    _embed_and_index_chunks(doc, db, settings)


def _process_table_file(doc: Document, db: Session, file_path: Path, extension: str, settings: Settings) -> None:
    """Extract CSV/XLSX rows into searchable, page-scoped table chunks."""
    doc.source_type = extension
    doc.processing_method = "pandas"
    doc.processing_status = "extracting"
    db.commit()

    pages = extract_table_file(file_path, extension)
    doc.page_count = len(pages)
    for page in pages:
        db.add(
            DocumentPage(
                document_id=doc.id,
                page_number=page.page_number,
                raw_text=page.text,
                processing_method="pandas",
            )
        )
        db.add(
            DocumentElement(
                document_id=doc.id,
                page_number=page.page_number,
                element_type="table",
                text=page.text,
                table_data=None,
                element_metadata={"sheet": page.title} if extension == "xlsx" else None,
            )
        )
    db.commit()

    doc.processing_status = "chunking"
    db.commit()
    for page in pages:
        db.add(
            Chunk(
                document_id=doc.id,
                page_number=page.page_number,
                chunk_index=page.page_number - 1,
                text=page.text,
                element_type="table",
                source_filename=doc.original_filename,
            )
        )
    db.commit()
    logger.info(f"Document {doc.id}: extracted {len(pages)} table page(s) ({extension}).")
    _embed_and_index_chunks(doc, db, settings)


def _process_docx(doc: Document, db: Session, file_path: Path, settings: Settings) -> None:
    """Extract DOCX paragraphs and tables into a searchable document page."""
    doc.source_type = "docx"
    doc.processing_method = "python_docx"
    doc.processing_status = "extracting"
    db.commit()

    pages = extract_docx(file_path)
    doc.page_count = len(pages)
    for page in pages:
        db.add(
            DocumentPage(
                document_id=doc.id,
                page_number=page.page_number,
                raw_text=page.text,
                processing_method="python_docx",
            )
        )
    db.commit()

    doc.processing_status = "chunking"
    db.commit()
    chunks = chunk_pages(
        pages,
        chunk_size=settings.CHUNK_SIZE_CHARS,
        chunk_overlap=settings.CHUNK_OVERLAP_CHARS,
    )
    for c in chunks:
        db.add(
            Chunk(
                document_id=doc.id,
                page_number=c.page_number,
                chunk_index=c.chunk_index,
                text=c.text,
                element_type="text",
                source_filename=doc.original_filename,
            )
        )
    db.commit()
    logger.info(f"Document {doc.id}: extracted {len(pages)} DOCX page(s), {len(chunks)} chunks.")
    _embed_and_index_chunks(doc, db, settings)


def _process_via_ocr(doc: Document, db: Session, image_paths: list[Path], settings: Settings) -> None:
    """Phase 3: run OpenCV preprocessing + PaddleOCR-VL over already-rendered
    page images (one per page — either rasterized scanned-PDF pages, or a
    single standalone image upload), then write DocumentPage,
    DocumentElement, and Chunk rows.

    `doc.processing_status` is expected to already be "ocr" when this is
    called; on success it becomes "extracting" -> "chunking" -> "completed",
    same status vocabulary as the PyMuPDF path so the frontend stepper
    doesn't need to know which pipeline produced a given document.
    """
    try:
        if settings.OCR_ENABLE_PREPROCESSING:
            for p in image_paths:
                preprocess_image(p)

        pipeline = get_ocr_pipeline(pipeline_version=settings.OCR_PIPELINE_VERSION)
        page_results = run_ocr_on_images(image_paths, pipeline=pipeline)
    except OCRUnavailableError as exc:
        doc.processing_status = "failed"
        doc.processing_error = f"OCR unavailable: {exc}"
        db.commit()
        logger.error(f"Document {doc.id}: OCR unavailable — {exc}")
        return

    doc.processing_method = "paddleocr_vl"
    doc.processing_status = "extracting"
    db.commit()

    for page in page_results:
        db.add(
            DocumentPage(
                document_id=doc.id,
                page_number=page.page_number,
                width=page.width,
                height=page.height,
                raw_text=page.raw_text,
                processing_method="paddleocr_vl",
            )
        )
        for el in page.elements:
            db.add(
                DocumentElement(
                    document_id=doc.id,
                    page_number=page.page_number,
                    element_type=el.element_type,
                    text=el.text,
                    bbox=el.bbox,
                    confidence=el.confidence,
                    table_data=el.table_data,
                    element_metadata={"order": el.order} if el.order is not None else None,
                )
            )
    db.commit()

    doc.processing_status = "chunking"
    db.commit()

    chunks = _chunk_ocr_pages(
        page_results,
        chunk_size=settings.CHUNK_SIZE_CHARS,
        chunk_overlap=settings.CHUNK_OVERLAP_CHARS,
    )
    for c in chunks:
        db.add(
            Chunk(
                document_id=doc.id,
                page_number=c.page_number,
                chunk_index=c.chunk_index,
                text=c.text,
                element_type=c.element_type,
                source_filename=doc.original_filename,
            )
        )
    db.commit()
    logger.info(f"Document {doc.id}: extracted {len(page_results)} pages, {len(chunks)} chunks (paddleocr_vl).")

    _embed_and_index_chunks(doc, db, settings)


def _chunk_ocr_pages(pages: list[OCRPageResult], chunk_size: int, chunk_overlap: int):
    """Chunk OCR output with one twist versus the plain-text chunker
    (`app/chunking/naive_chunker.py`): a `table` element is always kept as
    its own whole chunk (`element_type="table"`), never merged with
    surrounding paragraphs or split by the sliding window — splitting a
    table mid-row would silently corrupt it for retrieval, and merging it
    with prose would dilute the one property that makes tables
    reliably useful for the demo's "highest revenue year" / percentage
    questions: a chunk that is *only* that table.
    Everything else on the page (paragraphs, headings, formulas, list items)
    is concatenated in reading order and run through the same fixed-size
    sliding-window chunker Phase 2 already uses for text PDFs, so downstream
    retrieval code doesn't need two different chunk shapes.
    """
    from app.chunking.naive_chunker import ChunkResult, split_text_sliding_window

    out: list[ChunkResult] = []
    global_index = 0
    for page in pages:
        prose_parts = [el.text for el in page.elements if el.element_type != "table" and el.text]
        prose_text = "\n\n".join(prose_parts)
        for piece in split_text_sliding_window(prose_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
            out.append(
                ChunkResult(
                    page_number=page.page_number, chunk_index=global_index, text=piece, element_type="paragraph"
                )
            )
            global_index += 1

        for el in page.elements:
            if el.element_type != "table" or not el.text or not el.text.strip():
                continue
            out.append(
                ChunkResult(
                    page_number=page.page_number,
                    chunk_index=global_index,
                    text=el.text,
                    element_type="table",
                )
            )
            global_index += 1

    return out


def _embed_and_index_chunks(doc: Document, db: Session, settings: Settings) -> None:
    """Phase 4: embed every not-yet-embedded chunk belonging to this
    document and index it into Qdrant. Shared by both the PyMuPDF
    (`_process_pdf`) and OCR (`_process_via_ocr`) paths, since chunking
    output already converges on the same `Chunk` table shape by the time
    this is called — retrieval/generation in later phases never needs to
    know which pipeline produced a given chunk.

    On success, advances the document to "completed" (same terminal status
    both pipelines already used pre-Phase-4). On an embedding-boundary
    failure (model/package unavailable — see
    `app/embeddings/embedder.py`'s `EmbeddingUnavailableError`), marks the
    document "failed" with a clear message, exactly like
    `_process_via_ocr`'s `OCRUnavailableError` handling one step earlier in
    the same pipeline.
    """
    chunk_rows = (
        db.query(Chunk)
        .filter(Chunk.document_id == doc.id, Chunk.embedded.is_(False))
        .order_by(Chunk.chunk_index)
        .all()
    )
    if not chunk_rows:
        # No text extracted at all (e.g. a genuinely empty page) — nothing
        # to embed, but the document still finished processing correctly.
        doc.processing_status = "completed"
        db.commit()
        return

    doc.processing_status = "embedding"
    db.commit()

    try:
        model = get_embedding_model(
            model_name=settings.EMBEDDING_MODEL,
            device=settings.EMBEDDING_DEVICE,
            fallback_enabled=settings.EMBEDDING_FALLBACK_ENABLED,
        )
        # return_sparse defaults to True as of Phase 5 (see embedder.py) so
        # every indexed chunk gets both legs hybrid retrieval needs.
        result = embed_texts([c.text for c in chunk_rows], model=model, batch_size=settings.EMBEDDING_BATCH_SIZE)
    except EmbeddingUnavailableError as exc:
        doc.processing_status = "failed"
        doc.processing_error = f"Embedding unavailable: {exc}"
        db.commit()
        logger.error(f"Document {doc.id}: embedding unavailable — {exc}")
        return

    doc.processing_status = "indexing"
    db.commit()

    client = get_qdrant_client(settings)
    ensure_collection(client, settings.QDRANT_COLLECTION, dimension=settings.EMBEDDING_DIMENSION)

    payloads = [
        ChunkVectorPayload(
            chunk_id=c.id,
            document_id=c.document_id,
            text=c.text,
            page_number=c.page_number,
            chunk_index=c.chunk_index,
            element_type=c.element_type,
            source_filename=c.source_filename,
        )
        for c in chunk_rows
    ]
    # Defensive: only pass sparse vectors through if embed_texts() actually
    # returned one per chunk. A real bge-m3 call always keeps dense/sparse
    # in lockstep, but if it ever didn't (see embedder.py's length-mismatch
    # warning), upsert_chunks() would hard-fail the whole batch on a
    # ValueError — better to degrade to dense-only indexing for this batch
    # than fail an otherwise-successful embedding step over the sparse leg.
    sparse_for_upsert = result.sparse if result.sparse is not None and len(result.sparse) == len(chunk_rows) else None
    point_ids = upsert_chunks(client, settings.QDRANT_COLLECTION, payloads, result.dense, sparse_for_upsert)

    for c, point_id in zip(chunk_rows, point_ids, strict=True):
        c.embedded = True
        c.qdrant_point_id = point_id
    db.commit()

    doc.processing_status = "completed"
    db.commit()
    logger.info(f"Document {doc.id}: embedded and indexed {len(chunk_rows)} chunks into Qdrant.")
