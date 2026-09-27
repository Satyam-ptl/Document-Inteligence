"""
Document upload & management endpoints.

Phase 1 scope: accept a file, validate it, store it safely on disk, and
create a DB row with status "uploaded".

Phase 2: after the row is committed, a BackgroundTask hands the document
off to app.ingestion.router.process_document, which extracts text
(PyMuPDF, for normal PDFs) and naive-chunks it. Two read-only debug
endpoints (`/pages`, `/chunks`) were added so ingestion output can be
inspected directly — the frontend evidence viewer in a later phase will
likely reuse `/chunks`.
"""

import re
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, status
from loguru import logger
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.database.models import Chunk, Document, DocumentPage
from app.database.session import get_db
from app.ingestion.router import process_document
from app.retrieval.qdrant_store import delete_document_vectors, get_qdrant_client
from app.schemas.schemas import ChunkOut, DocumentOut, DocumentPageOut

router = APIRouter(prefix="/documents", tags=["documents"])

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_filename(name: str) -> str:
    """Strip path components and unsafe characters to prevent path traversal / injection."""
    name = Path(name).name  # drops any directory components (../, /etc/passwd, etc.)
    name = _SAFE_NAME_RE.sub("_", name)
    return name or "unnamed_file"


@router.post("", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> Document:
    original_name = file.filename or "unnamed_file"
    ext = Path(original_name).suffix.lower().lstrip(".")

    if ext not in settings.allowed_extensions_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '.{ext}'. Supported formats: "
            f"{', '.join(settings.allowed_extensions_list)}.",
        )

    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    contents = bytearray()
    while chunk := await file.read(1024 * 1024):
        contents.extend(chunk)
        if len(contents) > max_bytes:
            size_mb = len(contents) / (1024 * 1024)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File too large ({size_mb:.1f} MB). Maximum allowed is {settings.MAX_FILE_SIZE_MB} MB.",
            )
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large ({size_mb:.1f} MB). Maximum allowed is {settings.MAX_FILE_SIZE_MB} MB.",
        )
    if len(contents) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    safe_original = _sanitize_filename(original_name)
    stored_filename = f"{uuid.uuid4()}.{ext}"
    dest_path = settings.upload_path / stored_filename

    # Defence in depth: resolved path must stay inside the upload directory.
    if settings.upload_path.resolve() not in dest_path.resolve().parents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file path.")

    dest_path.write_bytes(contents)
    logger.info(f"Stored upload '{safe_original}' as '{stored_filename}' ({size_mb:.2f} MB)")

    doc = Document(
        filename=stored_filename,
        original_filename=safe_original,
        mime_type=file.content_type or "application/octet-stream",
        file_size=len(contents),
        processing_status="uploaded",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    background_tasks.add_task(process_document, doc.id)
    return doc


@router.get("", response_model=list[DocumentOut])
def list_documents(db: Session = Depends(get_db)) -> list[Document]:
    return db.query(Document).order_by(Document.upload_time.desc()).all()


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: str, db: Session = Depends(get_db)) -> Document:
    doc = db.get(Document, document_id)
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return doc


@router.get("/{document_id}/pages", response_model=list[DocumentPageOut])
def list_document_pages(document_id: str, db: Session = Depends(get_db)) -> list[DocumentPage]:
    """Debug/inspection endpoint: raw per-page extraction output (Phase 2+)."""
    doc = db.get(Document, document_id)
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return db.query(DocumentPage).filter(DocumentPage.document_id == document_id).order_by(DocumentPage.page_number).all()


@router.get("/{document_id}/chunks", response_model=list[ChunkOut])
def list_document_chunks(document_id: str, db: Session = Depends(get_db)) -> list[Chunk]:
    """Debug/inspection endpoint: chunked retrieval units (Phase 2+). Reused by the evidence viewer later."""
    doc = db.get(Document, document_id)
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return db.query(Chunk).filter(Chunk.document_id == document_id).order_by(Chunk.chunk_index).all()


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: str,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> None:
    doc = db.get(Document, document_id)
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    file_path = settings.upload_path / doc.filename
    if file_path.exists():
        file_path.unlink()

    # Phase 4: a document may have chunks indexed in Qdrant — remove those
    # too so deleting a document doesn't leave stale, orphaned vectors that
    # retrieval could still surface.
    try:
        client = get_qdrant_client(settings)
        delete_document_vectors(client, settings.QDRANT_COLLECTION, document_id)
    except Exception:  # noqa: BLE001 — best-effort cleanup, must not block the delete
        logger.warning(f"Could not clean up Qdrant vectors for deleted document {document_id}.")

    db.delete(doc)
    db.commit()
