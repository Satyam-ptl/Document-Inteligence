"""
Normalized internal document representation.

These four tables are the backbone of the whole pipeline: every ingestion
pipeline (PyMuPDF, PaddleOCR-VL, python-docx, pandas) writes into this same
shape, so retrieval/chat code never needs to know which pipeline produced a
given piece of text. Populated incrementally across phases:
  Phase 2: Document, DocumentPage, Chunk (text-based PDFs)
  Phase 3: DocumentElement (OCR/layout elements), OCR-specific fields
  Phase 4+: Chunk.embedded flag once indexed into Qdrant
  Phase 8: Conversation, ConversationTurn (multi-turn chat memory)
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.session import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(String(512))
    original_filename: Mapped[str] = mapped_column(String(512))
    mime_type: Mapped[str] = mapped_column(String(128))
    file_size: Mapped[int] = mapped_column(Integer)
    upload_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    # uploaded | detecting | processing | ocr | extracting | chunking | embedding | indexing | completed | failed
    processing_status: Mapped[str] = mapped_column(String(32), default="uploaded")
    processing_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # text_pdf | scanned_pdf | image | docx | xlsx | csv
    source_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # pymupdf | paddleocr_vl | python_docx | pandas | mixed
    processing_method: Mapped[str | None] = mapped_column(String(32), nullable=True)

    pages: Mapped[list["DocumentPage"]] = relationship(back_populates="document", cascade="all, delete-orphan")
    elements: Mapped[list["DocumentElement"]] = relationship(back_populates="document", cascade="all, delete-orphan")
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class DocumentPage(Base):
    __tablename__ = "document_pages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(String(36), ForeignKey("documents.id"))
    page_number: Mapped[int] = mapped_column(Integer)
    width: Mapped[float | None] = mapped_column(Float, nullable=True)
    height: Mapped[float | None] = mapped_column(Float, nullable=True)
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    processing_method: Mapped[str | None] = mapped_column(String(32), nullable=True)  # pymupdf | paddleocr_vl

    document: Mapped["Document"] = relationship(back_populates="pages")


class DocumentElement(Base):
    """A structural unit on a page: paragraph, heading, table, formula, image caption, etc."""

    __tablename__ = "document_elements"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(String(36), ForeignKey("documents.id"))
    page_number: Mapped[int] = mapped_column(Integer)
    element_type: Mapped[str] = mapped_column(String(32))  # paragraph | heading | table | formula | image | list
    text: Mapped[str | None] = mapped_column(Text, nullable=True)
    section: Mapped[str | None] = mapped_column(String(256), nullable=True)
    bbox: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # {"x0","y0","x1","y1"}
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    table_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # {"headers": [...], "rows": [[...]]}
    element_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    document: Mapped["Document"] = relationship(back_populates="elements")


class Chunk(Base):
    """Retrieval unit: what actually gets embedded and stored in Qdrant."""

    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(String(36), ForeignKey("documents.id"))
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section: Mapped[str | None] = mapped_column(String(256), nullable=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    element_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    source_filename: Mapped[str] = mapped_column(String(512))
    chunk_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    embedded: Mapped[bool] = mapped_column(default=False)  # set True once indexed in Qdrant (Phase 4)
    qdrant_point_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    document: Mapped["Document"] = relationship(back_populates="chunks")


class Conversation(Base):
    """Phase 8: a multi-turn chat session. Created lazily the first time a
    `/api/chat` caller supplies a `conversation_id` — see
    `app/conversations/memory.py`'s `get_or_create_conversation()`. Chat
    calls that never pass a `conversation_id` never create one of these,
    keeping single-turn use (Phase 6/7's original contract) exactly as
    cheap and stateless as before."""

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    # Optional: mirrors ChatRequest.document_id when every turn in this
    # conversation is scoped to one document. Purely informational — chat.py
    # still passes document_id per-request, this isn't used to re-derive it.
    document_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("documents.id"), nullable=True)

    turns: Mapped[list["ConversationTurn"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", order_by="ConversationTurn.turn_index"
    )


class ConversationTurn(Base):
    """Phase 8: one request/response pair within a Conversation.
    `raw_query` is exactly what the caller sent; `resolved_query` is what
    was actually searched/answered — equal to `raw_query` unless follow-up
    resolution rewrote it (see `app/conversations/memory.py`). Storing both
    lets a later turn's rewrite see what a prior turn actually asked
    *and* what it was understood to mean, and lets a frontend show the
    user what the system resolved their follow-up to, for transparency."""

    __tablename__ = "conversation_turns"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(String(36), ForeignKey("conversations.id"))
    turn_index: Mapped[int] = mapped_column(Integer)  # 0-based order within the conversation
    raw_query: Mapped[str] = mapped_column(Text)
    resolved_query: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    conversation: Mapped["Conversation"] = relationship(back_populates="turns")
