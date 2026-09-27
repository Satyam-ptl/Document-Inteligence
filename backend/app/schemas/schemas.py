from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    original_filename: str
    mime_type: str
    file_size: int
    upload_time: datetime
    processing_status: str
    processing_error: str | None = None
    page_count: int | None = None
    source_type: str | None = None
    processing_method: str | None = None


class DocumentPageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    page_number: int
    width: float | None = None
    height: float | None = None
    raw_text: str | None = None
    processing_method: str | None = None


class ChunkOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    page_number: int | None = None
    section: str | None = None
    chunk_index: int
    text: str
    element_type: str | None = None
    embedded: bool


class SearchRequest(BaseModel):
    query: str
    document_id: str | None = None  # optionally scope search to one document


class SearchResultOut(BaseModel):
    chunk_id: str
    document_id: str
    score: float
    reranked: bool
    dense_rank: int | None = None
    sparse_rank: int | None = None
    text: str
    page_number: int | None = None
    chunk_index: int
    element_type: str | None = None
    source_filename: str


class SearchResponseOut(BaseModel):
    query: str
    results: list[SearchResultOut]
    reranker_used: bool


class ChatRequest(BaseModel):
    query: str
    document_id: str | None = None  # optionally scope generation to one document
    # Phase 8: opt-in multi-turn memory. Omit/None for the original Phase
    # 6/7 stateless single-turn behavior. Pass an id (your own, or one
    # returned from a previous response's `conversation_id`) to track
    # history and enable follow-up resolution; an unrecognized id starts a
    # new conversation using that id rather than erroring.
    conversation_id: str | None = None


class SourcePassageOut(BaseModel):
    """One context passage offered to the LLM (Phase 6), whether or not the
    answer actually cited it — the frontend's evidence viewer (Phase 9)
    needs the full set to let a user check what was available, not just
    the subset the model chose to reference."""

    index: int  # 1-based passage number, matches the [n] marker convention
    chunk_id: str
    document_id: str
    source_filename: str
    page_number: int | None = None
    text: str
    cited: bool  # whether the answer's citations actually reference this passage number


class ConfidenceOut(BaseModel):
    """Phase 7: heuristic (not calibrated-probability) confidence for a
    generated answer — see app/reliability/confidence.py's module
    docstring for exactly how the score is derived and why it's a proxy,
    not a true probability."""

    level: str  # "high" | "medium" | "low"
    score: float  # 0.0-1.0, higher is more confident
    reasons: list[str]  # short, human-readable explanations for the level/score


class ConflictPairOut(BaseModel):
    """Phase 7: one flagged pair of passages that appear to state
    different numeric values for what looks like the same underlying
    fact — see app/reliability/conflict.py."""

    passage_index_a: int
    passage_index_b: int
    text_a: str
    text_b: str
    sentence_a: str
    sentence_b: str
    shared_context: list[str]


class ConflictsOut(BaseModel):
    has_conflict: bool
    conflicts: list[ConflictPairOut]


class ChatResponseOut(BaseModel):
    query: str
    answer: str
    insufficient_evidence: bool  # True if the LLM reported the retrieved passages don't answer the question
    fully_cited: bool  # True iff every sentence has >=1 valid citation and none are hallucinated
    invalid_citation_indices: list[int]  # passage numbers the LLM cited that don't correspond to a real passage
    sources: list[SourcePassageOut]
    confidence: ConfidenceOut  # Phase 7
    conflicts: ConflictsOut  # Phase 7
    # Phase 8: only set when the request carried/started a conversation_id.
    conversation_id: str | None = None
    # Phase 8: only set when follow-up resolution actually rewrote the raw
    # query into a different standalone query (None on a first turn, when
    # rewriting is disabled, or when the raw query was already standalone).
    resolved_query: str | None = None


class ConversationTurnOut(BaseModel):
    """Phase 8: one stored turn, as returned by GET /api/conversations/{id}."""

    model_config = ConfigDict(from_attributes=True)

    turn_index: int
    raw_query: str
    resolved_query: str
    answer: str
    created_at: datetime


class ConversationOut(BaseModel):
    """Phase 8."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    created_at: datetime
    document_id: str | None = None
    turns: list[ConversationTurnOut]


class HealthOut(BaseModel):
    status: str
    app_name: str
    app_env: str


class ErrorOut(BaseModel):
    detail: str
