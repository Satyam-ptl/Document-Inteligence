"""
Centralized application configuration.

Every configurable piece of the system (models, DB, vector store, LLM
provider, storage paths) is read from environment variables so that no
component in later phases needs to hard-code a choice. Copy `.env.example`
to `.env` and edit values there; never commit `.env`.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- App ---
    APP_ENV: Literal["development", "production", "test"] = "development"
    APP_NAME: str = "Document Intelligence Platform"
    API_V1_PREFIX: str = "/api"
    CORS_ORIGINS: str = "http://localhost:5173"
    API_KEY: str | None = None
    TRUSTED_HOSTS: str = "localhost,127.0.0.1"

    # --- Database ---
    # Phase 1 default: SQLite dev fallback. Phase 11 (Docker) switches this
    # to a PostgreSQL URL via the .env used by docker-compose.
    DATABASE_URL: str = "sqlite:///./data/dev.db"

    # --- Storage ---
    UPLOAD_DIR: str = "./data/uploads"
    PROCESSED_DIR: str = "./data/processed"
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: str = "pdf,png,jpg,jpeg,tiff,bmp,docx,xlsx,csv"

    # --- Vector DB (wired in Phase 4) ---
    # "server" talks to a real Qdrant instance at QDRANT_URL (docker-compose's
    # `qdrant` service in production). "local" opens an embedded, on-disk
    # Qdrant at QDRANT_LOCAL_PATH with no server/network needed at all — this
    # is what dev/tests use in this sandbox, since it's what's actually
    # verifiable here (see PROJECT_STATE.md's Phase 4 verification log).
    QDRANT_MODE: Literal["server", "local"] = "local"
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    QDRANT_LOCAL_PATH: str = "./data/qdrant"
    QDRANT_COLLECTION: str = "document_chunks"

    # --- Embeddings (wired in Phase 4) ---
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DEVICE: str = "cpu"  # "cpu" or "cuda"
    EMBEDDING_DIMENSION: int = 1024  # bge-m3's fixed dense-vector size; see app/embeddings/embedder.py
    EMBEDDING_BATCH_SIZE: int = 12
    EMBEDDING_FALLBACK_ENABLED: bool = True  # use local hashing embeddings when BGE-M3 is unavailable

    # --- Reranker (wired in Phase 5) ---
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANKER_DEVICE: str = "cpu"  # "cpu" or "cuda"
    RERANK_ENABLED: bool = True  # allow disabling the reranker step (falls back to fused order) without a redeploy

    # --- OCR (Phase 3) ---
    OCR_MODEL: str = "PaddleOCR-VL-1.6"
    OCR_PIPELINE_VERSION: str = "v1.6"
    OCR_DEVICE: str = "cpu"  # "cpu" or "gpu"
    OCR_RENDER_DPI: int = 200  # scanned-PDF page -> image render resolution
    OCR_ENABLE_PREPROCESSING: bool = True  # OpenCV denoise + deskew before OCR

    # --- Ingestion / chunking (Phase 2) ---
    # Naive fixed-size sliding-window chunker (see app/chunking/naive_chunker.py).
    # Chunks never cross a page boundary, so every chunk cites exactly one page.
    CHUNK_SIZE_CHARS: int = 1000
    CHUNK_OVERLAP_CHARS: int = 150
    # Below this average extractable-text-chars-per-page, a PDF is treated as
    # "scanned" (image-only) and routed to the Phase 3 OCR pipeline instead
    # of being ingested as near-empty text.
    SCANNED_PDF_MIN_CHARS_PER_PAGE: int = 20

    # --- LLM (wired in Phase 6). Never hard-code keys here. ---
    LLM_PROVIDER: Literal["deepseek", "gemini", "openai", "local"] = "deepseek"
    LLM_API_KEY: str | None = None
    LLM_MODEL: str = "deepseek-chat"
    LLM_MAX_TOKENS: int = 1024
    LLM_TEMPERATURE: float = 0.0  # deterministic by default — this is grounded Q&A, not creative writing
    LLM_TIMEOUT_SECONDS: float = 60.0
    # Base URLs for the raw-REST provider clients in app/generation/llm_client.py.
    # Only the one matching LLM_PROVIDER is actually used at runtime; all three
    # are always defined so switching LLM_PROVIDER never requires new env vars.
    LLM_DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    LLM_OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    LLM_GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta"

    # --- Retrieval tuning (wired in Phase 5) ---
    # How many candidates each of the dense/sparse legs fetches from Qdrant
    # before fusion — deliberately wider than RETRIEVAL_TOP_K_FINAL so fusion
    # and the reranker both have a real pool to work with, not just the
    # already-truncated top few from a single leg.
    RETRIEVAL_TOP_K_CANDIDATES: int = 40
    # How many fused candidates are actually sent to the reranker. Reranking
    # is the expensive step (a cross-encoder forward pass per candidate), so
    # this is deliberately smaller than RETRIEVAL_TOP_K_CANDIDATES.
    RETRIEVAL_RERANK_POOL: int = 20
    # Final number of results returned to the caller (search API / Phase 6
    # generation) after reranking.
    RETRIEVAL_TOP_K_FINAL: int = 8
    # Reciprocal Rank Fusion constant — standard default from the RRF paper
    # (Cormack et al.), not usually worth tuning.
    RRF_K: int = 60

    # --- Conversation memory (wired in Phase 8) ---
    # How many of the most recent turns in a conversation feed into
    # follow-up-query resolution (app/conversations/memory.py). Kept small
    # and fixed rather than "the whole history" so a long conversation's
    # early, possibly unrelated turns don't leak into the rewrite of a
    # much later message.
    CONVERSATION_HISTORY_TURNS: int = 3
    # Whether an elliptical follow-up ("what about last quarter?") is
    # rewritten into a standalone query via one extra small LLM call before
    # retrieval. False falls back to always searching/answering the raw
    # query as typed, with zero extra LLM cost — useful for keeping
    # `/api/chat` cheap when only single-turn Q&A is needed even inside a
    # tracked conversation.
    CONVERSATION_REWRITE_ENABLED: bool = True

    @model_validator(mode="after")
    def validate_production_configuration(self) -> "Settings":
        if self.APP_ENV != "production":
            return self
        if not self.API_KEY or len(self.API_KEY) < 32:
            raise ValueError("API_KEY must be at least 32 characters in production.")
        if not self.DATABASE_URL.startswith(("postgresql://", "postgresql+psycopg2://")):
            raise ValueError("Production requires a PostgreSQL DATABASE_URL.")
        if self.QDRANT_MODE != "server":
            raise ValueError("Production requires QDRANT_MODE=server.")
        if self.EMBEDDING_FALLBACK_ENABLED:
            raise ValueError("Production requires EMBEDDING_FALLBACK_ENABLED=false.")
        if self.LLM_PROVIDER == "local" or not self.LLM_API_KEY:
            raise ValueError("Production requires a configured external LLM provider and API key.")
        parsed_origins = self.cors_origins_list
        if not parsed_origins or any(urlparse(origin).scheme != "https" for origin in parsed_origins):
            raise ValueError("Production CORS_ORIGINS must contain only HTTPS origins.")
        return self

    @property
    def upload_path(self) -> Path:
        p = Path(self.UPLOAD_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def processed_path(self) -> Path:
        p = Path(self.PROCESSED_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def qdrant_local_path(self) -> Path:
        p = Path(self.QDRANT_LOCAL_PATH)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def allowed_extensions_list(self) -> list[str]:
        return [e.strip().lower() for e in self.ALLOWED_EXTENSIONS.split(",") if e.strip()]

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def trusted_hosts_list(self) -> list[str]:
        return [host.strip() for host in self.TRUSTED_HOSTS.split(",") if host.strip()]


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance — import and call this, don't instantiate Settings() directly."""
    return Settings()
