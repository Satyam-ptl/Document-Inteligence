from contextlib import asynccontextmanager
import secrets

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import JSONResponse
from loguru import logger

from app.api import chat, conversations, documents, search
from app.config import get_settings
from app.database.session import init_db
from app.schemas.schemas import HealthOut

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.APP_NAME} [{settings.APP_ENV}]")
    init_db()
    yield
    logger.info("Shutting down")


app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        if settings.APP_ENV == "production":
            response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        return response


class ApiKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        public_paths = {"/api/health", "/api/ready"}
        if settings.API_KEY and request.url.path.startswith(settings.API_V1_PREFIX) and request.url.path not in public_paths:
            supplied = request.headers.get("X-API-Key", "")
            if not secrets.compare_digest(supplied, settings.API_KEY):
                return JSONResponse({"detail": "Invalid or missing API key."}, status_code=401)
        return await call_next(request)


app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(ApiKeyMiddleware)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.trusted_hosts_list + (["testserver"] if settings.APP_ENV == "test" else []),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix=settings.API_V1_PREFIX)
app.include_router(search.router, prefix=settings.API_V1_PREFIX)  # Phase 5
app.include_router(chat.router, prefix=settings.API_V1_PREFIX)  # Phase 6
app.include_router(conversations.router, prefix=settings.API_V1_PREFIX)  # Phase 8


@app.get("/api/health", response_model=HealthOut, tags=["health"])
def health() -> HealthOut:
    return HealthOut(status="ok", app_name=settings.APP_NAME, app_env=settings.APP_ENV)


@app.get("/api/ready", tags=["health"])
def readiness() -> dict[str, str]:
    """Readiness probe that verifies the database and vector store are reachable."""
    from sqlalchemy import text
    from app.database.session import engine
    from app.retrieval.qdrant_store import get_qdrant_client

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        client = get_qdrant_client(settings)
        client.get_collections()
    except Exception as exc:
        logger.warning(f"Readiness check failed: {exc}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service dependencies unavailable") from exc
    return {"status": "ready"}
