from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

settings = get_settings()

# SQLite needs this connect_arg for multi-threaded FastAPI access; Postgres ignores it.
connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}

# SQLite (unlike Postgres) needs its parent directory to already exist, and
# won't create it itself — make sure it's there before the engine connects.
if settings.DATABASE_URL.startswith("sqlite"):
    db_path = settings.DATABASE_URL.removeprefix("sqlite:///")
    if db_path and db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(settings.DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a DB session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create tables that don't exist yet. Fine for SQLite dev; use Alembic migrations in production."""
    from app.database import models  # noqa: F401  (ensures models are registered on Base)

    Base.metadata.create_all(bind=engine)
