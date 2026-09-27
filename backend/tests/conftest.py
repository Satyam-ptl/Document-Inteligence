"""
Phase 10: shared pytest configuration for the whole backend test suite.

KNOWN GAPS item 1 / item 14(a) (see PROJECT_STATE.md): every test file up
through Phase 9 imported `app.main`/`app.config` directly, which meant they
all shared whatever `DATABASE_URL`/`QDRANT_LOCAL_PATH` `.env` (or its
defaults, `./data/dev.db` and `./data/qdrant`) resolved to — one on-disk
SQLite file and one on-disk local-Qdrant directory reused across every test
run. That's harmless for a single developer running `pytest` once, but it
means test runs are NOT isolated from each other (leftover rows/vectors
from a previous run, or a previous file in the same run, can leak into a
later one) and repeated CI runs slowly accumulate state on disk.

Fix: before ANY `app.*` module gets imported (which is what actually
triggers `get_settings()` and, in `app/database/session.py`, engine
creation at import time), point both `DATABASE_URL` and `QDRANT_LOCAL_PATH`
at a fresh temp directory for this test *session*. This is deliberately a
per-session temp path, not a per-test-function one (an in-memory
`sqlite:///:memory:` was considered and rejected: it doesn't work cleanly
across the multiple connections `TestClient`'s threaded FastAPI app opens,
since each connection would get its own empty in-memory DB). A per-session
temp file gives every run a guaranteed-clean starting point while still
letting fixtures/tests within a run share state the way they already
assume (e.g. Phase 1's module-scoped `client` fixture uploading a document
in one test and reading it back in another).

This file MUST NOT import anything from `app.*` at module level — only
stdlib — since the whole point is to set the environment variables before
that first import happens anywhere in the collected test files. pytest
always imports a directory's `conftest.py` before collecting/importing the
test modules inside it, which is what makes this ordering reliable.
"""

import os
import shutil
import tempfile

_TMP_ROOT = tempfile.mkdtemp(prefix="docintel_pytest_")

os.environ.setdefault("APP_ENV", "test")
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_ROOT}/test.db"
os.environ["QDRANT_LOCAL_PATH"] = f"{_TMP_ROOT}/qdrant"
# Keep uploads/processed output out of the real ./data tree too, so a test
# run never leaves files next to whatever a developer has actually uploaded
# via the running dev server.
os.environ["UPLOAD_DIR"] = f"{_TMP_ROOT}/uploads"
os.environ["PROCESSED_DIR"] = f"{_TMP_ROOT}/processed"

import pytest  # noqa: E402  (must follow the os.environ block above)


@pytest.fixture(scope="session", autouse=True)
def _cleanup_tmp_root():
    """Remove the whole per-session temp tree once every test has run."""
    yield
    shutil.rmtree(_TMP_ROOT, ignore_errors=True)
