import io

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    # Using the app as a context manager triggers the lifespan handler
    # (init_db()), exactly like a real server startup would.
    with TestClient(app) as c:
        yield c


def test_health(client) -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_upload_rejects_unsupported_type(client) -> None:
    resp = client.post(
        "/api/documents",
        files={"file": ("malware.exe", io.BytesIO(b"not really an exe"), "application/octet-stream")},
    )
    assert resp.status_code == 400


def test_upload_rejects_empty_file(client) -> None:
    resp = client.post(
        "/api/documents",
        files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")},
    )
    assert resp.status_code == 400


def test_upload_list_get_delete_roundtrip(client) -> None:
    resp = client.post(
        "/api/documents",
        files={"file": ("sample.pdf", io.BytesIO(b"%PDF-1.4 fake content"), "application/pdf")},
    )
    assert resp.status_code == 201
    doc = resp.json()
    assert doc["processing_status"] == "uploaded"
    assert doc["original_filename"] == "sample.pdf"

    list_resp = client.get("/api/documents")
    assert list_resp.status_code == 200
    assert any(d["id"] == doc["id"] for d in list_resp.json())

    get_resp = client.get(f"/api/documents/{doc['id']}")
    assert get_resp.status_code == 200

    del_resp = client.delete(f"/api/documents/{doc['id']}")
    assert del_resp.status_code == 204

    missing_resp = client.get(f"/api/documents/{doc['id']}")
    assert missing_resp.status_code == 404


def test_path_traversal_filename_is_sanitized(client) -> None:
    resp = client.post(
        "/api/documents",
        files={"file": ("../../etc/passwd.pdf", io.BytesIO(b"%PDF-1.4 x"), "application/pdf")},
    )
    assert resp.status_code == 201
    assert "/" not in resp.json()["original_filename"]
    assert ".." not in resp.json()["original_filename"]
