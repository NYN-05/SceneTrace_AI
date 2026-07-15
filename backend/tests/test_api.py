import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

def test_health():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "indexed_videos" in data

def test_upload_no_file():
    resp = client.post("/api/videos/upload")
    assert resp.status_code == 422

def test_upload_invalid_extension():
    resp = client.post(
        "/api/videos/upload",
        files={"file": ("test.txt", b"hello", "text/plain")}
    )
    assert resp.status_code == 400

def test_upload_empty_file():
    resp = client.post(
        "/api/videos/upload",
        files={"file": ("empty.mp4", b"", "video/mp4")}
    )
    assert resp.status_code == 400

def test_search_no_indexes():
    resp = client.post("/api/search", json={"query": "test", "top_k": 5})
    assert resp.status_code == 400

def test_v2_search_no_indexes():
    resp = client.post("/api/v2/search", json={"query": "test", "top_k": 5})
    assert resp.status_code == 400

def test_status_nonexistent():
    resp = client.get("/api/videos/nonexistent/status")
    assert resp.status_code == 404

def test_search_invalid_top_k():
    resp = client.post("/api/search", json={"query": "test", "top_k": -1})
    assert resp.status_code == 422

def test_search_empty_query():
    resp = client.post("/api/search", json={"query": "", "top_k": 5})
    assert resp.status_code == 422

def test_search_query_too_long():
    resp = client.post("/api/search", json={"query": "x" * 501, "top_k": 5})
    assert resp.status_code == 422

def test_metrics():
    resp = client.get("/api/metrics")
    assert resp.status_code == 200

def test_dashboard_metrics():
    resp = client.get("/api/dashboard/metrics")
    assert resp.status_code == 200
