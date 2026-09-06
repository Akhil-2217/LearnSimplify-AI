"""
Phase 1 tests — health endpoint.
Run with:  pytest tests/test_health.py -v
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_200():
    response = client.get("/api/health")
    assert response.status_code == 200


def test_health_response_body():
    response = client.get("/api/health")
    body = response.json()
    assert body["status"] == "ok"
    assert "LearnSimplify" in body["message"]
    assert "version" in body


def test_health_content_type():
    response = client.get("/api/health")
    assert "application/json" in response.headers["content-type"]
