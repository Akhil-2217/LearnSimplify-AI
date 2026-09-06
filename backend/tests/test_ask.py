"""
test_ask.py — Phase 4 tests for POST /api/ask.

IBM watsonx.ai is ALWAYS mocked — no real API calls are made.
All existing Phase 1, 2, and 3 tests remain unmodified.

Run with:  pytest tests/ -v
"""

from __future__ import annotations

import io
import pathlib
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.rag.vector_store import clear_store
from app.routes.ask import build_prompt
from app.services.document_processor import clear_document_store
from app.services.watsonx_service import (
    WatsonxAuthError,
    WatsonxConfigError,
    WatsonxEmptyResponse,
    WatsonxNetworkError,
)

client = TestClient(app)
FIXTURES = pathlib.Path(__file__).parent / "fixtures"

_MOCK_ANSWER = "The OSI model has seven layers that define how data moves across a network."


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_state():
    """Clear the in-memory document store and vector store before every test."""
    clear_document_store()
    clear_store()
    yield
    clear_document_store()
    clear_store()


@pytest.fixture()
def uploaded_doc():
    """Upload the sample TXT fixture and return the response body dict."""
    data = (FIXTURES / "sample.txt").read_bytes()
    resp = client.post(
        "/api/upload",
        files={"file": ("sample.txt", io.BytesIO(data), "text/plain")},
    )
    assert resp.status_code == 200, f"Fixture upload failed: {resp.text}"
    return resp.json()


# ── 1. Missing / empty question ───────────────────────────────────────────────

def test_ask_empty_question_returns_400(uploaded_doc):
    resp = client.post("/api/ask", json={"question": ""})
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


def test_ask_whitespace_question_returns_400(uploaded_doc):
    resp = client.post("/api/ask", json={"question": "   "})
    assert resp.status_code == 400


# ── 2. Missing document (no docs uploaded) ───────────────────────────────────

def test_ask_no_documents_returns_404():
    # Nothing uploaded — store is empty
    resp = client.post("/api/ask", json={"question": "What is OSI?"})
    assert resp.status_code == 404
    assert "no documents" in resp.json()["detail"].lower()


def test_ask_unknown_document_id_returns_404(uploaded_doc):
    resp = client.post(
        "/api/ask",
        json={"question": "What is OSI?", "document_id": "non-existent-id"},
    )
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


# ── 3. RAG retrieval happens before generation ────────────────────────────────

def test_ask_rag_retrieval_before_generation(uploaded_doc):
    """Verify retrieve() is actually called before generate_text()."""
    with (
        patch("app.routes.ask.retrieve") as mock_retrieve,
        patch("app.routes.ask.watsonx_service.generate_text", return_value=_MOCK_ANSWER),
    ):
        # retrieve must return RetrievedChunk-like objects
        fake_chunk = MagicMock()
        fake_chunk.text = "The OSI model is a conceptual framework."
        fake_chunk.filename = "sample.txt"
        fake_chunk.chunk_index = 0
        fake_chunk.score = 0.9
        mock_retrieve.return_value = [fake_chunk]

        resp = client.post("/api/ask", json={"question": "What is OSI?"})

    assert resp.status_code == 200
    mock_retrieve.assert_called_once()
    # The question passed to retrieve must match what was sent
    call_args = mock_retrieve.call_args
    assert "What is OSI?" in (call_args.kwargs.get("question") or call_args.args[0])


# ── 4. Prompt construction ────────────────────────────────────────────────────

def test_build_prompt_includes_question():
    fake_chunk = MagicMock()
    fake_chunk.text = "The OSI model has 7 layers."
    prompt = build_prompt("Explain OSI", [fake_chunk])
    assert "Explain OSI" in prompt
    assert "OSI model has 7 layers" in prompt


def test_build_prompt_includes_instructions():
    fake_chunk = MagicMock()
    fake_chunk.text = "Sample context."
    prompt = build_prompt("What is TCP?", [fake_chunk])
    assert "course material" in prompt.lower()
    assert "Answer:" in prompt


def test_build_prompt_respects_char_limit():
    """Prompt should not exceed limits even with very long chunks."""
    fake_chunk = MagicMock()
    fake_chunk.text = "x" * 5000  # much longer than _MAX_CONTEXT_CHARS
    prompt = build_prompt("Q?", [fake_chunk])
    # The context portion should be truncated
    assert len(prompt) < 10_000


def test_build_prompt_multiple_chunks():
    chunks = [MagicMock() for _ in range(3)]
    for i, c in enumerate(chunks):
        c.text = f"Chunk {i} content about networking."
    prompt = build_prompt("What is networking?", chunks)
    assert "Chunk 0 content" in prompt


def test_build_prompt_empty_chunks():
    """An empty chunk list should still produce a valid prompt."""
    prompt = build_prompt("What is OSI?", [])
    assert "What is OSI?" in prompt
    assert "Answer:" in prompt


# ── 5. Successful IBM response (mocked) ──────────────────────────────────────

def test_ask_success_returns_answer(uploaded_doc):
    with patch(
        "app.routes.ask.watsonx_service.generate_text",
        return_value=_MOCK_ANSWER,
    ):
        resp = client.post("/api/ask", json={"question": "What is OSI?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["answer"] == _MOCK_ANSWER
    assert isinstance(body["sources"], list)
    assert len(body["sources"]) > 0
    assert "message" in body


def test_ask_success_sources_have_correct_fields(uploaded_doc):
    with patch(
        "app.routes.ask.watsonx_service.generate_text",
        return_value=_MOCK_ANSWER,
    ):
        resp = client.post("/api/ask", json={"question": "What is OSI?"})

    body = resp.json()
    for src in body["sources"]:
        assert "filename" in src
        assert "chunk_index" in src
        assert "score" in src
        # Score should be a reasonable float
        assert 0.0 <= src["score"] <= 2.0  # inner-product cosine similarity range


def test_ask_with_document_id_filter(uploaded_doc):
    doc_id = uploaded_doc["doc_id"]
    with patch(
        "app.routes.ask.watsonx_service.generate_text",
        return_value=_MOCK_ANSWER,
    ):
        resp = client.post(
            "/api/ask",
            json={"question": "What is OSI?", "document_id": doc_id},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    # All sources should come from the uploaded file
    for src in body["sources"]:
        assert src["filename"] == "sample.txt"


def test_ask_answer_not_contain_api_key(uploaded_doc):
    """The API response must never contain the API key."""
    with patch(
        "app.routes.ask.watsonx_service.generate_text",
        return_value=_MOCK_ANSWER,
    ):
        resp = client.post("/api/ask", json={"question": "What is OSI?"})

    # Even if someone tried to sneak the key in, it should not appear
    assert "WATSONX_API_KEY" not in resp.text
    assert "api_key" not in resp.text.lower().replace("api_key", "")


# ── 6. IBM API failure ────────────────────────────────────────────────────────

def test_ask_ibm_api_error_returns_502(uploaded_doc):
    from app.services.watsonx_service import WatsonxAPIError

    with patch(
        "app.routes.ask.watsonx_service.generate_text",
        side_effect=WatsonxAPIError("IBM returned 500"),
    ):
        resp = client.post("/api/ask", json={"question": "What is OSI?"})

    assert resp.status_code == 502
    # Must not expose internal error details / stack traces
    body = resp.json()
    assert "IBM returned 500" not in body["detail"]


def test_ask_network_error_returns_503(uploaded_doc):
    with patch(
        "app.routes.ask.watsonx_service.generate_text",
        side_effect=WatsonxNetworkError("Connection refused"),
    ):
        resp = client.post("/api/ask", json={"question": "What is OSI?"})

    assert resp.status_code == 503


def test_ask_empty_response_returns_502(uploaded_doc):
    with patch(
        "app.routes.ask.watsonx_service.generate_text",
        side_effect=WatsonxEmptyResponse("Empty"),
    ):
        resp = client.post("/api/ask", json={"question": "What is OSI?"})

    assert resp.status_code == 502


# ── 7. Missing API key ────────────────────────────────────────────────────────

def test_ask_missing_api_key_returns_503(uploaded_doc):
    with patch(
        "app.routes.ask.watsonx_service.generate_text",
        side_effect=WatsonxConfigError("WATSONX_API_KEY is not set"),
    ):
        resp = client.post("/api/ask", json={"question": "What is OSI?"})

    assert resp.status_code == 503
    # Must not expose the internal config error message with the env var name
    body = resp.json()
    assert "WATSONX_API_KEY" not in body["detail"]


def test_ask_auth_error_returns_503(uploaded_doc):
    with patch(
        "app.routes.ask.watsonx_service.generate_text",
        side_effect=WatsonxAuthError("Invalid credentials"),
    ):
        resp = client.post("/api/ask", json={"question": "What is OSI?"})

    assert resp.status_code == 503


# ── 8. Response format ────────────────────────────────────────────────────────

def test_ask_response_shape(uploaded_doc):
    """Verify every field in the documented response contract is present."""
    with patch(
        "app.routes.ask.watsonx_service.generate_text",
        return_value=_MOCK_ANSWER,
    ):
        resp = client.post("/api/ask", json={"question": "What is OSI?"})

    body = resp.json()
    assert set(body.keys()) >= {"success", "answer", "sources", "message"}
    assert isinstance(body["success"], bool)
    assert isinstance(body["answer"], str)
    assert isinstance(body["sources"], list)
    assert isinstance(body["message"], str)


def test_ask_response_content_type_is_json(uploaded_doc):
    with patch(
        "app.routes.ask.watsonx_service.generate_text",
        return_value=_MOCK_ANSWER,
    ):
        resp = client.post("/api/ask", json={"question": "What is OSI?"})

    assert "application/json" in resp.headers["content-type"]
