"""
test_phase5.py — Phase 5 tests for:
  - Learning level in POST /api/ask
  - POST /api/summarize
  - POST /api/quiz

IBM watsonx.ai is ALWAYS mocked — no real API calls are made.

Run with:  pytest tests/ -v
"""

from __future__ import annotations

import io
import json
import pathlib
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.upload_models import LearningLevel, SummaryLength
from app.rag.vector_store import clear_store
from app.routes.ask import build_prompt, _LEVEL_INSTRUCTIONS
from app.routes.quiz import build_quiz_prompt, parse_quiz_response
from app.routes.summarize import build_summary_prompt
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
_MOCK_SUMMARY = "This document covers the OSI model, its seven layers, and their functions."
_MOCK_QUIZ_JSON = json.dumps([
    {
        "number": 1,
        "question": "How many layers does the OSI model have?",
        "options": [
            {"label": "A", "text": "5"},
            {"label": "B", "text": "7"},
            {"label": "C", "text": "4"},
            {"label": "D", "text": "6"},
        ],
        "correct_answer": "B",
        "explanation": "The OSI model has 7 layers.",
    },
    {
        "number": 2,
        "question": "What does OSI stand for?",
        "options": [
            {"label": "A", "text": "Open Systems Interconnection"},
            {"label": "B", "text": "Open Source Interface"},
            {"label": "C", "text": "Optical Sensor Integration"},
            {"label": "D", "text": "Output Systems Interface"},
        ],
        "correct_answer": "A",
        "explanation": "OSI stands for Open Systems Interconnection.",
    },
    {
        "number": 3,
        "question": "Which layer is responsible for routing?",
        "options": [
            {"label": "A", "text": "Physical"},
            {"label": "B", "text": "Data Link"},
            {"label": "C", "text": "Network"},
            {"label": "D", "text": "Transport"},
        ],
        "correct_answer": "C",
        "explanation": "The Network layer handles routing.",
    },
    {
        "number": 4,
        "question": "What protocol operates at the Transport layer?",
        "options": [
            {"label": "A", "text": "HTTP"},
            {"label": "B", "text": "IP"},
            {"label": "C", "text": "TCP"},
            {"label": "D", "text": "Ethernet"},
        ],
        "correct_answer": "C",
        "explanation": "TCP operates at the Transport layer.",
    },
    {
        "number": 5,
        "question": "Which layer handles physical transmission of bits?",
        "options": [
            {"label": "A", "text": "Physical"},
            {"label": "B", "text": "Data Link"},
            {"label": "C", "text": "Network"},
            {"label": "D", "text": "Application"},
        ],
        "correct_answer": "A",
        "explanation": "The Physical layer transmits raw bits.",
    },
])


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


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Learning Level on /api/ask
# ═══════════════════════════════════════════════════════════════════════════════

class TestLearningLevel:
    """Tests for the learning_level field on POST /api/ask."""

    # 1a. build_prompt inserts the correct level instructions ──────────────────

    def test_build_prompt_beginner_level(self):
        chunk = MagicMock()
        chunk.text = "The OSI model has 7 layers."
        prompt = build_prompt("Explain OSI", [chunk], LearningLevel.beginner)
        assert "Beginner" in prompt
        assert "simple language" in prompt.lower() or "everyday" in prompt.lower()

    def test_build_prompt_intermediate_level(self):
        chunk = MagicMock()
        chunk.text = "The OSI model has 7 layers."
        prompt = build_prompt("Explain OSI", [chunk], LearningLevel.intermediate)
        assert "Intermediate" in prompt

    def test_build_prompt_advanced_level(self):
        chunk = MagicMock()
        chunk.text = "The OSI model has 7 layers."
        prompt = build_prompt("Explain OSI", [chunk], LearningLevel.advanced)
        assert "Advanced" in prompt

    def test_build_prompt_expert_level(self):
        chunk = MagicMock()
        chunk.text = "The OSI model has 7 layers."
        prompt = build_prompt("Explain OSI", [chunk], LearningLevel.expert)
        assert "Expert" in prompt

    def test_level_instructions_all_four_defined(self):
        """All four learning levels must have a non-empty instruction string."""
        for level in LearningLevel:
            assert level in _LEVEL_INSTRUCTIONS
            assert _LEVEL_INSTRUCTIONS[level].strip()

    # 1b. Default level is beginner ───────────────────────────────────────────

    def test_ask_defaults_to_beginner(self, uploaded_doc):
        with patch(
            "app.routes.ask.watsonx_service.generate_text",
            return_value=_MOCK_ANSWER,
        ) as mock_gen:
            resp = client.post("/api/ask", json={"question": "What is OSI?"})

        assert resp.status_code == 200
        # Capture prompt passed to generate_text
        call_args = mock_gen.call_args
        prompt_text = call_args.args[0] if call_args.args else call_args.kwargs.get("prompt", "")
        assert "Beginner" in prompt_text

    # 1c. Explicit level is passed through ────────────────────────────────────

    def test_ask_with_expert_level(self, uploaded_doc):
        with patch(
            "app.routes.ask.watsonx_service.generate_text",
            return_value=_MOCK_ANSWER,
        ) as mock_gen:
            resp = client.post(
                "/api/ask",
                json={"question": "What is OSI?", "learning_level": "expert"},
            )

        assert resp.status_code == 200
        prompt_text = mock_gen.call_args.args[0]
        assert "Expert" in prompt_text

    def test_ask_with_intermediate_level(self, uploaded_doc):
        with patch(
            "app.routes.ask.watsonx_service.generate_text",
            return_value=_MOCK_ANSWER,
        ) as mock_gen:
            resp = client.post(
                "/api/ask",
                json={"question": "What is OSI?", "learning_level": "intermediate"},
            )

        assert resp.status_code == 200
        prompt_text = mock_gen.call_args.args[0]
        assert "Intermediate" in prompt_text

    def test_ask_with_advanced_level(self, uploaded_doc):
        with patch(
            "app.routes.ask.watsonx_service.generate_text",
            return_value=_MOCK_ANSWER,
        ) as mock_gen:
            resp = client.post(
                "/api/ask",
                json={"question": "What is OSI?", "learning_level": "advanced"},
            )

        assert resp.status_code == 200
        prompt_text = mock_gen.call_args.args[0]
        assert "Advanced" in prompt_text

    # 1d. Invalid level returns 422 ──────────────────────────────────────────

    def test_ask_invalid_learning_level_returns_422(self, uploaded_doc):
        resp = client.post(
            "/api/ask",
            json={"question": "What is OSI?", "learning_level": "god_mode"},
        )
        assert resp.status_code == 422

    # 1e. Response shape is unchanged ────────────────────────────────────────

    def test_ask_response_shape_with_level(self, uploaded_doc):
        with patch(
            "app.routes.ask.watsonx_service.generate_text",
            return_value=_MOCK_ANSWER,
        ):
            resp = client.post(
                "/api/ask",
                json={"question": "What is OSI?", "learning_level": "advanced"},
            )

        body = resp.json()
        assert set(body.keys()) >= {"success", "answer", "sources", "message"}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — POST /api/summarize
# ═══════════════════════════════════════════════════════════════════════════════

class TestSummarize:
    """Tests for POST /api/summarize."""

    # 2a. No document → 404 ───────────────────────────────────────────────────

    def test_summarize_no_documents_returns_404(self):
        resp = client.post("/api/summarize", json={})
        assert resp.status_code == 404
        assert "no documents" in resp.json()["detail"].lower()

    def test_summarize_unknown_doc_id_returns_404(self, uploaded_doc):
        resp = client.post("/api/summarize", json={"document_id": "nonexistent"})
        assert resp.status_code == 404

    # 2b. Successful summary — all three lengths ──────────────────────────────

    def test_summarize_short_success(self, uploaded_doc):
        with patch(
            "app.routes.summarize.watsonx_service.generate_text",
            return_value=_MOCK_SUMMARY,
        ):
            resp = client.post("/api/summarize", json={"length": "short"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["summary"] == _MOCK_SUMMARY
        assert body["length"] == "short"
        assert "message" in body

    def test_summarize_medium_success(self, uploaded_doc):
        with patch(
            "app.routes.summarize.watsonx_service.generate_text",
            return_value=_MOCK_SUMMARY,
        ):
            resp = client.post("/api/summarize", json={"length": "medium"})

        assert resp.status_code == 200
        assert resp.json()["length"] == "medium"

    def test_summarize_detailed_success(self, uploaded_doc):
        with patch(
            "app.routes.summarize.watsonx_service.generate_text",
            return_value=_MOCK_SUMMARY,
        ):
            resp = client.post("/api/summarize", json={"length": "detailed"})

        assert resp.status_code == 200
        assert resp.json()["length"] == "detailed"

    def test_summarize_default_length_is_medium(self, uploaded_doc):
        with patch(
            "app.routes.summarize.watsonx_service.generate_text",
            return_value=_MOCK_SUMMARY,
        ):
            resp = client.post("/api/summarize", json={})

        assert resp.status_code == 200
        assert resp.json()["length"] == "medium"

    # 2c. Response shape ──────────────────────────────────────────────────────

    def test_summarize_response_shape(self, uploaded_doc):
        with patch(
            "app.routes.summarize.watsonx_service.generate_text",
            return_value=_MOCK_SUMMARY,
        ):
            resp = client.post("/api/summarize", json={"length": "short"})

        body = resp.json()
        assert set(body.keys()) >= {"success", "summary", "length", "message"}
        assert isinstance(body["success"], bool)
        assert isinstance(body["summary"], str)
        assert isinstance(body["length"], str)

    # 2d. Invalid length → 422 ────────────────────────────────────────────────

    def test_summarize_invalid_length_returns_422(self, uploaded_doc):
        resp = client.post("/api/summarize", json={"length": "paragraph"})
        assert resp.status_code == 422

    # 2e. Prompt construction ─────────────────────────────────────────────────

    def test_build_summary_prompt_short(self):
        chunk = MagicMock()
        chunk.text = "The OSI model is a conceptual framework."
        prompt = build_summary_prompt([chunk], SummaryLength.short)
        assert "SHORT" in prompt
        assert "OSI model" in prompt
        assert "Summary:" in prompt

    def test_build_summary_prompt_medium(self):
        chunk = MagicMock()
        chunk.text = "The OSI model is a conceptual framework."
        prompt = build_summary_prompt([chunk], SummaryLength.medium)
        assert "MEDIUM" in prompt

    def test_build_summary_prompt_detailed(self):
        chunk = MagicMock()
        chunk.text = "The OSI model is a conceptual framework."
        prompt = build_summary_prompt([chunk], SummaryLength.detailed)
        assert "DETAILED" in prompt

    def test_build_summary_prompt_respects_char_limit(self):
        chunk = MagicMock()
        chunk.text = "x" * 6000
        prompt = build_summary_prompt([chunk], SummaryLength.detailed)
        assert len(prompt) < 12_000

    # 2f. IBM API error handling ──────────────────────────────────────────────

    def test_summarize_ibm_config_error_returns_503(self, uploaded_doc):
        with patch(
            "app.routes.summarize.watsonx_service.generate_text",
            side_effect=WatsonxConfigError("WATSONX_API_KEY is not set"),
        ):
            resp = client.post("/api/summarize", json={})
        assert resp.status_code == 503
        assert "WATSONX_API_KEY" not in resp.json()["detail"]

    def test_summarize_network_error_returns_503(self, uploaded_doc):
        with patch(
            "app.routes.summarize.watsonx_service.generate_text",
            side_effect=WatsonxNetworkError("timeout"),
        ):
            resp = client.post("/api/summarize", json={})
        assert resp.status_code == 503

    def test_summarize_empty_response_returns_502(self, uploaded_doc):
        with patch(
            "app.routes.summarize.watsonx_service.generate_text",
            side_effect=WatsonxEmptyResponse("empty"),
        ):
            resp = client.post("/api/summarize", json={})
        assert resp.status_code == 502

    def test_summarize_with_document_id(self, uploaded_doc):
        doc_id = uploaded_doc["doc_id"]
        with patch(
            "app.routes.summarize.watsonx_service.generate_text",
            return_value=_MOCK_SUMMARY,
        ):
            resp = client.post("/api/summarize", json={"document_id": doc_id, "length": "short"})

        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_summarize_no_api_key_in_response(self, uploaded_doc):
        with patch(
            "app.routes.summarize.watsonx_service.generate_text",
            return_value=_MOCK_SUMMARY,
        ):
            resp = client.post("/api/summarize", json={})
        assert "WATSONX_API_KEY" not in resp.text


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — POST /api/quiz
# ═══════════════════════════════════════════════════════════════════════════════

class TestQuiz:
    """Tests for POST /api/quiz."""

    # 3a. No document → 404 ───────────────────────────────────────────────────

    def test_quiz_no_documents_returns_404(self):
        resp = client.post("/api/quiz", json={})
        assert resp.status_code == 404
        assert "no documents" in resp.json()["detail"].lower()

    def test_quiz_unknown_doc_id_returns_404(self, uploaded_doc):
        resp = client.post("/api/quiz", json={"document_id": "nonexistent"})
        assert resp.status_code == 404

    # 3b. Successful quiz generation ─────────────────────────────────────────

    def test_quiz_success_returns_5_questions(self, uploaded_doc):
        with patch(
            "app.routes.quiz.watsonx_service.generate_text",
            return_value=_MOCK_QUIZ_JSON,
        ):
            resp = client.post("/api/quiz", json={})

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert len(body["questions"]) == 5

    def test_quiz_response_shape(self, uploaded_doc):
        with patch(
            "app.routes.quiz.watsonx_service.generate_text",
            return_value=_MOCK_QUIZ_JSON,
        ):
            resp = client.post("/api/quiz", json={})

        body = resp.json()
        assert set(body.keys()) >= {"success", "questions", "message"}
        assert isinstance(body["questions"], list)

    def test_quiz_question_fields(self, uploaded_doc):
        with patch(
            "app.routes.quiz.watsonx_service.generate_text",
            return_value=_MOCK_QUIZ_JSON,
        ):
            resp = client.post("/api/quiz", json={})

        for q in resp.json()["questions"]:
            assert "number" in q
            assert "question" in q
            assert "options" in q
            assert "correct_answer" in q
            assert "explanation" in q

    def test_quiz_options_have_4_choices(self, uploaded_doc):
        with patch(
            "app.routes.quiz.watsonx_service.generate_text",
            return_value=_MOCK_QUIZ_JSON,
        ):
            resp = client.post("/api/quiz", json={})

        for q in resp.json()["questions"]:
            assert len(q["options"]) == 4
            labels = [o["label"] for o in q["options"]]
            assert set(labels) == {"A", "B", "C", "D"}

    def test_quiz_correct_answer_is_valid_label(self, uploaded_doc):
        with patch(
            "app.routes.quiz.watsonx_service.generate_text",
            return_value=_MOCK_QUIZ_JSON,
        ):
            resp = client.post("/api/quiz", json={})

        for q in resp.json()["questions"]:
            assert q["correct_answer"] in {"A", "B", "C", "D"}

    def test_quiz_with_document_id(self, uploaded_doc):
        doc_id = uploaded_doc["doc_id"]
        with patch(
            "app.routes.quiz.watsonx_service.generate_text",
            return_value=_MOCK_QUIZ_JSON,
        ):
            resp = client.post("/api/quiz", json={"document_id": doc_id})

        assert resp.status_code == 200

    # 3c. Parse quiz response helper ─────────────────────────────────────────

    def test_parse_quiz_response_valid_json(self):
        questions = parse_quiz_response(_MOCK_QUIZ_JSON)
        assert len(questions) == 5
        assert questions[0].number == 1
        assert questions[0].correct_answer == "B"

    def test_parse_quiz_response_with_fenced_markdown(self):
        fenced = f"```json\n{_MOCK_QUIZ_JSON}\n```"
        questions = parse_quiz_response(fenced)
        assert len(questions) == 5

    def test_parse_quiz_response_invalid_json_raises(self):
        import pytest as _pytest
        with _pytest.raises(ValueError, match="not valid JSON"):
            parse_quiz_response("this is not json at all")

    def test_parse_quiz_response_non_array_raises(self):
        import pytest as _pytest
        with _pytest.raises(ValueError, match="array"):
            parse_quiz_response('{"number": 1}')

    def test_build_quiz_prompt_contains_context(self):
        chunk = MagicMock()
        chunk.text = "The OSI model has 7 layers."
        prompt = build_quiz_prompt([chunk])
        assert "OSI model" in prompt
        assert "JSON" in prompt
        assert "5" in prompt

    # 3d. IBM API error handling ──────────────────────────────────────────────

    def test_quiz_ibm_config_error_returns_503(self, uploaded_doc):
        with patch(
            "app.routes.quiz.watsonx_service.generate_text",
            side_effect=WatsonxConfigError("key not set"),
        ):
            resp = client.post("/api/quiz", json={})
        assert resp.status_code == 503
        assert "WATSONX_API_KEY" not in resp.json()["detail"]

    def test_quiz_network_error_returns_503(self, uploaded_doc):
        with patch(
            "app.routes.quiz.watsonx_service.generate_text",
            side_effect=WatsonxNetworkError("timeout"),
        ):
            resp = client.post("/api/quiz", json={})
        assert resp.status_code == 503

    def test_quiz_empty_response_returns_502(self, uploaded_doc):
        with patch(
            "app.routes.quiz.watsonx_service.generate_text",
            side_effect=WatsonxEmptyResponse("empty"),
        ):
            resp = client.post("/api/quiz", json={})
        assert resp.status_code == 502

    def test_quiz_bad_json_from_model_returns_502(self, uploaded_doc):
        """If Granite returns garbage JSON, we get a 502."""
        with patch(
            "app.routes.quiz.watsonx_service.generate_text",
            return_value="Sorry, I cannot generate a quiz.",
        ):
            resp = client.post("/api/quiz", json={})
        assert resp.status_code == 502

    def test_quiz_no_api_key_in_response(self, uploaded_doc):
        with patch(
            "app.routes.quiz.watsonx_service.generate_text",
            return_value=_MOCK_QUIZ_JSON,
        ):
            resp = client.post("/api/quiz", json={})
        assert "WATSONX_API_KEY" not in resp.text

    def test_quiz_auth_error_returns_503(self, uploaded_doc):
        with patch(
            "app.routes.quiz.watsonx_service.generate_text",
            side_effect=WatsonxAuthError("bad credentials"),
        ):
            resp = client.post("/api/quiz", json={})
        assert resp.status_code == 503
