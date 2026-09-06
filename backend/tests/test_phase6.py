"""
test_phase6.py — Phase 6 tests for conversation history and follow-up Q&A.

Covers:
    - Conversation creation (first question via POST /api/conversation/ask)
    - Follow-up questions (context is passed to the prompt)
    - Conversation history retrieval (GET /api/conversation/{doc_id})
    - Clearing conversation history (DELETE /api/conversation/{doc_id})
    - Context handling (history injected into follow-up prompt)
    - IBM watsonx.ai is ALWAYS mocked — no real API calls are made.
    - All existing Phase 1-5 tests remain unmodified.

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
from app.routes.conversation import build_followup_prompt
from app.services.conversation_store import clear_all as clear_conversation_all
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
_MOCK_FOLLOWUP = "The physical layer is layer 1 and handles raw bit transmission."


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_state():
    """Clear all in-memory state before every test."""
    clear_document_store()
    clear_store()
    clear_conversation_all()
    yield
    clear_document_store()
    clear_store()
    clear_conversation_all()


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


# ── 1. Conversation creation (first question) ─────────────────────────────────

class TestConversationCreation:
    def test_first_question_returns_200(self, uploaded_doc):
        with patch(
            "app.routes.conversation.watsonx_service.generate_text",
            return_value=_MOCK_ANSWER,
        ):
            resp = client.post(
                "/api/conversation/ask",
                json={"question": "What is OSI?"},
            )
        assert resp.status_code == 200

    def test_first_question_response_shape(self, uploaded_doc):
        with patch(
            "app.routes.conversation.watsonx_service.generate_text",
            return_value=_MOCK_ANSWER,
        ):
            resp = client.post(
                "/api/conversation/ask",
                json={"question": "What is OSI?"},
            )
        body = resp.json()
        assert set(body.keys()) >= {"success", "answer", "sources", "message"}
        assert body["success"] is True
        assert body["answer"] == _MOCK_ANSWER
        assert isinstance(body["sources"], list)

    def test_first_question_creates_history_entry(self, uploaded_doc):
        doc_id = uploaded_doc["doc_id"]
        with patch(
            "app.routes.conversation.watsonx_service.generate_text",
            return_value=_MOCK_ANSWER,
        ):
            client.post(
                "/api/conversation/ask",
                json={"question": "What is OSI?", "doc_id": doc_id},
            )

        # Verify the turn was stored
        hist_resp = client.get(f"/api/conversation/{doc_id}")
        assert hist_resp.status_code == 200
        body = hist_resp.json()
        assert body["success"] is True
        assert len(body["turns"]) == 1
        assert body["turns"][0]["question"] == "What is OSI?"
        assert body["turns"][0]["answer"] == _MOCK_ANSWER

    def test_empty_question_returns_400(self, uploaded_doc):
        resp = client.post(
            "/api/conversation/ask",
            json={"question": ""},
        )
        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"].lower()

    def test_whitespace_question_returns_400(self, uploaded_doc):
        resp = client.post(
            "/api/conversation/ask",
            json={"question": "   "},
        )
        assert resp.status_code == 400

    def test_no_documents_returns_404(self):
        resp = client.post(
            "/api/conversation/ask",
            json={"question": "What is OSI?"},
        )
        assert resp.status_code == 404
        assert "no documents" in resp.json()["detail"].lower()

    def test_unknown_doc_id_returns_404(self, uploaded_doc):
        resp = client.post(
            "/api/conversation/ask",
            json={"question": "What is OSI?", "doc_id": "non-existent-id"},
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


# ── 2. Follow-up questions ────────────────────────────────────────────────────

class TestFollowUpQuestions:
    def _ask_first(self, doc_id):
        with patch(
            "app.routes.conversation.watsonx_service.generate_text",
            return_value=_MOCK_ANSWER,
        ):
            client.post(
                "/api/conversation/ask",
                json={"question": "What is OSI?", "doc_id": doc_id},
            )

    def test_followup_question_returns_200(self, uploaded_doc):
        doc_id = uploaded_doc["doc_id"]
        self._ask_first(doc_id)
        with patch(
            "app.routes.conversation.watsonx_service.generate_text",
            return_value=_MOCK_FOLLOWUP,
        ):
            resp = client.post(
                "/api/conversation/ask",
                json={"question": "Tell me more about layer 1.", "doc_id": doc_id},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == _MOCK_FOLLOWUP

    def test_multiple_turns_stored_in_order(self, uploaded_doc):
        doc_id = uploaded_doc["doc_id"]
        questions = ["What is OSI?", "Explain layer 1.", "What about layer 7?"]
        answers = ["Answer 1", "Answer 2", "Answer 3"]

        for q, a in zip(questions, answers):
            with patch(
                "app.routes.conversation.watsonx_service.generate_text",
                return_value=a,
            ):
                client.post(
                    "/api/conversation/ask",
                    json={"question": q, "doc_id": doc_id},
                )

        hist_resp = client.get(f"/api/conversation/{doc_id}")
        turns = hist_resp.json()["turns"]
        assert len(turns) == 3
        for i, (q, a) in enumerate(zip(questions, answers)):
            assert turns[i]["question"] == q
            assert turns[i]["answer"] == a

    def test_followup_adds_to_history(self, uploaded_doc):
        doc_id = uploaded_doc["doc_id"]
        self._ask_first(doc_id)

        with patch(
            "app.routes.conversation.watsonx_service.generate_text",
            return_value=_MOCK_FOLLOWUP,
        ):
            client.post(
                "/api/conversation/ask",
                json={"question": "What is layer 1?", "doc_id": doc_id},
            )

        hist_resp = client.get(f"/api/conversation/{doc_id}")
        assert len(hist_resp.json()["turns"]) == 2


# ── 3. Context handling ───────────────────────────────────────────────────────

class TestContextHandling:
    def test_build_followup_prompt_no_history(self):
        fake_chunk = MagicMock()
        fake_chunk.text = "The OSI model has 7 layers."
        prompt = build_followup_prompt("What is OSI?", [fake_chunk], [])
        assert "What is OSI?" in prompt
        assert "OSI model has 7 layers" in prompt
        assert "PREVIOUS CONVERSATION" not in prompt

    def test_build_followup_prompt_with_history(self):
        from app.models.upload_models import ConversationTurn
        fake_chunk = MagicMock()
        fake_chunk.text = "The OSI model has 7 layers."
        history = [
            ConversationTurn(
                turn_id="1",
                question="What is OSI?",
                answer="It is a networking model.",
                learning_level="beginner",
                sources=[],
                timestamp="2024-01-01T00:00:00+00:00",
            )
        ]
        prompt = build_followup_prompt("Tell me more.", [fake_chunk], history)
        assert "PREVIOUS CONVERSATION" in prompt
        assert "What is OSI?" in prompt
        assert "It is a networking model." in prompt
        assert "Tell me more." in prompt

    def test_build_followup_prompt_history_capped_at_5(self):
        from app.models.upload_models import ConversationTurn, SourceChunk
        fake_chunk = MagicMock()
        fake_chunk.text = "Context text."
        history = [
            ConversationTurn(
                turn_id=str(i),
                question=f"Q{i}",
                answer=f"A{i}",
                learning_level="beginner",
                sources=[],
                timestamp="2024-01-01T00:00:00+00:00",
            )
            for i in range(10)
        ]
        prompt = build_followup_prompt("New question.", [fake_chunk], history)
        # Only the last 5 turns should appear — check Q0-Q4 are NOT in prompt
        # and Q5-Q9 ARE
        assert "Q9" in prompt
        assert "Q5" in prompt
        # Q0 through Q4 should be excluded from the capped history section
        # (they might appear incidentally in "New question." but Q0 is "Q0" which
        # should not be in the last-5 window)
        assert "Q0" not in prompt

    def test_history_context_passed_to_generate(self, uploaded_doc):
        """Verify that when a follow-up is asked, history context is in the prompt."""
        doc_id = uploaded_doc["doc_id"]

        # First turn
        with patch(
            "app.routes.conversation.watsonx_service.generate_text",
            return_value=_MOCK_ANSWER,
        ):
            client.post(
                "/api/conversation/ask",
                json={"question": "What is OSI?", "doc_id": doc_id},
            )

        # Second turn — capture the prompt sent to generate_text
        captured_prompts = []

        def capture(prompt):
            captured_prompts.append(prompt)
            return _MOCK_FOLLOWUP

        with patch(
            "app.routes.conversation.watsonx_service.generate_text",
            side_effect=capture,
        ):
            client.post(
                "/api/conversation/ask",
                json={"question": "Explain layer 1.", "doc_id": doc_id},
            )

        assert len(captured_prompts) == 1
        # The prompt sent on the second turn must contain the first Q&A
        assert "What is OSI?" in captured_prompts[0]

    def test_learning_level_stored_in_turn(self, uploaded_doc):
        doc_id = uploaded_doc["doc_id"]
        with patch(
            "app.routes.conversation.watsonx_service.generate_text",
            return_value=_MOCK_ANSWER,
        ):
            client.post(
                "/api/conversation/ask",
                json={
                    "question": "What is OSI?",
                    "doc_id": doc_id,
                    "learning_level": "expert",
                },
            )

        hist = client.get(f"/api/conversation/{doc_id}").json()["turns"]
        assert hist[0]["learning_level"] == "expert"


# ── 4. History retrieval ──────────────────────────────────────────────────────

class TestConversationHistory:
    def test_get_history_empty_for_new_doc(self, uploaded_doc):
        doc_id = uploaded_doc["doc_id"]
        resp = client.get(f"/api/conversation/{doc_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["doc_id"] == doc_id
        assert body["turns"] == []

    def test_get_history_response_shape(self, uploaded_doc):
        doc_id = uploaded_doc["doc_id"]
        with patch(
            "app.routes.conversation.watsonx_service.generate_text",
            return_value=_MOCK_ANSWER,
        ):
            client.post(
                "/api/conversation/ask",
                json={"question": "What is OSI?", "doc_id": doc_id},
            )

        resp = client.get(f"/api/conversation/{doc_id}")
        body = resp.json()
        assert set(body.keys()) >= {"success", "doc_id", "turns", "message"}
        assert len(body["turns"]) == 1
        turn = body["turns"][0]
        assert set(turn.keys()) >= {
            "turn_id", "question", "answer", "learning_level", "sources", "timestamp"
        }

    def test_get_history_turn_sources_shape(self, uploaded_doc):
        doc_id = uploaded_doc["doc_id"]
        with patch(
            "app.routes.conversation.watsonx_service.generate_text",
            return_value=_MOCK_ANSWER,
        ):
            client.post(
                "/api/conversation/ask",
                json={"question": "What is OSI?", "doc_id": doc_id},
            )

        resp = client.get(f"/api/conversation/{doc_id}")
        turn = resp.json()["turns"][0]
        for src in turn["sources"]:
            assert "filename" in src
            assert "chunk_index" in src
            assert "score" in src

    def test_history_not_shared_across_doc_ids(self, uploaded_doc):
        doc_id = uploaded_doc["doc_id"]
        with patch(
            "app.routes.conversation.watsonx_service.generate_text",
            return_value=_MOCK_ANSWER,
        ):
            client.post(
                "/api/conversation/ask",
                json={"question": "What is OSI?", "doc_id": doc_id},
            )

        other_resp = client.get("/api/conversation/other-doc-id")
        assert other_resp.json()["turns"] == []


# ── 5. Clear conversation ─────────────────────────────────────────────────────

class TestClearConversation:
    def test_clear_conversation_returns_200(self, uploaded_doc):
        doc_id = uploaded_doc["doc_id"]
        resp = client.delete(f"/api/conversation/{doc_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["doc_id"] == doc_id

    def test_clear_removes_all_turns(self, uploaded_doc):
        doc_id = uploaded_doc["doc_id"]
        # Add 2 turns
        for q in ["What is OSI?", "Explain layer 1."]:
            with patch(
                "app.routes.conversation.watsonx_service.generate_text",
                return_value=_MOCK_ANSWER,
            ):
                client.post(
                    "/api/conversation/ask",
                    json={"question": q, "doc_id": doc_id},
                )

        # Confirm 2 turns exist
        assert len(client.get(f"/api/conversation/{doc_id}").json()["turns"]) == 2

        # Clear
        client.delete(f"/api/conversation/{doc_id}")

        # Confirm empty
        assert client.get(f"/api/conversation/{doc_id}").json()["turns"] == []

    def test_clear_idempotent_on_empty_conversation(self, uploaded_doc):
        doc_id = uploaded_doc["doc_id"]
        # Clearing a non-existent conversation should not error
        resp = client.delete(f"/api/conversation/{doc_id}")
        assert resp.status_code == 200

    def test_can_ask_again_after_clear(self, uploaded_doc):
        doc_id = uploaded_doc["doc_id"]
        with patch(
            "app.routes.conversation.watsonx_service.generate_text",
            return_value=_MOCK_ANSWER,
        ):
            client.post(
                "/api/conversation/ask",
                json={"question": "What is OSI?", "doc_id": doc_id},
            )

        client.delete(f"/api/conversation/{doc_id}")

        with patch(
            "app.routes.conversation.watsonx_service.generate_text",
            return_value=_MOCK_FOLLOWUP,
        ):
            resp = client.post(
                "/api/conversation/ask",
                json={"question": "Fresh start: what is TCP?", "doc_id": doc_id},
            )

        assert resp.status_code == 200
        turns = client.get(f"/api/conversation/{doc_id}").json()["turns"]
        assert len(turns) == 1
        assert turns[0]["question"] == "Fresh start: what is TCP?"


# ── 6. API key / security ──────────────────────────────────────────────────────

class TestConversationSecurity:
    def test_api_key_not_in_response(self, uploaded_doc):
        with patch(
            "app.routes.conversation.watsonx_service.generate_text",
            return_value=_MOCK_ANSWER,
        ):
            resp = client.post(
                "/api/conversation/ask",
                json={"question": "What is OSI?"},
            )
        assert "WATSONX_API_KEY" not in resp.text

    def test_missing_api_key_returns_503(self, uploaded_doc):
        with patch(
            "app.routes.conversation.watsonx_service.generate_text",
            side_effect=WatsonxConfigError("WATSONX_API_KEY is not set"),
        ):
            resp = client.post(
                "/api/conversation/ask",
                json={"question": "What is OSI?"},
            )
        assert resp.status_code == 503
        assert "WATSONX_API_KEY" not in resp.json()["detail"]

    def test_auth_error_returns_503(self, uploaded_doc):
        with patch(
            "app.routes.conversation.watsonx_service.generate_text",
            side_effect=WatsonxAuthError("Bad credentials"),
        ):
            resp = client.post(
                "/api/conversation/ask",
                json={"question": "What is OSI?"},
            )
        assert resp.status_code == 503

    def test_network_error_returns_503(self, uploaded_doc):
        with patch(
            "app.routes.conversation.watsonx_service.generate_text",
            side_effect=WatsonxNetworkError("timeout"),
        ):
            resp = client.post(
                "/api/conversation/ask",
                json={"question": "What is OSI?"},
            )
        assert resp.status_code == 503

    def test_empty_response_returns_502(self, uploaded_doc):
        with patch(
            "app.routes.conversation.watsonx_service.generate_text",
            side_effect=WatsonxEmptyResponse("empty"),
        ):
            resp = client.post(
                "/api/conversation/ask",
                json={"question": "What is OSI?"},
            )
        assert resp.status_code == 502


# ── 7. Conversation store unit tests ──────────────────────────────────────────

class TestConversationStore:
    def test_add_turn_and_get_history(self):
        from app.services import conversation_store
        turn = conversation_store.add_turn(
            doc_id="doc1",
            question="What is TCP?",
            answer="TCP is a transport protocol.",
            learning_level="beginner",
            sources=[],
        )
        assert turn.question == "What is TCP?"
        assert turn.answer == "TCP is a transport protocol."
        history = conversation_store.get_history("doc1")
        assert len(history) == 1

    def test_get_history_empty_for_unknown_doc(self):
        from app.services import conversation_store
        assert conversation_store.get_history("unknown-doc") == []

    def test_clear_history_removes_turns(self):
        from app.services import conversation_store
        conversation_store.add_turn("doc2", "Q", "A", "beginner", [])
        conversation_store.clear_history("doc2")
        assert conversation_store.get_history("doc2") == []

    def test_clear_all_wipes_store(self):
        from app.services import conversation_store
        conversation_store.add_turn("doc3", "Q", "A", "beginner", [])
        conversation_store.add_turn("doc4", "Q", "A", "beginner", [])
        conversation_store.clear_all()
        assert conversation_store.get_history("doc3") == []
        assert conversation_store.get_history("doc4") == []

    def test_multiple_docs_independent(self):
        from app.services import conversation_store
        conversation_store.add_turn("docA", "Q1", "A1", "beginner", [])
        conversation_store.add_turn("docB", "Q2", "A2", "advanced", [])
        assert len(conversation_store.get_history("docA")) == 1
        assert len(conversation_store.get_history("docB")) == 1
        conversation_store.clear_history("docA")
        assert conversation_store.get_history("docA") == []
        assert len(conversation_store.get_history("docB")) == 1

    def test_turn_has_required_fields(self):
        from app.services import conversation_store
        turn = conversation_store.add_turn(
            doc_id="docX",
            question="Q",
            answer="A",
            learning_level="intermediate",
            sources=[],
        )
        assert turn.turn_id
        assert turn.timestamp
        assert turn.learning_level == "intermediate"
