"""
conversation_store.py — Phase 6 in-memory conversation history store.

Stores conversation turns per document ID.  Each turn captures the question,
the generated answer, the learning level used, and the source citations.

The store lives for the lifetime of the server process — on restart it resets.
This is intentional and consistent with the existing in-memory vector store
and document processor from Phases 2–5.

Public API
----------
    add_turn(doc_id, question, answer, learning_level, sources) → ConversationTurn
    get_history(doc_id)     → list[ConversationTurn]
    clear_history(doc_id)   → None
    clear_all()             → None          (used in tests)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.upload_models import ConversationTurn, SourceChunk

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

# Maps doc_id → list of ConversationTurn (ordered oldest → newest)
_store: dict[str, list] = {}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def add_turn(
    doc_id: str,
    question: str,
    answer: str,
    learning_level: str,
    sources: list,
) -> "ConversationTurn":
    """
    Append a new Q&A turn to the conversation for *doc_id* and return it.

    Imports ConversationTurn and SourceChunk at call time to avoid circular
    imports (models → services is fine; the reverse is avoided here).
    """
    from app.models.upload_models import ConversationTurn  # local import

    if doc_id not in _store:
        _store[doc_id] = []

    turn = ConversationTurn(
        turn_id=str(uuid.uuid4()),
        question=question,
        answer=answer,
        learning_level=learning_level,
        sources=sources,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    _store[doc_id].append(turn)
    return turn


def get_history(doc_id: str) -> list:
    """Return all conversation turns for *doc_id* (empty list if none)."""
    return list(_store.get(doc_id, []))


def clear_history(doc_id: str) -> None:
    """Delete all conversation turns for *doc_id*."""
    _store.pop(doc_id, None)


def clear_all() -> None:
    """Wipe the entire store (used in tests to reset state between cases)."""
    _store.clear()
