"""
Pydantic request/response models for Phase 2/3/4/5/6 — document upload, RAG retrieval,
IBM Granite Q&A, summarization, quiz generation, and conversation history.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel


# ── Upload ─────────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    success: bool
    doc_id: str
    filename: str
    file_type: str
    text_length: int
    chunk_count: int
    message: str


# ── Retrieval ──────────────────────────────────────────────────────────────────

class RetrieveRequest(BaseModel):
    question: str
    doc_id: str | None = None
    top_k: int = 5


class ChunkResult(BaseModel):
    text: str
    filename: str
    chunk_index: int
    score: float


class RetrieveResponse(BaseModel):
    success: bool
    question: str
    doc_id: str | None
    results: list[ChunkResult]
    message: str


# ── Phase 5 — Learning Level ───────────────────────────────────────────────────

class LearningLevel(str, Enum):
    """Learner proficiency levels that control how Granite explains content."""
    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"
    expert = "expert"


# ── Ask (Phase 4/5 — RAG + IBM Granite + Learning Level) ──────────────────────

class AskRequest(BaseModel):
    question: str
    document_id: str | None = None
    top_k: int = 5
    learning_level: LearningLevel = LearningLevel.beginner


class SourceChunk(BaseModel):
    """A document chunk cited as a source for the answer."""
    filename: str
    chunk_index: int
    score: float


class AskResponse(BaseModel):
    success: bool
    answer: str
    sources: list[SourceChunk]
    message: str


# ── Phase 5 — Summarize ────────────────────────────────────────────────────────

class SummaryLength(str, Enum):
    """How long / detailed the summary should be."""
    short = "short"
    medium = "medium"
    detailed = "detailed"


class SummarizeRequest(BaseModel):
    document_id: str | None = None
    length: SummaryLength = SummaryLength.medium


class SummarizeResponse(BaseModel):
    success: bool
    summary: str
    length: str
    message: str


# ── Phase 5 — Quiz ────────────────────────────────────────────────────────────

class QuizOption(BaseModel):
    """One A/B/C/D option for a quiz question."""
    label: str          # "A", "B", "C", or "D"
    text: str


class QuizQuestion(BaseModel):
    """A single multiple-choice question."""
    number: int
    question: str
    options: list[QuizOption]
    correct_answer: str   # e.g. "A"
    explanation: str


class QuizRequest(BaseModel):
    document_id: str | None = None


class QuizResponse(BaseModel):
    success: bool
    questions: list[QuizQuestion]
    message: str


# ── Phase 6 — Conversation History ────────────────────────────────────────────

class ConversationTurn(BaseModel):
    """A single question-answer exchange in a conversation."""
    turn_id: str
    question: str
    answer: str
    learning_level: str
    sources: list[SourceChunk]
    timestamp: str


class ConversationHistoryResponse(BaseModel):
    """Response containing the full conversation history for a document."""
    success: bool
    doc_id: str
    turns: list[ConversationTurn]
    message: str


class FollowUpRequest(BaseModel):
    """Request to ask a follow-up question within an existing conversation."""
    question: str
    doc_id: str | None = None
    top_k: int = 5
    learning_level: LearningLevel = LearningLevel.beginner


class ClearConversationResponse(BaseModel):
    """Response after clearing a conversation."""
    success: bool
    doc_id: str | None
    message: str
