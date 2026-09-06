"""
conversation.py — Phase 6 conversation routes.

Endpoints
---------
    POST /api/conversation/ask      Follow-up Q&A with conversation context
    GET  /api/conversation/{doc_id} Retrieve conversation history
    DELETE /api/conversation/{doc_id} Clear conversation history

The follow-up endpoint reuses the existing RAG retrieval pipeline from
Phase 4/5 (ask.py) and adds conversation context to the IBM Granite prompt
so follow-up questions are answered in the context of prior exchanges.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.upload_models import (
    AskResponse,
    ClearConversationResponse,
    ConversationHistoryResponse,
    FollowUpRequest,
    LearningLevel,
    SourceChunk,
)
from app.rag.retriever import retrieve
from app.rag.vector_store import get_doc_ids
from app.routes.ask import _CONTEXT_TOP_K, _LEVEL_INSTRUCTIONS, _MAX_CONTEXT_CHARS
from app.services import conversation_store, watsonx_service
from app.services.watsonx_service import (
    WatsonxAuthError,
    WatsonxConfigError,
    WatsonxEmptyResponse,
    WatsonxError,
    WatsonxNetworkError,
)

router = APIRouter(prefix="/conversation", tags=["Conversation"])

# How many recent turns to include as context in the follow-up prompt.
# Including every past turn would balloon the prompt; 5 is a practical limit.
_MAX_HISTORY_TURNS = 5


# ── Prompt construction ───────────────────────────────────────────────────────

def build_followup_prompt(
    question: str,
    context_chunks: list,
    history: list,
    learning_level: LearningLevel = LearningLevel.beginner,
) -> str:
    """
    Build a grounded follow-up prompt that includes:
    - Retrieved document chunks (RAG context)
    - The last _MAX_HISTORY_TURNS Q&A exchanges for conversational context
    - The new question

    The model is instructed to use only the document content, but may refer to
    the prior conversation for continuity.
    """
    # --- Document context ---
    context_parts: list[str] = []
    total_chars = 0
    for chunk in context_chunks:
        snippet = chunk.text.strip()
        if not snippet:
            continue
        if total_chars + len(snippet) > _MAX_CONTEXT_CHARS:
            remaining = _MAX_CONTEXT_CHARS - total_chars
            if remaining > 100:
                snippet = snippet[:remaining] + "…"
                context_parts.append(snippet)
            break
        context_parts.append(snippet)
        total_chars += len(snippet)

    context_text = "\n\n---\n\n".join(context_parts)

    # --- Conversation history (most recent N turns) ---
    recent = history[-_MAX_HISTORY_TURNS:] if history else []
    history_lines: list[str] = []
    for turn in recent:
        history_lines.append(f"Student: {turn.question}")
        history_lines.append(f"Assistant: {turn.answer}")
    history_text = "\n".join(history_lines)

    level_instruction = _LEVEL_INSTRUCTIONS.get(
        learning_level, _LEVEL_INSTRUCTIONS[LearningLevel.beginner]
    )
    level_label = (
        learning_level.value.capitalize()
        if hasattr(learning_level, "value")
        else str(learning_level).capitalize()
    )

    # --- Build the full prompt ---
    parts = [
        "You are an educational course-content assistant helping a student understand "
        "their uploaded course material.\n\n"
        "Use ONLY the course material provided below to answer the student's question. "
        "Your answer must be grounded in the provided context. "
        "Do not invent information or claim it came from the course material. "
        "If the context does not contain enough information to answer the question, "
        "clearly say: \"I couldn't find enough information about this in the uploaded "
        "course material.\"\n\n"
        f"The student's learning level is: {level_label}. {level_instruction}\n\n",
    ]

    if history_text:
        parts.append(
            "=== PREVIOUS CONVERSATION ===\n"
            f"{history_text}\n\n"
            "=== END OF PREVIOUS CONVERSATION ===\n\n"
        )

    parts.append(
        "=== COURSE MATERIAL ===\n"
        f"{context_text}\n\n"
        "=== END OF COURSE MATERIAL ===\n\n"
        f"Student question: {question}\n\n"
        "Answer:"
    )

    return "".join(parts)


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/ask", response_model=AskResponse)
async def conversation_ask(request: FollowUpRequest) -> AskResponse:
    """
    Ask a follow-up question with conversation context.

    Behaves identically to POST /api/ask but additionally:
    - Loads prior conversation turns for the given document.
    - Injects those turns into the IBM Granite prompt so the model can
      answer follow-up questions coherently.
    - Saves the new Q&A turn to the conversation history.
    """
    # ── 1. Validate question ──────────────────────────────────────────────────
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    # ── 2. Check documents ────────────────────────────────────────────────────
    doc_ids = get_doc_ids()
    if not doc_ids:
        raise HTTPException(
            status_code=404,
            detail=(
                "No documents have been indexed yet. "
                "Please upload a document before asking a question."
            ),
        )

    if request.doc_id and request.doc_id not in doc_ids:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Document ID '{request.doc_id}' not found. "
                "Please upload the document first."
            ),
        )

    effective_doc_id = request.doc_id or (list(doc_ids)[0] if len(doc_ids) == 1 else None)
    top_k = max(1, min(request.top_k or _CONTEXT_TOP_K, 20))

    # ── 3. Load conversation history ──────────────────────────────────────────
    # Use the effective doc_id for history; fall back to a global "no-doc" key
    # if the user did not supply one.
    history_key = effective_doc_id or "__global__"
    history = conversation_store.get_history(history_key)

    # ── 4. Retrieve relevant chunks ───────────────────────────────────────────
    try:
        chunks = retrieve(
            question=request.question.strip(),
            doc_id=request.doc_id,
            top_k=top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Retrieval failed due to an internal error.",
        ) from exc

    if not chunks:
        answer = (
            "I couldn't find enough information about this in the uploaded "
            "course material."
        )
        sources: list[SourceChunk] = []
        # Still save this turn so conversation flow is preserved
        conversation_store.add_turn(
            doc_id=history_key,
            question=request.question.strip(),
            answer=answer,
            learning_level=request.learning_level.value,
            sources=sources,
        )
        return AskResponse(
            success=True,
            answer=answer,
            sources=sources,
            message="No relevant content found for this question.",
        )

    # ── 5. Build follow-up prompt with history ────────────────────────────────
    prompt = build_followup_prompt(
        question=request.question.strip(),
        context_chunks=chunks,
        history=history,
        learning_level=request.learning_level,
    )

    # ── 6. Call IBM Granite ───────────────────────────────────────────────────
    try:
        answer = watsonx_service.generate_text(prompt)
    except WatsonxConfigError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "The AI service is not configured. "
                "Please contact the administrator to set up the API key."
            ),
        ) from exc
    except WatsonxAuthError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Authentication with IBM watsonx.ai failed. "
                "Please contact the administrator."
            ),
        ) from exc
    except WatsonxNetworkError as exc:
        raise HTTPException(
            status_code=503,
            detail="Could not reach the IBM watsonx.ai service. Please try again later.",
        ) from exc
    except WatsonxEmptyResponse as exc:
        raise HTTPException(
            status_code=502,
            detail="The AI model returned an empty response. Please try rephrasing your question.",
        ) from exc
    except WatsonxError as exc:
        raise HTTPException(
            status_code=502,
            detail="The AI service encountered an error. Please try again.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while generating the answer.",
        ) from exc

    # ── 7. Build sources list ─────────────────────────────────────────────────
    sources = [
        SourceChunk(
            filename=chunk.filename,
            chunk_index=chunk.chunk_index,
            score=round(chunk.score, 4),
        )
        for chunk in chunks
    ]

    # ── 8. Save turn to conversation history ─────────────────────────────────
    conversation_store.add_turn(
        doc_id=history_key,
        question=request.question.strip(),
        answer=answer,
        learning_level=request.learning_level.value,
        sources=sources,
    )

    return AskResponse(
        success=True,
        answer=answer,
        sources=sources,
        message=f"Answer generated from {len(sources)} source chunk(s).",
    )


@router.get("/{doc_id}", response_model=ConversationHistoryResponse)
async def get_conversation_history(doc_id: str) -> ConversationHistoryResponse:
    """
    Return the stored conversation history for the given document.

    Returns an empty turns list if no conversation exists yet.
    """
    turns = conversation_store.get_history(doc_id)
    return ConversationHistoryResponse(
        success=True,
        doc_id=doc_id,
        turns=turns,
        message=f"{len(turns)} turn(s) in conversation history.",
    )


@router.delete("/{doc_id}", response_model=ClearConversationResponse)
async def clear_conversation(doc_id: str) -> ClearConversationResponse:
    """
    Clear the conversation history for the given document.

    Safe to call even if no conversation exists.
    """
    conversation_store.clear_history(doc_id)
    return ClearConversationResponse(
        success=True,
        doc_id=doc_id,
        message="Conversation history cleared.",
    )
