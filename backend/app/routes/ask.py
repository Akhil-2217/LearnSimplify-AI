"""
ask.py — POST /api/ask route.
Phase 4/5: retrieves relevant document chunks then calls IBM Granite to generate
a grounded educational answer at the requested learning level.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.upload_models import AskRequest, AskResponse, LearningLevel, SourceChunk
from app.rag.retriever import retrieve
from app.rag.vector_store import get_doc_ids
from app.services import watsonx_service
from app.services.watsonx_service import (
    WatsonxAuthError,
    WatsonxConfigError,
    WatsonxEmptyResponse,
    WatsonxError,
    WatsonxNetworkError,
)

router = APIRouter()

# Number of chunks to retrieve for context (can be tuned)
_CONTEXT_TOP_K = 5

# Maximum characters of chunk text to include in the prompt
_MAX_CONTEXT_CHARS = 3000

# ── Per-level instructions injected into the prompt ───────────────────────────
_LEVEL_INSTRUCTIONS: dict[str, str] = {
    LearningLevel.beginner: (
        "Use very simple language. Avoid jargon. Explain every term. "
        "Write short sentences and use everyday analogies."
    ),
    LearningLevel.intermediate: (
        "Use clear language. Define specialised terms when first introduced. "
        "Include relevant examples from the course material."
    ),
    LearningLevel.advanced: (
        "Use precise technical language. Assume familiarity with core concepts. "
        "Highlight nuances and connect ideas across the material."
    ),
    LearningLevel.expert: (
        "Use expert-level language with full technical depth. "
        "Reference specific mechanisms, trade-offs, and edge cases from the material."
    ),
}


# ── Prompt construction ───────────────────────────────────────────────────────

def build_prompt(
    question: str,
    context_chunks: list,
    learning_level: LearningLevel = LearningLevel.beginner,
) -> str:
    """
    Build a grounded educational prompt from retrieved chunks.

    The prompt instructs Granite to answer based only on the provided
    course material and to admit when it does not have enough information.
    The explanation style is adapted to *learning_level*.
    """
    # Concatenate chunk texts, separated clearly
    context_parts: list[str] = []
    total_chars = 0
    for chunk in context_chunks:
        snippet = chunk.text.strip()
        if not snippet:
            continue
        if total_chars + len(snippet) > _MAX_CONTEXT_CHARS:
            # Trim the last snippet so we stay within the limit
            remaining = _MAX_CONTEXT_CHARS - total_chars
            if remaining > 100:
                snippet = snippet[:remaining] + "…"
                context_parts.append(snippet)
            break
        context_parts.append(snippet)
        total_chars += len(snippet)

    context_text = "\n\n---\n\n".join(context_parts)
    level_instruction = _LEVEL_INSTRUCTIONS.get(learning_level, _LEVEL_INSTRUCTIONS[LearningLevel.beginner])
    level_label = learning_level.value.capitalize() if hasattr(learning_level, "value") else str(learning_level).capitalize()

    prompt = (
        "You are an educational course-content assistant helping a student understand "
        "their uploaded course material.\n\n"
        "Use ONLY the course material provided below to answer the student's question. "
        "Your answer must be grounded in the provided context. "
        "Do not invent information or claim it came from the course material. "
        "If the context does not contain enough information to answer the question, "
        'clearly say: "I couldn\'t find enough information about this in the uploaded '
        'course material."\n\n'
        f"The student's learning level is: {level_label}. {level_instruction}\n\n"
        "=== COURSE MATERIAL ===\n"
        f"{context_text}\n\n"
        "=== END OF COURSE MATERIAL ===\n\n"
        f"Student question: {question}\n\n"
        "Answer:"
    )
    return prompt


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post("/ask", response_model=AskResponse, tags=["Q&A"])
async def ask_question(request: AskRequest) -> AskResponse:
    """
    Retrieve relevant document chunks and generate a grounded answer using
    IBM Granite.

    Steps:
    1. Validate the question.
    2. Retrieve top-k relevant chunks via the existing RAG pipeline.
    3. Build a grounded prompt.
    4. Call IBM Granite through watsonx.ai.
    5. Return the answer and source citations.
    """
    # ── 1. Validate question ──────────────────────────────────────────────────
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    # ── 2. Check documents are indexed ───────────────────────────────────────
    doc_ids = get_doc_ids()
    if not doc_ids:
        raise HTTPException(
            status_code=404,
            detail=(
                "No documents have been indexed yet. "
                "Please upload a document before asking a question."
            ),
        )

    if request.document_id and request.document_id not in doc_ids:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Document ID '{request.document_id}' not found. "
                "Please upload the document first."
            ),
        )

    top_k = max(1, min(request.top_k or _CONTEXT_TOP_K, 20))

    # ── 3. Retrieve relevant chunks ───────────────────────────────────────────
    try:
        chunks = retrieve(
            question=request.question.strip(),
            doc_id=request.document_id,
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
        return AskResponse(
            success=True,
            answer=(
                "I couldn't find enough information about this in the uploaded "
                "course material."
            ),
            sources=[],
            message="No relevant content found for this question.",
        )

    # ── 4. Build prompt ───────────────────────────────────────────────────────
    prompt = build_prompt(
        request.question.strip(),
        chunks,
        learning_level=request.learning_level,
    )

    # ── 5. Call IBM Granite ───────────────────────────────────────────────────
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

    # ── 6. Build response ─────────────────────────────────────────────────────
    sources = [
        SourceChunk(
            filename=chunk.filename,
            chunk_index=chunk.chunk_index,
            score=round(chunk.score, 4),
        )
        for chunk in chunks
    ]

    return AskResponse(
        success=True,
        answer=answer,
        sources=sources,
        message=f"Answer generated from {len(sources)} source chunk(s).",
    )
