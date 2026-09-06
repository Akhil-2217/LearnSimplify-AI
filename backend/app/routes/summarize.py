"""
summarize.py — POST /api/summarize route.
Phase 5: generates a short, medium, or detailed summary of the uploaded document
using the existing RAG pipeline and IBM Granite.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.upload_models import (
    SummarizeRequest,
    SummarizeResponse,
    SummaryLength,
)
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

# How many chunks to pull for summarisation (more than Q&A for broader coverage)
_SUMMARY_TOP_K = 10

# Maximum context characters fed into the summary prompt
_MAX_CONTEXT_CHARS = 4000

# Per-length instructions
_LENGTH_INSTRUCTIONS: dict[str, str] = {
    SummaryLength.short: (
        "Write a SHORT summary in 3–5 sentences. "
        "Cover only the most essential points."
    ),
    SummaryLength.medium: (
        "Write a MEDIUM-LENGTH summary in 2–3 paragraphs. "
        "Cover the main topics and key concepts."
    ),
    SummaryLength.detailed: (
        "Write a DETAILED summary. "
        "Cover all major topics, key concepts, examples, and conclusions "
        "found in the material. Use headings or bullet points where helpful."
    ),
}


# ── Prompt construction ───────────────────────────────────────────────────────

def build_summary_prompt(context_chunks: list, length: SummaryLength) -> str:
    """Build a summarisation prompt from document chunks."""
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
    length_instruction = _LENGTH_INSTRUCTIONS.get(length, _LENGTH_INSTRUCTIONS[SummaryLength.medium])

    prompt = (
        "You are an educational assistant. Your job is to summarise the following "
        "course material for a student.\n\n"
        "Use ONLY the course material provided below. "
        "Do not add information that is not in the material.\n\n"
        f"{length_instruction}\n\n"
        "=== COURSE MATERIAL ===\n"
        f"{context_text}\n\n"
        "=== END OF COURSE MATERIAL ===\n\n"
        "Summary:"
    )
    return prompt


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post("/summarize", response_model=SummarizeResponse, tags=["Learning"])
async def summarize_document(request: SummarizeRequest) -> SummarizeResponse:
    """
    Generate a summary of the uploaded document.

    Steps:
    1. Validate that at least one document is indexed.
    2. Retrieve a broad set of chunks via the RAG pipeline
       (using a generic "summarize the document" query).
    3. Build a summarisation prompt for the requested length.
    4. Call IBM Granite through watsonx.ai.
    5. Return the summary.
    """
    # ── 1. Check documents ────────────────────────────────────────────────────
    doc_ids = get_doc_ids()
    if not doc_ids:
        raise HTTPException(
            status_code=404,
            detail=(
                "No documents have been indexed yet. "
                "Please upload a document before requesting a summary."
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

    # ── 2. Retrieve chunks ────────────────────────────────────────────────────
    # A broad query fetches representative chunks from across the document.
    try:
        chunks = retrieve(
            question="summarize the main topics and key concepts of this document",
            doc_id=request.document_id,
            top_k=_SUMMARY_TOP_K,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Retrieval failed due to an internal error.",
        ) from exc

    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="Could not retrieve any content from the document to summarise.",
        )

    # ── 3. Build prompt ───────────────────────────────────────────────────────
    prompt = build_summary_prompt(chunks, request.length)

    # ── 4. Call IBM Granite ───────────────────────────────────────────────────
    try:
        summary = watsonx_service.generate_text(prompt)
    except WatsonxConfigError as exc:
        raise HTTPException(
            status_code=503,
            detail="The AI service is not configured. Please contact the administrator.",
        ) from exc
    except WatsonxAuthError as exc:
        raise HTTPException(
            status_code=503,
            detail="Authentication with IBM watsonx.ai failed. Please contact the administrator.",
        ) from exc
    except WatsonxNetworkError as exc:
        raise HTTPException(
            status_code=503,
            detail="Could not reach the IBM watsonx.ai service. Please try again later.",
        ) from exc
    except WatsonxEmptyResponse as exc:
        raise HTTPException(
            status_code=502,
            detail="The AI model returned an empty response. Please try again.",
        ) from exc
    except WatsonxError as exc:
        raise HTTPException(
            status_code=502,
            detail="The AI service encountered an error. Please try again.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while generating the summary.",
        ) from exc

    return SummarizeResponse(
        success=True,
        summary=summary,
        length=request.length.value,
        message=f"{request.length.value.capitalize()} summary generated from {len(chunks)} chunk(s).",
    )
