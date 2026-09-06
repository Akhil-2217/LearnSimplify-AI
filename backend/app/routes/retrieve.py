"""
retrieve.py — POST /api/retrieve route.
Phase 3: RAG similarity search over indexed document chunks.
Does NOT call IBM Granite (that is Phase 4).
"""

from fastapi import APIRouter, HTTPException

from app.models.upload_models import ChunkResult, RetrieveRequest, RetrieveResponse
from app.rag.retriever import retrieve
from app.rag.vector_store import get_doc_ids

router = APIRouter()


@router.post("/retrieve", response_model=RetrieveResponse, tags=["RAG"])
async def retrieve_chunks(request: RetrieveRequest) -> RetrieveResponse:
    """
    Embed the question and return the most relevant document chunks.

    - question: the student's question
    - doc_id: (optional) restrict search to a specific document
    - top_k: number of chunks to return (default 5)

    Returns chunks sorted by similarity score (highest first).
    Does NOT generate an AI answer.
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    doc_ids = get_doc_ids()
    if not doc_ids:
        raise HTTPException(
            status_code=404,
            detail="No documents have been indexed yet. Please upload a document first.",
        )

    if request.doc_id and request.doc_id not in doc_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Document ID '{request.doc_id}' not found. Please upload the document first.",
        )

    top_k = max(1, min(request.top_k or 5, 20))  # clamp 1–20

    try:
        results = retrieve(
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

    if not results:
        return RetrieveResponse(
            success=True,
            question=request.question,
            doc_id=request.doc_id,
            results=[],
            message="No relevant content found for this question in the uploaded material.",
        )

    chunk_results = [
        ChunkResult(
            text=r.text,
            filename=r.filename,
            chunk_index=r.chunk_index,
            score=round(r.score, 4),
        )
        for r in results
    ]

    return RetrieveResponse(
        success=True,
        question=request.question,
        doc_id=request.doc_id,
        results=chunk_results,
        message=f"Found {len(chunk_results)} relevant chunk(s).",
    )
