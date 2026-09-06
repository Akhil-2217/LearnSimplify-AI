"""
upload.py — POST /api/upload route.
Phase 2: accepts multipart/form-data, validates and extracts document text.
"""

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.models.upload_models import UploadResponse
from app.services.document_processor import (
    MAX_FILE_SIZE_BYTES,
    is_allowed_extension,
    process_document,
)

router = APIRouter()


@router.post("/upload", response_model=UploadResponse, tags=["Upload"])
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """
    Accept a PDF, DOCX, or TXT file and extract its text.

    - Validates file type and size.
    - Extracts and cleans the text.
    - Stores the text in memory for later RAG use (Phase 3+).
    - Returns metadata about the processed document.
    """
    # ── 1. Basic filename / extension validation ───────────────────────────────
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided.",
        )

    if not is_allowed_extension(file.filename):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type. "
                f"Please upload a PDF, DOCX, or TXT file."
            ),
        )

    # ── 2. Read file bytes (enforcing size limit) ─────────────────────────────
    # Read in chunks so we can bail out early if the file is too large.
    max_bytes = MAX_FILE_SIZE_BYTES
    chunks: list[bytes] = []
    total_read = 0

    while True:
        chunk = await file.read(65_536)  # 64 KB chunks
        if not chunk:
            break
        total_read += len(chunk)
        if total_read > max_bytes:
            raise HTTPException(
                status_code=413,
                detail="File exceeds the 10 MB limit. Please upload a smaller file.",
            )
        chunks.append(chunk)

    file_bytes = b"".join(chunks)

    # ── 3. Process (extract + clean + store) ──────────────────────────────────
    try:
        result = process_document(file.filename, file_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    return UploadResponse(
        success=True,
        doc_id=result["doc_id"],
        filename=result["filename"],
        file_type=result["file_type"],
        text_length=result["text_length"],
        chunk_count=result["chunk_count"],
        message="Document processed and indexed successfully.",
    )
