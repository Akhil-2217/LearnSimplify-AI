"""
document_processor.py — Phase 2/3 document processing service.

Extracts text from PDF, DOCX, and TXT files (Phase 2).
Also triggers RAG indexing after extraction (Phase 3).
"""

from __future__ import annotations

import io
import re
import unicodedata
from pathlib import Path

# ── In-memory document store (Phase 2 — no database) ─────────────────────────
# Stores {session_key: {"filename": str, "text": str, "file_type": str}}
# In Phase 3 this dict will be extended with chunk/embedding data.
_document_store: dict[str, dict] = {}

# ── Constants ─────────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

# Sentinel key used while only one document is active (Phase 2 simplification)
CURRENT_DOC_KEY = "current"


# ── Public helpers ─────────────────────────────────────────────────────────────

def is_allowed_extension(filename: str) -> bool:
    """Return True if *filename* has a supported extension."""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def clean_text(raw: str) -> str:
    """
    Normalise Unicode, collapse whitespace, and remove control characters.
    Preserves paragraph structure by keeping single newlines between paragraphs.
    """
    # Normalise unicode to NFC form
    text = unicodedata.normalize("NFC", raw)
    # Remove control characters (except newline / tab)
    text = re.sub(r"[^\S\n\t ]+", " ", text)
    # Collapse multiple spaces/tabs on a single line into one space
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Collapse 3+ consecutive newlines to two
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)
    return text.strip()


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF byte stream using PyMuPDF."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages_text: list[str] = []
        for page in doc:
            pages_text.append(page.get_text())
        doc.close()
        return "\n".join(pages_text)
    except Exception as exc:
        raise ValueError(f"Failed to extract text from PDF: {exc}") from exc


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from a DOCX byte stream using python-docx."""
    try:
        from docx import Document

        doc = Document(io.BytesIO(file_bytes))
        paragraphs = [para.text for para in doc.paragraphs]
        return "\n".join(paragraphs)
    except Exception as exc:
        raise ValueError(f"Failed to extract text from DOCX: {exc}") from exc


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Decode a TXT byte stream as UTF-8 (with fallback to latin-1)."""
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return file_bytes.decode("latin-1")
        except Exception as exc:
            raise ValueError(f"Failed to decode TXT file: {exc}") from exc


def process_document(filename: str, file_bytes: bytes) -> dict:
    """
    Main entry point for Phase 2/3.

    Validates, extracts, cleans, stores, and RAG-indexes the document.

    Returns a dict suitable for building an UploadResponse.
    Raises ValueError with a user-friendly message on any failure.
    """
    import uuid
    from app.rag.retriever import index_document_text

    # ── 1. File size check ────────────────────────────────────────────────────
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"File is too large ({len(file_bytes) / 1_048_576:.1f} MB). "
            f"Maximum allowed size is 10 MB."
        )

    # ── 2. Extension check ────────────────────────────────────────────────────
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Please upload a PDF, DOCX, or TXT file."
        )

    # ── 3. Text extraction ────────────────────────────────────────────────────
    if ext == ".pdf":
        raw_text = extract_text_from_pdf(file_bytes)
        file_type = "pdf"
    elif ext == ".docx":
        raw_text = extract_text_from_docx(file_bytes)
        file_type = "docx"
    else:  # .txt
        raw_text = extract_text_from_txt(file_bytes)
        file_type = "txt"

    # ── 4. Clean ──────────────────────────────────────────────────────────────
    cleaned = clean_text(raw_text)

    # ── 5. Empty document check ───────────────────────────────────────────────
    if not cleaned:
        raise ValueError(
            "The uploaded document appears to be empty or contains no readable text."
        )

    # ── 6. Assign a stable document ID ────────────────────────────────────────
    doc_id = str(uuid.uuid4())

    # ── 7. Store raw text for Phase 4 prompt building ─────────────────────────
    _document_store[CURRENT_DOC_KEY] = {
        "doc_id": doc_id,
        "filename": filename,
        "text": cleaned,
        "file_type": file_type,
    }

    # ── 8. RAG indexing (Phase 3) ─────────────────────────────────────────────
    try:
        chunk_count = index_document_text(doc_id, filename, cleaned)
    except Exception as exc:
        raise ValueError(f"Document indexing failed: {exc}") from exc

    return {
        "doc_id": doc_id,
        "filename": filename,
        "file_type": file_type,
        "text_length": len(cleaned),
        "chunk_count": chunk_count,
    }


def get_current_document() -> dict | None:
    """Return the currently loaded document dict, or None if nothing is uploaded."""
    return _document_store.get(CURRENT_DOC_KEY)


def clear_document_store() -> None:
    """Remove all stored documents (used by tests)."""
    _document_store.clear()
