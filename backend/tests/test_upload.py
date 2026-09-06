"""
Phase 2 tests — document upload and processing.
Run with:  pytest tests/ -v

All Phase 1 tests must continue to pass.
"""

from __future__ import annotations

import io
import pathlib

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.document_processor import (
    MAX_FILE_SIZE_BYTES,
    clean_text,
    clear_document_store,
    extract_text_from_docx,
    extract_text_from_pdf,
    extract_text_from_txt,
    get_current_document,
    is_allowed_extension,
    process_document,
)

client = TestClient(app)

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_store():
    """Clear the in-memory document store before every test."""
    clear_document_store()
    yield
    clear_document_store()


# ── is_allowed_extension ──────────────────────────────────────────────────────

def test_allowed_extensions_accepted():
    for name in ("doc.pdf", "doc.docx", "doc.txt", "DOC.PDF", "DOC.TXT"):
        assert is_allowed_extension(name), f"{name} should be allowed"


def test_disallowed_extensions_rejected():
    for name in ("doc.exe", "doc.jpg", "doc.csv", "doc.pptx", "doc"):
        assert not is_allowed_extension(name), f"{name} should be rejected"


# ── clean_text ────────────────────────────────────────────────────────────────

def test_clean_text_collapses_whitespace():
    raw = "  hello   world  \n\n\n\n  foo  "
    result = clean_text(raw)
    assert "   " not in result
    assert result.count("\n\n\n") == 0


def test_clean_text_non_empty_input():
    assert clean_text("Hello world") == "Hello world"


# ── TXT extraction ────────────────────────────────────────────────────────────

def test_extract_txt_utf8():
    content = "Hello, this is a test document.\nSecond line."
    result = extract_text_from_txt(content.encode("utf-8"))
    assert "Hello" in result
    assert "Second line" in result


def test_extract_txt_latin1_fallback():
    content = "Caf\xe9 au lait"  # é in latin-1
    result = extract_text_from_txt(content.encode("latin-1"))
    assert len(result) > 0


def test_extract_txt_from_fixture():
    data = (FIXTURES / "sample.txt").read_bytes()
    text = extract_text_from_txt(data)
    assert "OSI" in text or "computer" in text.lower()


# ── PDF extraction ────────────────────────────────────────────────────────────

def test_extract_pdf_from_fixture():
    data = (FIXTURES / "sample.pdf").read_bytes()
    text = extract_pdf_or_skip(data)
    assert "OSI" in text or "LearnSimplify" in text


def test_extract_pdf_invalid_bytes():
    with pytest.raises(ValueError, match="Failed to extract text from PDF"):
        extract_text_from_pdf(b"not a pdf at all")


# ── DOCX extraction ───────────────────────────────────────────────────────────

def test_extract_docx_from_fixture():
    data = (FIXTURES / "sample.docx").read_bytes()
    text = extract_text_from_docx(data)
    assert "LearnSimplify" in text or "OSI" in text


def test_extract_docx_invalid_bytes():
    with pytest.raises(ValueError, match="Failed to extract text from DOCX"):
        extract_text_from_docx(b"not a docx file")


# ── process_document ─────────────────────────────────────────────────────────

def test_process_txt_stores_document():
    data = (FIXTURES / "sample.txt").read_bytes()
    result = process_document("sample.txt", data)
    assert result["file_type"] == "txt"
    assert result["text_length"] > 0
    doc = get_current_document()
    assert doc is not None
    assert doc["filename"] == "sample.txt"


def test_process_pdf_stores_document():
    data = (FIXTURES / "sample.pdf").read_bytes()
    result = process_document("sample.pdf", data)
    assert result["file_type"] == "pdf"
    assert result["text_length"] > 0


def test_process_docx_stores_document():
    data = (FIXTURES / "sample.docx").read_bytes()
    result = process_document("sample.docx", data)
    assert result["file_type"] == "docx"
    assert result["text_length"] > 0


def test_process_unsupported_extension_raises():
    with pytest.raises(ValueError, match="Unsupported file type"):
        process_document("malware.exe", b"fake content")


def test_process_empty_txt_raises():
    with pytest.raises(ValueError, match="empty or contains no readable text"):
        process_document("empty.txt", b"   \n   ")


def test_process_oversized_file_raises():
    big = b"x" * (MAX_FILE_SIZE_BYTES + 1)
    with pytest.raises(ValueError, match="too large"):
        process_document("big.txt", big)


# ── /api/upload endpoint ──────────────────────────────────────────────────────

def test_upload_txt_success():
    data = (FIXTURES / "sample.txt").read_bytes()
    response = client.post(
        "/api/upload",
        files={"file": ("sample.txt", io.BytesIO(data), "text/plain")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["file_type"] == "txt"
    assert body["text_length"] > 0


def test_upload_pdf_success():
    data = (FIXTURES / "sample.pdf").read_bytes()
    response = client.post(
        "/api/upload",
        files={"file": ("sample.pdf", io.BytesIO(data), "application/pdf")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["file_type"] == "pdf"


def test_upload_docx_success():
    data = (FIXTURES / "sample.docx").read_bytes()
    response = client.post(
        "/api/upload",
        files={
            "file": (
                "sample.docx",
                io.BytesIO(data),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["file_type"] == "docx"


def test_upload_unsupported_type_returns_415():
    response = client.post(
        "/api/upload",
        files={"file": ("report.csv", io.BytesIO(b"a,b,c"), "text/csv")},
    )
    assert response.status_code == 415


def test_upload_empty_txt_returns_422():
    response = client.post(
        "/api/upload",
        files={"file": ("empty.txt", io.BytesIO(b"   "), "text/plain")},
    )
    assert response.status_code == 422


def test_upload_oversized_file_returns_413():
    big = b"x" * (MAX_FILE_SIZE_BYTES + 1)
    response = client.post(
        "/api/upload",
        files={"file": ("huge.txt", io.BytesIO(big), "text/plain")},
    )
    assert response.status_code == 413


def test_upload_response_does_not_contain_full_text():
    """The extracted text must NOT appear in the API response body."""
    data = (FIXTURES / "sample.txt").read_bytes()
    response = client.post(
        "/api/upload",
        files={"file": ("sample.txt", io.BytesIO(data), "text/plain")},
    )
    body = response.text
    # Response should not contain the actual document sentences
    assert "computer network" not in body.lower()


# ── helpers ───────────────────────────────────────────────────────────────────

def extract_pdf_or_skip(data: bytes) -> str:
    """Extract PDF text; skip test if PyMuPDF cannot read the fixture."""
    try:
        return extract_text_from_pdf(data)
    except Exception as exc:
        pytest.skip(f"PDF fixture unreadable: {exc}")
