"""
chunker.py — Phase 3 RAG text chunking.

Splits cleaned document text into overlapping chunks.
Each chunk carries metadata for later retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ── Configurable constants ────────────────────────────────────────────────────
CHUNK_SIZE = 1000      # target characters per chunk
CHUNK_OVERLAP = 150    # characters of overlap between consecutive chunks
MIN_CHUNK_SIZE = 50    # chunks smaller than this are discarded


@dataclass
class Chunk:
    """A single piece of a document with positional metadata."""
    doc_id: str
    filename: str
    chunk_index: int
    text: str


def split_into_chunks(
    text: str,
    doc_id: str,
    filename: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """
    Split *text* into overlapping character-based chunks.

    Strategy:
    1. Try to break at paragraph boundaries (double newlines) first.
    2. Within a paragraph-block, break at sentence-end punctuation.
    3. Fall back to hard character split if no natural break is found.

    This avoids cutting words or sentences unnecessarily.
    """
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    idx = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end < text_len:
            # Try to break at a paragraph boundary
            para_break = text.rfind("\n\n", start, end)
            if para_break != -1 and para_break > start + MIN_CHUNK_SIZE:
                end = para_break + 2  # include the double newline
            else:
                # Try to break at sentence end (. ! ?)
                for punct in (". ", "! ", "? ", ".\n", "!\n", "?\n"):
                    sent_break = text.rfind(punct, start, end)
                    if sent_break != -1 and sent_break > start + MIN_CHUNK_SIZE:
                        end = sent_break + len(punct)
                        break
                else:
                    # Break at a word boundary (space)
                    space_break = text.rfind(" ", start, end)
                    if space_break != -1 and space_break > start + MIN_CHUNK_SIZE:
                        end = space_break + 1

        chunk_text = text[start:end].strip()
        if len(chunk_text) >= MIN_CHUNK_SIZE:
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    filename=filename,
                    chunk_index=idx,
                    text=chunk_text,
                )
            )
            idx += 1

        # Advance start, stepping back by the overlap amount
        start = max(start + 1, end - overlap)

    return chunks
