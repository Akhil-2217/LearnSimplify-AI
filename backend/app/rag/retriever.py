"""
retriever.py — Phase 3 RAG retrieval orchestrator.

Public API used by routes and later phases:
    index_document_text(doc_id, filename, text)  → int  (number of chunks)
    retrieve(question, doc_id, top_k)            → list[RetrievedChunk]
"""

from __future__ import annotations

from app.rag import vector_store
from app.rag.chunker import split_into_chunks
from app.rag.embedder import embed_query, embed_texts
from app.rag.vector_store import RetrievedChunk

DEFAULT_TOP_K = 5


def index_document_text(doc_id: str, filename: str, text: str) -> int:
    """
    Chunk, embed, and index the given text under *doc_id*.

    Returns the number of chunks created.
    Raises ValueError if text is empty or no chunks are produced.
    """
    if not text or not text.strip():
        raise ValueError("Cannot index empty text.")

    chunks = split_into_chunks(text, doc_id=doc_id, filename=filename)
    if not chunks:
        raise ValueError("Text produced no indexable chunks.")

    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)

    vector_store.index_document(doc_id, chunks, embeddings)
    return len(chunks)


def retrieve(
    question: str,
    doc_id: str | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> list[RetrievedChunk]:
    """
    Embed *question* and return the top_k most relevant chunks.

    Raises ValueError on empty question or no indexed documents.
    """
    if not question or not question.strip():
        raise ValueError("Question must not be empty.")

    if doc_id and not vector_store.is_indexed(doc_id):
        raise ValueError(f"Document '{doc_id}' has not been indexed.")

    if not doc_id and not vector_store.get_doc_ids():
        raise ValueError("No documents have been indexed yet. Please upload a document first.")

    query_emb = embed_query(question.strip())
    return vector_store.search(query_emb, doc_id=doc_id, top_k=top_k)
