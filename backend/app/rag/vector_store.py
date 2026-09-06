"""
vector_store.py — Phase 3 in-memory FAISS vector store.

Stores chunk embeddings per document and supports cosine-similarity search.
Because embeddings are L2-normalised (unit vectors), inner-product search
is equivalent to cosine similarity.

Architecture (simple, no database):
    {doc_id: DocumentIndex}

DocumentIndex holds:
    - faiss.IndexFlatIP  (inner-product / cosine index)
    - list of Chunk objects (parallel to index rows)
"""

from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np

from app.rag.chunker import Chunk
from app.rag.embedder import EMBEDDING_DIM


@dataclass
class RetrievedChunk:
    """A chunk returned by a similarity search."""
    doc_id: str
    filename: str
    chunk_index: int
    text: str
    score: float


class DocumentIndex:
    """FAISS index + chunk list for a single document."""

    def __init__(self, doc_id: str) -> None:
        self.doc_id = doc_id
        # IndexFlatIP: exact inner-product search (cosine after L2-norm)
        self._index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self._chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Add chunks and their embeddings to this index."""
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks and embeddings must have the same length")
        self._index.add(embeddings)
        self._chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        """Return the top_k most similar chunks for a query embedding."""
        if self._index.ntotal == 0:
            return []

        k = min(top_k, self._index.ntotal)
        # query_embedding shape: (1, EMBEDDING_DIM)
        scores, indices = self._index.search(query_embedding, k)

        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for empty slots
                continue
            chunk = self._chunks[idx]
            results.append(
                RetrievedChunk(
                    doc_id=chunk.doc_id,
                    filename=chunk.filename,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    score=float(score),
                )
            )
        return results

    @property
    def chunk_count(self) -> int:
        return self._index.ntotal


# ── Global registry ────────────────────────────────────────────────────────────
# {doc_id: DocumentIndex}
_registry: dict[str, DocumentIndex] = {}


def index_document(doc_id: str, chunks: list[Chunk], embeddings: np.ndarray) -> None:
    """Index a document's chunks and embeddings, replacing any existing index."""
    di = DocumentIndex(doc_id)
    di.add(chunks, embeddings)
    _registry[doc_id] = di


def search(
    query_embedding: np.ndarray,
    doc_id: str | None = None,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """
    Search the vector store.

    If *doc_id* is provided, search only that document's index.
    Otherwise, search all indexed documents and merge results.
    """
    if doc_id:
        if doc_id not in _registry:
            return []
        results = _registry[doc_id].search(query_embedding, top_k)
    else:
        # Search all documents and merge
        all_results: list[RetrievedChunk] = []
        for di in _registry.values():
            all_results.extend(di.search(query_embedding, top_k))
        results = sorted(all_results, key=lambda r: r.score, reverse=True)[:top_k]

    return sorted(results, key=lambda r: r.score, reverse=True)


def get_doc_ids() -> list[str]:
    """Return all currently indexed document IDs."""
    return list(_registry.keys())


def is_indexed(doc_id: str) -> bool:
    """Return True if *doc_id* has been indexed."""
    return doc_id in _registry


def clear_store() -> None:
    """Remove all indexed documents (used by tests)."""
    _registry.clear()
