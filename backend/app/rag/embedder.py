"""
embedder.py — Lightweight embedding generation.

Model: BAAI/bge-small-en-v1.5
384-dimensional embeddings.
"""

from __future__ import annotations

import numpy as np
from fastembed import TextEmbedding

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

_model: TextEmbedding | None = None


def get_model() -> TextEmbedding:
    """Return the singleton embedding model."""
    global _model

    if _model is None:
        _model = TextEmbedding(
            model_name=EMBEDDING_MODEL_NAME,
            threads=1,
        )

    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Generate normalized 384-dimensional embeddings."""
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    model = get_model()

    embeddings = np.asarray(
        list(model.embed(texts, batch_size=1)),
        dtype=np.float32,
    )

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)

    return embeddings.astype(np.float32)


def embed_query(question: str) -> np.ndarray:
    """Generate an embedding for a single query."""
    return embed_texts([question])