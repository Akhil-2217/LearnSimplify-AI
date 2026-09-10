"""
embedder.py — Local embedding generation.

Model: all-MiniLM-L6-v2
384-dimensional embeddings.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Return the singleton model, loading it only once."""
    global _model

    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Generate normalized 384-dimensional embeddings."""
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    model = get_model()

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=8,
    )

    return embeddings.astype(np.float32)


def embed_query(question: str) -> np.ndarray:
    """Generate an embedding for a single question."""
    return embed_texts([question])