"""
embedder.py — Phase 3 local embedding generation.

Model: all-MiniLM-L6-v2
  - 22 MB on disk
  - 384-dimensional output vectors
  - Fast on CPU, good semantic quality for retrieval
  - Runs entirely locally — no external API required

The SentenceTransformer is loaded once (module-level singleton) so it is
not re-initialised on every request.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

# ── Model selection ────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # fixed output dimension for this model

# ── Singleton ──────────────────────────────────────────────────────────────────
# _model is None until the first call to get_model() — lazy loading.
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Return the singleton SentenceTransformer, loading it on first call."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of strings.

    Returns a float32 numpy array of shape (len(texts), EMBEDDING_DIM).
    Vectors are L2-normalised so that dot-product == cosine similarity.
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    model = get_model()
    # encode returns (N, 384) float32; normalize_embeddings=True → unit vectors
    embeddings: np.ndarray = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.astype(np.float32)


def embed_query(question: str) -> np.ndarray:
    """
    Embed a single query string.

    Returns a shape-(1, EMBEDDING_DIM) float32 array.
    """
    return embed_texts([question])
