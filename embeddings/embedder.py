"""
embeddings/embedder.py
Wraps sentence-transformers to embed text chunks into dense vectors.
The model is loaded once and reused across calls (lazy singleton).
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from utils import get_env

# ── singleton ─────────────────────────────────────────────────────────────────
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        model_name = get_env("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        logger.info(f"Loading embedding model: {model_name}")
        _model = SentenceTransformer(model_name)
        logger.success(f"Embedding model loaded (dim={_model.get_sentence_embedding_dimension()})")
    return _model


# ── public API ────────────────────────────────────────────────────────────────

def embed_texts(
    texts: list[str],
    *,
    batch_size: int = 64,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Embed a list of strings.

    Args:
        texts:         Strings to embed.
        batch_size:    Batch size for the underlying model call.
        show_progress: Show a tqdm progress bar.

    Returns:
        Float32 numpy array of shape ``(len(texts), embedding_dim)``.
    """
    if not texts:
        raise ValueError("embed_texts: received an empty list")

    model = _get_model()
    logger.info(f"Embedding {len(texts)} text(s) (batch_size={batch_size})")

    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,   # unit-norm → cosine sim == dot product
    )
    logger.success(f"Embeddings shape: {vectors.shape}")
    return vectors.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.

    Returns:
        Float32 array of shape ``(1, embedding_dim)``.
    """
    return embed_texts([query])


def embedding_dim() -> int:
    """Return the dimension of the loaded embedding model."""
    return _get_model().get_sentence_embedding_dimension()
