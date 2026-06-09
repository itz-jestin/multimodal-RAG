"""
vectordb/faiss_store.py
Thin wrapper around FAISS for storing and querying chunk embeddings.

Index type: IndexFlatIP (inner-product) — works as cosine similarity when
vectors are L2-normalised (which the embedder always does).

Persistence:
  - index   → <index_dir>/index.faiss
  - metadata → <index_dir>/metadata.json
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from loguru import logger

from utils import ensure_dir


# ── data model ────────────────────────────────────────────────────────────────

class FaissStore:
    """
    In-memory FAISS index with a parallel metadata list.
    Supports save/load for persistence.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(dim)
        self.metadata: list[dict] = []   # one entry per vector, in insertion order

    # ── ingestion ─────────────────────────────────────────────────────────────

    def add(self, vectors: np.ndarray, metadata: list[dict]) -> None:
        """
        Add *vectors* and their associated *metadata* to the store.

        Args:
            vectors:  Float32 array shape ``(n, dim)``.
            metadata: List of n dicts (e.g. chunk text, source, page number).
        """
        if vectors.shape[0] != len(metadata):
            raise ValueError(
                f"vectors ({vectors.shape[0]}) and metadata ({len(metadata)}) must have the same length."
            )
        vectors = np.asarray(vectors, dtype=np.float32)
        self.index.add(vectors)
        self.metadata.extend(metadata)
        logger.debug(f"Added {len(metadata)} vectors — total: {self.index.ntotal}")

    # ── retrieval ─────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Return the *top_k* most similar chunks for *query_vector*.

        Args:
            query_vector: Float32 array of shape ``(1, dim)`` or ``(dim,)``.
            top_k:        Number of results to return.

        Returns:
            List of dicts — each is the stored metadata plus ``"score"`` (0–1).
        """
        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty — no results.")
            return []

        qv = np.asarray(query_vector, dtype=np.float32).reshape(1, self.dim)
        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(qv, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            entry = {**self.metadata[idx], "score": float(score)}
            results.append(entry)

        logger.debug(f"Search returned {len(results)} results")
        return results

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, index_dir: str | Path) -> None:
        """Persist the FAISS index and metadata to *index_dir*."""
        index_dir = ensure_dir(index_dir)
        faiss.write_index(self.index, str(index_dir / "index.faiss"))
        with open(index_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        logger.success(f"Index saved to {index_dir} ({self.index.ntotal} vectors)")

    @classmethod
    def load(cls, index_dir: str | Path) -> "FaissStore":
        """Load a previously saved store from *index_dir*."""
        index_dir = Path(index_dir)
        index_path = index_dir / "index.faiss"
        meta_path = index_dir / "metadata.json"

        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"No saved index found in {index_dir}. "
                "Ingest documents first."
            )

        raw_index = faiss.read_index(str(index_path))
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)

        store = cls(dim=raw_index.d)
        store.index = raw_index
        store.metadata = metadata
        logger.success(f"Index loaded from {index_dir} ({raw_index.ntotal} vectors)")
        return store

    # ── helpers ───────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.index.ntotal

    def reset(self) -> None:
        """Clear the index and metadata (useful for re-ingestion)."""
        self.index.reset()
        self.metadata.clear()
        logger.info("FaissStore reset.")
