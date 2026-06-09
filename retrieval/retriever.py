"""
retrieval/retriever.py
High-level retriever: embeds a query, searches FAISS, returns ranked chunks.
Also handles multi-modal queries: if an image is supplied it is first OCR'd,
and the text is prepended to the query string before embedding.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from PIL import Image

from embeddings.embedder import embed_query
from extraction.image_ocr import ocr_image
from utils import get_env
from vectordb.faiss_store import FaissStore


class Retriever:
    """
    Retrieves relevant chunks from a FaissStore.

    Usage::

        store = FaissStore.load("data/faiss_index")
        retriever = Retriever(store)
        results = retriever.retrieve("What is the capital of France?", top_k=5)
    """

    def __init__(self, store: FaissStore, top_k: int | None = None) -> None:
        self.store = store
        self.top_k = top_k or int(get_env("TOP_K_RESULTS", "5"))

    # ── public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        image: Image.Image | str | Path | None = None,
    ) -> list[dict]:
        """
        Retrieve the most relevant chunks for *query*.

        Args:
            query:  Natural-language question or keyword string.
            top_k:  Override the default number of results.
            image:  Optional query image; its OCR text is prepended to *query*.

        Returns:
            List of metadata dicts (from FaissStore) with an added ``"score"`` key,
            sorted by descending relevance.
        """
        k = top_k or self.top_k

        # ── multi-modal: OCR the query image and prepend ──────────────────────
        if image is not None:
            logger.info("Query image provided — running OCR …")
            ocr_text = ocr_image(image)
            if ocr_text:
                query = f"{ocr_text}\n\n{query}"
                logger.debug(f"OCR prefix ({len(ocr_text)} chars) prepended to query")

        if not query.strip():
            logger.warning("Empty query after preprocessing — returning no results.")
            return []

        # ── embed ─────────────────────────────────────────────────────────────
        logger.info(f"Retrieving top-{k} chunks for query: {query[:80]!r}")
        query_vec = embed_query(query)

        # ── search ────────────────────────────────────────────────────────────
        results = self.store.search(query_vec, top_k=k)
        logger.success(f"Retrieved {len(results)} chunk(s)")
        return results

    def retrieve_texts(
        self,
        query: str,
        **kwargs,
    ) -> list[str]:
        """Convenience wrapper — returns just the chunk text strings."""
        return [r["text"] for r in self.retrieve(query, **kwargs)]

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_index_dir(
        cls,
        index_dir: str | Path,
        **kwargs,
    ) -> "Retriever":
        """Load a FaissStore from *index_dir* and return a ready Retriever."""
        store = FaissStore.load(index_dir)
        return cls(store, **kwargs)
