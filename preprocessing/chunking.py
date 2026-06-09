"""
preprocessing/chunking.py
Splits raw extracted text into overlapping chunks suitable for embedding.

Strategy:
  1. Use LangChain's RecursiveCharacterTextSplitter (sentence-aware).
  2. Each chunk carries metadata so we can trace it back to its source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger


# ── data model ────────────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    """A single piece of text ready for embedding."""
    chunk_id: str          # unique identifier: "{source}::p{page}::c{idx}"
    text: str              # the chunk content
    source: str            # original file path
    page_number: int       # 1-based page from which this chunk came
    chunk_index: int       # 0-based position within the page's chunks
    metadata: dict = field(default_factory=dict)


# ── public API ────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    source: str,
    page_number: int,
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    extra_metadata: dict | None = None,
) -> list[TextChunk]:
    """
    Split *text* into overlapping chunks.

    Args:
        text:           Raw page text.
        source:         Original file path (for tracing).
        page_number:    1-based page index.
        chunk_size:     Target token/character budget per chunk.
        chunk_overlap:  Overlap between consecutive chunks.
        extra_metadata: Additional key-value pairs merged into each chunk's metadata.

    Returns:
        List of TextChunk objects.
    """
    if not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    raw_chunks = splitter.split_text(text)
    meta_base = {"source": source, "page": page_number, **(extra_metadata or {})}

    chunks = []
    for idx, raw in enumerate(raw_chunks):
        raw = raw.strip()
        if not raw:
            continue
        chunk_id = f"{source}::p{page_number}::c{idx}"
        chunks.append(
            TextChunk(
                chunk_id=chunk_id,
                text=raw,
                source=source,
                page_number=page_number,
                chunk_index=idx,
                metadata={**meta_base, "chunk_index": idx, "chunk_id": chunk_id},
            )
        )

    logger.debug(
        f"Page {page_number} of '{source}': "
        f"{len(text)} chars → {len(chunks)} chunks"
    )
    return chunks


def chunk_pages(
    pages: Sequence,          # list[RawPage] — avoids circular import
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    include_ocr: bool = True,
) -> list[TextChunk]:
    """
    Chunk all pages from a document.

    If a page has no native text but has images and *include_ocr* is True,
    the caller is expected to have already run OCR and placed the result in
    ``page.text`` (the pipeline in ``app/streamlit_app.py`` does this).

    Args:
        pages:        List of RawPage objects.
        chunk_size:   Characters per chunk.
        chunk_overlap: Overlap between chunks.
        include_ocr:  Log a warning when a page has no text (instead of skipping silently).

    Returns:
        Flat list of TextChunk objects for the whole document.
    """
    all_chunks: list[TextChunk] = []

    for page in pages:
        text = page.text or ""
        if not text.strip():
            if include_ocr:
                logger.warning(
                    f"Page {page.page_number} of '{page.source}' has no text. "
                    "Run OCR first if this is a scanned page."
                )
            continue

        chunks = chunk_text(
            text,
            source=page.source,
            page_number=page.page_number,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        all_chunks.extend(chunks)

    logger.info(
        f"Total chunks from '{pages[0].source if pages else 'unknown'}': {len(all_chunks)}"
    )
    return all_chunks
