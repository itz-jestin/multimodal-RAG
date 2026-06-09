"""
extraction/pdf_reader.py
Extracts text (and optionally embedded images) from PDF files using PyMuPDF.
Each page becomes one RawPage dataclass so downstream code stays decoupled.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF
from loguru import logger
from PIL import Image


@dataclass
class RawPage:
    """Holds everything extracted from a single PDF page."""
    source: str          # original file path
    page_number: int     # 1-based
    text: str            # plain text (may be empty for scanned pages)
    images: list[Image.Image] = field(default_factory=list)  # embedded images
    metadata: dict = field(default_factory=dict)


def extract_pdf(
    pdf_path: str | Path,
    *,
    extract_images: bool = True,
    dpi: int = 150,
) -> list[RawPage]:
    """
    Extract every page from *pdf_path*.

    Args:
        pdf_path:       Path to the PDF file.
        extract_images: Whether to extract embedded raster images.
        dpi:            Resolution used when rasterising a page for OCR fallback.

    Returns:
        Ordered list of RawPage objects (one per page).
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info(f"Opening PDF: {pdf_path.name}")
    pages: list[RawPage] = []

    with fitz.open(str(pdf_path)) as doc:
        doc_meta = {
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "pages": doc.page_count,
        }
        logger.debug(f"PDF metadata: {doc_meta}")

        for page_index in range(doc.page_count):
            page = doc[page_index]
            text = page.get_text("text").strip()
            images: list[Image.Image] = []

            if extract_images:
                images = _extract_images_from_page(doc, page)

            # If no text was found, rasterise the page so OCR can handle it
            if not text and extract_images:
                logger.debug(f"Page {page_index + 1}: no text layer — rasterising for OCR")
                pix = page.get_pixmap(dpi=dpi)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                images.append(img)

            pages.append(
                RawPage(
                    source=str(pdf_path),
                    page_number=page_index + 1,
                    text=text,
                    images=images,
                    metadata={**doc_meta, "page": page_index + 1},
                )
            )
            logger.debug(
                f"  Page {page_index + 1}/{doc.page_count}: "
                f"{len(text)} chars, {len(images)} image(s)"
            )

    logger.success(f"Extracted {len(pages)} pages from {pdf_path.name}")
    return pages


def iter_pdf_folder(
    folder: str | Path,
    **kwargs,
) -> Iterator[tuple[Path, list[RawPage]]]:
    """Yield (pdf_path, pages) for every PDF in *folder*."""
    folder = Path(folder)
    pdf_files = sorted(folder.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {folder}")
    for pdf_file in pdf_files:
        yield pdf_file, extract_pdf(pdf_file, **kwargs)


# ── helpers ───────────────────────────────────────────────────────────────────

def _extract_images_from_page(doc: fitz.Document, page: fitz.Page) -> list[Image.Image]:
    """Return all raster images embedded in *page*."""
    images: list[Image.Image] = []
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        try:
            base_img = doc.extract_image(xref)
            raw = base_img["image"]
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            images.append(img)
        except Exception as exc:
            logger.warning(f"Could not extract image xref={xref}: {exc}")
    return images
