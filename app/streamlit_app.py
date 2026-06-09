"""
app/streamlit_app.py
Multimodal RAG — Streamlit UI

Tabs:
  1. Ingest    — upload PDFs / images, build the FAISS index
  2. Query     — ask questions (text or image query), see citations
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── make project root importable ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from loguru import logger
from PIL import Image

from embeddings.embedder import embed_texts, embedding_dim
from extraction.image_ocr import ocr_image
from extraction.pdf_reader import extract_pdf
from llm.generator import generate_answer_streaming
from preprocessing.chunking import chunk_pages, TextChunk
from retrieval.retriever import Retriever
from utils import ensure_dir, get_env
from vectordb.faiss_store import FaissStore

# ── paths ─────────────────────────────────────────────────────────────────────
INDEX_DIR = ROOT / get_env("FAISS_INDEX_PATH", "data/faiss_index")
PDF_DIR   = ROOT / "data" / "pdfs"
IMG_DIR   = ROOT / "data" / "images"

for d in (INDEX_DIR, PDF_DIR, IMG_DIR):
    ensure_dir(d)



# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multimodal RAG",
    page_icon="🔍",
    layout="wide",
)

st.title("Multimodal RAG")
st.caption("Upload PDFs & images → build an index → ask questions")

tab_ingest, tab_query = st.tabs(["📥 Ingest", "💬 Query"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — INGEST
# ═══════════════════════════════════════════════════════════════════════════════

with tab_ingest:
    st.subheader("Upload documents")

    col1, col2 = st.columns(2)

    with col1:
        uploaded_pdfs = st.file_uploader(
            "PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader",
        )

    with col2:
        uploaded_images = st.file_uploader(
            "Image files (PNG / JPG)",
            type=["png", "jpg", "jpeg", "tiff", "bmp"],
            accept_multiple_files=True,
            key="img_uploader",
        )

    if st.button("⚡ Build / Update Index", type="primary"):
        if not uploaded_pdfs and not uploaded_images:
            st.warning("Please upload at least one file before building the index.")
        else:
            all_chunks: list[TextChunk] = []

            progress = st.progress(0, text="Starting …")
            total_files = len(uploaded_pdfs) + len(uploaded_images)
            file_count = 0

            # ── process PDFs ──────────────────────────────────────────────────
            for pdf_file in uploaded_pdfs:
                dest = PDF_DIR / pdf_file.name
                dest.write_bytes(pdf_file.read())

                progress.progress(
                    file_count / total_files,
                    text=f"Extracting PDF: {pdf_file.name}",
                )
                with st.spinner(f"Extracting {pdf_file.name} …"):
                    pages = extract_pdf(dest, extract_images=True)

                # OCR any scanned (text-less) pages
                for page in pages:
                    if not page.text.strip() and page.images:
                        page.text = ocr_image(page.images[0])

                chunks = chunk_pages(pages)
                all_chunks.extend(chunks)
                file_count += 1

            # ── process images ────────────────────────────────────────────────
            for img_file in uploaded_images:
                dest = IMG_DIR / img_file.name
                dest.write_bytes(img_file.read())

                progress.progress(
                    file_count / total_files,
                    text=f"OCR: {img_file.name}",
                )
                with st.spinner(f"Running OCR on {img_file.name} …"):
                    pil_img = Image.open(dest).convert("RGB")
                    text = ocr_image(pil_img)

                if text.strip():
                    from preprocessing.chunking import chunk_text
                    chunks = chunk_text(
                        text,
                        source=str(dest),
                        page_number=1,
                        extra_metadata={"type": "image"},
                    )
                    all_chunks.extend(chunks)
                else:
                    st.warning(f"No text found in {img_file.name} — skipping.")

                file_count += 1

            # ── embed & store ─────────────────────────────────────────────────
            if not all_chunks:
                st.error("No text could be extracted from the uploaded files.")
            else:
                progress.progress(0.9, text="Embedding chunks …")
                with st.spinner(f"Embedding {len(all_chunks)} chunks …"):
                    texts  = [c.text for c in all_chunks]
                    metas  = [
                        {
                            "text": c.text,
                            "source": c.source,
                            "page_number": c.page_number,
                            "chunk_id": c.chunk_id,
                        }
                        for c in all_chunks
                    ]
                    vectors = embed_texts(texts, show_progress=False)

                    dim = embedding_dim()
                    store = FaissStore(dim)
                    store.add(vectors, metas)
                    store.save(INDEX_DIR)

                progress.progress(1.0, text="Done!")
                st.success(
                    f"✅ Index built — {len(all_chunks)} chunks from "
                    f"{len(uploaded_pdfs)} PDF(s) and {len(uploaded_images)} image(s)."
                )
                st.session_state["index_ready"] = True


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — QUERY
# ═══════════════════════════════════════════════════════════════════════════════

with tab_query:
    st.subheader("Ask a question")

    # ── load index ────────────────────────────────────────────────────────────
    @st.cache_resource(show_spinner="Loading index …")
    def load_retriever() -> Retriever | None:
        try:
            return Retriever.from_index_dir(INDEX_DIR)
        except FileNotFoundError:
            return None

    retriever = load_retriever()

    if retriever is None and not st.session_state.get("index_ready"):
        st.info("No index found. Go to the **Ingest** tab to upload documents first.")
    else:
        if st.session_state.get("index_ready"):
            # Reload after fresh ingest
            st.cache_resource.clear()
            retriever = load_retriever()

        query_col, img_col = st.columns([3, 1])

        with query_col:
            query = st.text_area(
                "Your question",
                placeholder="e.g. What are the main findings in the report?",
                height=100,
            )

        with img_col:
            query_image = st.file_uploader(
                "Optional query image",
                type=["png", "jpg", "jpeg"],
                help="Upload an image — its text will be included in the query.",
            )
            if query_image:
                st.image(query_image, use_column_width=True)

        top_k = st.slider("Number of context chunks", min_value=1, max_value=10, value=5)

        if st.button("🔎 Search & Answer", type="primary"):
            if not query.strip() and not query_image:
                st.warning("Please enter a question or upload a query image.")
            else:
                pil_query_image = None
                if query_image:
                    pil_query_image = Image.open(query_image).convert("RGB")

                with st.spinner("Retrieving …"):
                    chunks = retriever.retrieve(
                        query,
                        top_k=top_k,
                        image=pil_query_image,
                    )

                if not chunks:
                    st.warning("No relevant chunks found — try a different query.")
                else:
                    # ── answer ────────────────────────────────────────────────
                    st.markdown("### Answer")
                    answer_placeholder = st.empty()
                    full_answer = ""
                    with st.spinner("Generating …"):
                        for delta in generate_answer_streaming(query, chunks):
                            full_answer += delta
                            answer_placeholder.markdown(full_answer + "▌")
                    answer_placeholder.markdown(full_answer)

                    # ── sources ───────────────────────────────────────────────
                    with st.expander(f"📄 Sources ({len(chunks)} chunks)", expanded=False):
                        for i, chunk in enumerate(chunks, start=1):
                            source = Path(chunk.get("source", "unknown")).name
                            page   = chunk.get("page_number", "?")
                            score  = chunk.get("score", 0.0)
                            text   = chunk.get("text", "")
                            st.markdown(
                                f"**[{i}]** `{source}` — page {page} "
                                f"*(relevance: {score:.3f})*"
                            )
                            st.caption(text[:400] + ("…" if len(text) > 400 else ""))
                            st.divider()
