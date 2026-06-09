# Multimodal RAG

A Retrieval-Augmented Generation system that ingests **PDFs and images**, extracts text (including via OCR for scanned content), builds a FAISS vector index, and answers questions using **Claude** — all via a Streamlit UI.

---

## Architecture

```
multimodal-rag/
│
├── app/                  ← Streamlit UI (Ingest + Query tabs)
│   └── streamlit_app.py
│
├── data/
│   ├── pdfs/             ← uploaded PDFs land here
│   ├── images/           ← uploaded images land here
│   └── faiss_index/      ← persisted FAISS index (auto-created)
│
├── extraction/
│   ├── pdf_reader.py     ← PyMuPDF text + image extraction
│   └── image_ocr.py      ← Tesseract OCR with OpenCV pre-processing
│
├── preprocessing/
│   └── chunking.py       ← Recursive text splitter → TextChunk dataclass
│
├── embeddings/
│   └── embedder.py       ← sentence-transformers (lazy singleton)
│
├── vectordb/
│   └── faiss_store.py    ← IndexFlatIP + metadata list, save/load
│
├── retrieval/
│   └── retriever.py      ← embed query, search index, multi-modal support
│
├── llm/
│   └── generator.py      ← Claude via Anthropic SDK, streaming support
│
├── utils/
│   └── __init__.py       ← env loading, path helpers
│
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Tesseract must also be installed at the OS level:

```bash
# macOS
brew install tesseract

# Ubuntu / Debian
sudo apt-get install tesseract-ocr
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 3. Run the app

```bash
streamlit run app/streamlit_app.py
```

---

## Usage

| Tab | What to do |
|-----|-----------|
| **📥 Ingest** | Upload PDFs and/or images, then click **Build / Update Index** |
| **💬 Query** | Type a question (optionally attach a query image) and click **Search & Answer** |

### Tips
- Scanned PDFs are handled automatically — pages without a text layer are rasterised and passed through OCR.
- For image uploads, OCR is run directly on the file.
- You can attach an image to your *query* too; its OCR text is prepended to the search string.
- Sources and relevance scores are shown in the collapsible **Sources** panel below each answer.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | *(required)* | Your Anthropic API key |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace sentence-transformers model |
| `FAISS_INDEX_PATH` | `data/faiss_index` | Where to save/load the FAISS index |
| `TOP_K_RESULTS` | `5` | Default number of chunks to retrieve |
