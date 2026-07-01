# Multimodal PDF Image Pipeline — Design

## Goal

Extract embedded images from uploaded PDFs, describe them with a **Gemma-family vision model**, and use those **text captions** during RAG answers — **without storing image bytes or image embeddings in Chroma**.

## Architecture

```text
Upload PDF
  ├─ Text path (existing): PyPDFLoader → chunk → MiniLM → Chroma
  └─ Image path (new): PyMuPDF → PNG on disk → Gemma caption → SQLite manifest

Question
  ├─ Text retrieval from Chroma (unchanged)
  └─ Join image captions for retrieved (source, page) pairs
       → append caption text to LLM context
```

Images are **references on disk** plus **metadata in SQLite** (`image_manifest.db`). Chroma remains **text-only**.

## Components

| Module | Role |
|--------|------|
| `pdf_image_extractor.py` | PyMuPDF extraction, size filter, dedup by hash |
| `image_store.py` | SQLite manifest (`pdf_images` table) |
| `image_captioner.py` | Gemma vision captioning via OpenRouter or Gemini |
| `pdf_image_pipeline.py` | Orchestrates extract → store → caption on upload |
| `image_rag.py` | Appends caption blocks to retrieved chunks at answer time |

## Data model (`pdf_images`)

- `image_id`, `session_id`, `source`, `page`, `page_label`, `image_index`
- `file_path`, `width`, `height`, `bytes_sha256`
- `caption`, `caption_model`, `created_at`

Files live under `./data/extracted_images/{session_id}/{source_stem}/`.

## Configuration (`.env`)

```env
IMAGE_EXTRACTION_ENABLED=true
IMAGE_CAPTION_ENABLED=true
IMAGE_CAPTION_PROVIDER=openrouter
IMAGE_CAPTION_MODEL=google/gemma-3-12b-it
IMAGE_MIN_WIDTH=50
IMAGE_MIN_HEIGHT=50
```

Use a **vision-capable** Gemma model. Text-only models (e.g. `google/gemma-2-9b-it:free`) cannot caption images.

## RAG integration

1. **Ingest:** After PDF is saved to `uploaded_pdfs/`, run `process_pdf_images(session_id, source, pdf_path)`.
2. **Retrieve:** `get_retriever(vector_store, session_id)` wraps the balanced retriever with `ImageEnrichingRetriever`.
3. **Answer:** Captions for matching pages are appended as plain text, e.g. `[Images on report.pdf, page 3]`.

Page-targeted and multi-document overview paths call `enrich_documents_with_image_context()` explicitly.

## API

- `GET /api/images?session_id=&source=&page=` — list manifest rows
- `GET /api/images/{image_id}/file` — serve extracted image bytes
- Upload/status responses include per-file `images` count

## Limitations

- **Embedded images only** — full-page scans without embedded image objects require page rendering (future stretch).
- **Caption cost/latency** — captions run at ingest when `IMAGE_CAPTION_ENABLED=true`.
- **Session scope** — images are namespaced by `session_id` (Streamlit and React API each have one).

## Future stretch

- Query-time captioning for uncaptioned images
- Thumbnails in source citation panel
- OCR fallback for scanned PDFs
- Optional caption-as-text chunks in Chroma (still not image vectors)
