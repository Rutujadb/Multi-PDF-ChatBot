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

| Column | Purpose |
|--------|---------|
| `image_id` | Primary key |
| `session_id` | Namespace per chat session |
| `source`, `page`, `page_label`, `image_index` | Locate image in a PDF |
| `file_path` | On-disk PNG/JPEG |
| `width`, `height`, `bytes_sha256` | Size filter + dedup |
| `caption`, `caption_model` | Gemma-generated text |
| `created_at` | ISO timestamp |

Files live under `./data/extracted_images/{session_id}/{source_stem}/`.

## Configuration (`.env`)

```env
IMAGE_EXTRACTION_ENABLED=true
IMAGE_CAPTION_ENABLED=true
IMAGE_CAPTION_PROVIDER=openrouter
IMAGE_CAPTION_MODEL=google/gemma-3-12b-it
IMAGE_MIN_WIDTH=50
IMAGE_MIN_HEIGHT=50
# IMAGE_DB_PATH=./data/image_manifest.db
# EXTRACTED_IMAGES_DIR=./data/extracted_images
```

Use a **vision-capable** Gemma model. Text-only models (e.g. `google/gemma-2-9b-it:free`) cannot caption images.

## Ingest pipeline

1. Save PDF to `uploaded_pdfs/`
2. `process_pdf_images(session_id, source, pdf_path)` — extract + optional caption
3. Load text with PyPDFLoader → split into chunks
4. **If no text chunks:** `build_caption_chunks_for_sources()` creates page-level text from image captions
5. Embed caption/text chunks into Chroma (text only — not image vectors)
6. Validate `indexed_files` and RAG `chain` before showing success

Streamlit shows *Indexed from extracted image captions instead* when step 4 runs.

## RAG integration

1. **Ingest:** Images and captions stay outside Chroma; only text enters the vector index.
2. **Retrieve:** `get_retriever(vector_store, session_id)` wraps the balanced retriever with `ImageEnrichingRetriever`.
3. **Answer:** Captions for matching pages are appended as plain text, e.g. `[Images on report.pdf, page 3]`.
4. **Page/overview paths:** `enrich_documents_with_image_context()` injects captions into explicit document sets.

## API

| Endpoint | Purpose |
|----------|---------|
| `GET /api/images?session_id=&source=&page=` | List manifest rows |
| `GET /api/images/{image_id}/file` | Serve extracted image bytes |
| `POST /api/upload` | Response includes `image_summary` and per-file `images` count |

## Design constraint (course requirement)

> Do not store images in the vector store. Extract image → LLM (Gemma) → use caption text.

| Store | What |
|-------|------|
| Chroma | Text chunks + text embeddings only |
| Disk | Raw PNG/JPEG bytes |
| SQLite | Image paths + Gemma captions |
| LLM context | Caption text appended at retrieval time |

## Logging

Every image pipeline module (`pdf_image_extractor`, `image_store`, `image_captioner`, `pdf_image_pipeline`, `image_rag`) uses Python's `logging` library. Logs include:

- Image extraction counts and page scans
- Caption API calls (provider, model, image name, response length)
- Manifest DB inserts, updates, and deletes
- Retriever enrichment steps and document counts
- Errors with full stack traces

Set `LOG_LEVEL=DEBUG` for verbose output (e.g. per-image captioning details).

## Limitations

- **Embedded images only** — full-page scans without embedded image objects require page rendering (future stretch).
- **Caption cost/latency** — captions run at ingest when `IMAGE_CAPTION_ENABLED=true`.
- **Session scope** — images are namespaced by `session_id` (Streamlit and React API each have one).
- **Caption failures** — if Gemma rate-limits, extraction still works but chat may lack caption fallback for image-only PDFs.

## Future stretch

- Query-time captioning for uncaptioned images
- Thumbnails in source citation panel
- OCR fallback for scanned PDFs
- Page render + vision for scan-only pages

## Related documents

- [DESIGN.md](./DESIGN.md) — full system architecture
- [USER_MANUAL.md](./USER_MANUAL.md) — end-user guide
- [../README.md](../README.md) — quick start
