"""Orchestrate PDF image extraction, manifest storage, and Gemma captioning."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from config import IMAGE_CAPTION_ENABLED, IMAGE_EXTRACTION_ENABLED
from image_captioner import caption_image
from image_store import insert_images, list_images, update_caption
from pdf_image_extractor import extract_images_from_pdf
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _is_valid_caption(caption: str) -> bool:
    """Return True when a caption looks like real image description text."""
    text = (caption or "").strip()
    if len(text) < 10:
        return False
    lowered = text.lower()
    failure_markers = (
        "rate limit",
        "api key",
        "unauthorized",
        "429",
        "error:",
        "failed to",
    )
    return not any(marker in lowered for marker in failure_markers)


def process_pdf_images(
    session_id: str,
    source: str,
    pdf_path: Path,
) -> Dict[str, Any]:
    """Extract images from one PDF, store references, and optionally caption them.

    Image bytes and captions are kept outside Chroma. Failures are logged and
    do not block text indexing.

    Args:
        session_id: Session identifier for namespacing image records.
        source: Original PDF filename.
        pdf_path: On-disk path to the saved PDF.

    Returns:
        Dict with ``extracted``, ``stored``, and ``captioned`` counts.
    """
    if not IMAGE_EXTRACTION_ENABLED:
        logger.info("Image extraction disabled; skipping %s", source)
        return {"extracted": 0, "stored": 0, "captioned": 0}

    logger.info("Starting image pipeline for %s (session=%s)", source, session_id)
    try:
        records = extract_images_from_pdf(pdf_path, session_id, source)
        stored_rows = insert_images(records)
        logger.info("Extracted %d, stored %d images for %s",
                     len(records), len(stored_rows), source)
    except Exception as exc:
        logger.error("Image extraction failed for %s: %s", source, exc, exc_info=True)
        return {"extracted": 0, "stored": 0, "captioned": 0, "error": str(exc)}

    captioned = 0
    if IMAGE_CAPTION_ENABLED:
        logger.info("Captioning %d image(s) for %s", len(stored_rows), source)
        for row in stored_rows:
            if row.get("caption"):
                continue
            caption, caption_model = caption_image(
                Path(row["file_path"]),
                source=row.get("source", source),
                page_label=str(row.get("page_label", "")),
            )
            if _is_valid_caption(caption):
                update_caption(row["image_id"], caption, caption_model)
                captioned += 1

    logger.info("Image pipeline done for %s: extracted=%d, stored=%d, captioned=%d",
                source, len(records), len(stored_rows), captioned)
    return {
        "extracted": len(records),
        "stored": len(stored_rows),
        "captioned": captioned,
    }


def enrich_indexed_files_with_image_counts(
    session_id: str,
    files: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Attach per-file extracted image counts to API index stats."""
    from image_store import count_images_by_source

    counts = count_images_by_source(session_id)
    enriched: List[Dict[str, Any]] = []
    for item in files:
        copy = dict(item)
        copy["images"] = counts.get(item.get("name", ""), 0)
        enriched.append(copy)
    return enriched


def build_caption_chunks_for_sources(
    session_id: str,
    sources: List[str],
) -> List[Document]:
    """Build text chunks from stored image captions when PDF text is empty.

    Image-heavy or scanned PDFs may yield no text chunks after splitting.
    This fallback lets RAG answer using Gemma-generated image descriptions.

    Args:
        session_id: Session identifier for the image manifest.
        sources: PDF filenames that were just processed.

    Returns:
        List of page-level ``Document`` objects derived from image captions.
    """
    sid = (session_id or "").strip()
    if not sid or not sources:
        return []

    chunks: List[Document] = []
    for source in sources:
        images = list_images(sid, source=source)
        if not images:
            continue

        by_page: Dict[int, List[Dict[str, Any]]] = {}
        for image in images:
            page = int(image.get("page", 0))
            by_page.setdefault(page, []).append(image)

        for page in sorted(by_page):
            lines: List[str] = []
            for image in sorted(by_page[page], key=lambda row: row.get("image_index", 0)):
                caption = (image.get("caption") or "").strip()
                if caption:
                    lines.append(caption)
                else:
                    lines.append(
                        "This page contains an embedded image that could not be "
                        "captioned automatically."
                    )
            if not lines:
                continue
            chunks.append(
                Document(
                    page_content="\n".join(lines),
                    metadata={
                        "source": source,
                        "page": page,
                        "page_label": str(page + 1),
                        "from_image_captions": True,
                    },
                )
            )
    return chunks
