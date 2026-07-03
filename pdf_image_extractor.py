"""Extract embedded images from PDF files using PyMuPDF.

Images are written to disk under ``EXTRACTED_IMAGES_DIR`` and described by
metadata records for the SQLite manifest (not stored in Chroma).
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List

from config import EXTRACTED_IMAGES_DIR, IMAGE_MIN_HEIGHT, IMAGE_MIN_WIDTH

logger = logging.getLogger(__name__)


def _safe_source_stem(source: str) -> str:
    """Return a filesystem-safe directory name for one PDF source."""
    stem = Path(source).stem or "document"
    cleaned = re.sub(r"[^\w\-.]+", "_", stem).strip("._")
    return cleaned or "document"


def extract_images_from_pdf(
    pdf_path: Path,
    session_id: str,
    source: str,
) -> List[Dict[str, Any]]:
    """Extract embedded images from a PDF and save them as files on disk.

    Tiny icons and duplicate byte hashes within one PDF are skipped.

    Args:
        pdf_path: Path to the saved PDF file.
        session_id: Chat/API session identifier used for image namespacing.
        source: Original PDF filename (for metadata and folder layout).

    Returns:
        List of image metadata dicts ready for ``image_store.insert_images``.
    """
    import fitz

    sid = (session_id or "").strip()
    if not sid:
        raise ValueError("session_id is required for image extraction.")

    path = Path(pdf_path)
    if not path.is_file():
        logger.warning("PDF file not found for image extraction: %s", path)
        return []

    logger.info("Extracting images from PDF: %s (session=%s)", source, sid)
    doc = fitz.open(path)
    records: List[Dict[str, Any]] = []
    seen_hashes: set[str] = set()
    output_root = EXTRACTED_IMAGES_DIR / sid / _safe_source_stem(source)
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            for image_index, image_info in enumerate(page.get_images(full=True)):
                xref = image_info[0]
                try:
                    extracted = doc.extract_image(xref)
                except Exception:
                    continue

                width = int(extracted.get("width") or 0)
                height = int(extracted.get("height") or 0)
                if width < IMAGE_MIN_WIDTH or height < IMAGE_MIN_HEIGHT:
                    continue

                image_bytes = extracted.get("image") or b""
                if not image_bytes:
                    continue

                digest = hashlib.sha256(image_bytes).hexdigest()
                if digest in seen_hashes:
                    continue
                seen_hashes.add(digest)

                extension = (extracted.get("ext") or "png").lower()
                if extension == "jpeg":
                    extension = "jpg"
                file_name = f"p{page_num + 1}_img{image_index}.{extension}"
                file_path = output_root / file_name
                file_path.write_bytes(image_bytes)

                records.append(
                    {
                        "image_id": str(uuid.uuid4()),
                        "session_id": sid,
                        "source": source,
                        "page": page_num,
                        "page_label": str(page_num + 1),
                        "image_index": image_index,
                        "file_path": str(file_path.resolve()),
                        "width": width,
                        "height": height,
                        "bytes_sha256": digest,
                    }
                )
    finally:
        doc.close()

    logger.info("Extracted %d image(s) from %s (%d pages scanned)",
                len(records), source, len(doc) if hasattr(doc, '__len__') else 0)
    return records
