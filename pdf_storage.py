"""Persist uploaded PDF files on disk for source preview in the Streamlit UI."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from config import UPLOADED_PDF_DIR


def ensure_storage_dir() -> Path:
    """Create the uploaded-PDF directory if it does not exist."""
    UPLOADED_PDF_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOADED_PDF_DIR


def save_uploaded_pdf(filename: str, file_bytes: bytes) -> Path:
    """Write an uploaded PDF to disk, keyed by its original filename.

    Args:
        filename: Original upload name (e.g. ``report.pdf``).
        file_bytes: Raw PDF bytes.

    Returns:
        Path to the saved file.
    """
    ensure_storage_dir()
    path = UPLOADED_PDF_DIR / filename
    path.write_bytes(file_bytes)
    return path


def get_pdf_path(filename: str) -> Optional[Path]:
    """Return the on-disk path for an indexed PDF, if it exists."""
    path = UPLOADED_PDF_DIR / filename
    return path if path.is_file() else None


def delete_pdf(filename: str) -> None:
    """Remove a single persisted PDF from disk."""
    path = UPLOADED_PDF_DIR / filename
    if path.is_file():
        path.unlink()


def clear_all_pdfs() -> None:
    """Delete every persisted PDF (used by Reset All)."""
    if UPLOADED_PDF_DIR.exists():
        shutil.rmtree(UPLOADED_PDF_DIR)
    ensure_storage_dir()
