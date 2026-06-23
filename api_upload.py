"""FastAPI upload helpers for the React dashboard.

These utilities adapt Starlette ``UploadFile`` objects into in-memory uploads
that match the Streamlit ``UploadedFile`` interface (``.name``, ``.read()``,
``.size``) so the shared RAG pipeline can run unchanged for Streamlit.
"""

from __future__ import annotations

import os
import tempfile
from typing import List, Tuple

from fastapi import UploadFile
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


class BufferedPdfUpload:
    """In-memory PDF upload with the same interface Streamlit uploads expose."""

    def __init__(self, filename: str, content: bytes):
        """Store filename and raw PDF bytes for the ingestion pipeline."""
        self.name = filename
        self.size = len(content)
        self._content = content

    def read(self) -> bytes:
        """Return the buffered PDF bytes."""
        return self._content

    def seek(self, pos: int = 0) -> None:
        """No-op seek for compatibility with Streamlit upload objects."""
        return None


async def buffer_fastapi_uploads(upload_files: List[UploadFile]) -> List[BufferedPdfUpload]:
    """Read FastAPI ``UploadFile`` objects into buffered pipeline uploads."""
    buffered: List[BufferedPdfUpload] = []
    for upload in upload_files:
        content = await upload.read()
        filename = upload.filename or "upload.pdf"
        buffered.append(BufferedPdfUpload(filename, content))
    return buffered


def persist_api_uploads(uploaded_files: List[BufferedPdfUpload]) -> None:
    """Save buffered API uploads to disk for source preview in the React UI."""
    from pdf_storage import save_uploaded_pdf

    for upload in uploaded_files:
        save_uploaded_pdf(upload.name, upload.read())


def validate_api_pdf_files(
    uploaded_files: List[BufferedPdfUpload],
) -> Tuple[List[BufferedPdfUpload], List[str]]:
    """Split API uploads into valid PDFs and invalid filenames."""
    valid: List[BufferedPdfUpload] = []
    invalid: List[str] = []
    for upload in uploaded_files:
        if upload.name.lower().endswith(".pdf") and upload.size > 0:
            valid.append(upload)
        else:
            invalid.append(upload.name)
    return valid, invalid


def filter_new_api_files(
    uploaded_files: List[BufferedPdfUpload],
    already_indexed: List[str],
) -> Tuple[List[BufferedPdfUpload], List[str]]:
    """Return API uploads that are not already indexed."""
    new_files: List[BufferedPdfUpload] = []
    skipped: List[str] = []
    seen_names = set(already_indexed)
    for upload in uploaded_files:
        if upload.name in seen_names:
            skipped.append(upload.name)
        else:
            new_files.append(upload)
            seen_names.add(upload.name)
    return new_files, skipped


def load_buffered_pdfs(
    uploaded_files: List[BufferedPdfUpload],
) -> Tuple[List[Document], List[str]]:
    """Load and extract text from buffered API PDF uploads."""
    all_documents: List[Document] = []
    failed_files: List[str] = []

    for uploaded_file in uploaded_files:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".pdf"
            ) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            documents = loader.load()

            for doc in documents:
                doc.metadata["source"] = uploaded_file.name

            all_documents.extend(documents)
        except Exception:
            failed_files.append(uploaded_file.name)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return all_documents, failed_files
