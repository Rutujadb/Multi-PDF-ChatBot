"""PDF ingestion pipeline: load, extract text, chunk, and detect duplicates.

This module turns Streamlit ``UploadedFile`` objects into chunked LangChain
``Document`` objects tagged with source metadata, ready for embedding.

SRS references: FR-PDF-01, FR-PDF-02, FR-PDF-03, FR-PDF-04, FR-PDF-05.
"""

import os
import tempfile
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP


def load_pdfs(uploaded_files) -> Tuple[List[Document], List[str]]:
    """Load and extract text from uploaded Streamlit PDF files.

    Each uploaded file is written to a temporary file (PyPDFLoader requires a
    file path), loaded page by page, tagged with its original filename, and
    then the temporary file is removed. Files that cannot be read are recorded
    and skipped so a single bad PDF never crashes the pipeline.

    Args:
        uploaded_files: List of Streamlit ``UploadedFile`` objects (each exposes
            a ``.name`` attribute and a ``.read()`` method returning bytes).

    Returns:
        Tuple of:
            - List of ``Document`` objects (one per page) with ``source`` and
              ``page`` metadata populated.
            - List of filenames that failed to load.
    """
    all_documents: List[Document] = []
    failed_files: List[str] = []

    for uploaded_file in uploaded_files:
        tmp_path = None
        try:
            # PyPDFLoader needs a real file path, so persist the upload to a
            # temporary file first.
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".pdf"
            ) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            documents = loader.load()

            # Tag every page with the original filename for source citation.
            for doc in documents:
                doc.metadata["source"] = uploaded_file.name

            all_documents.extend(documents)
        except Exception:
            failed_files.append(uploaded_file.name)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return all_documents, failed_files


def split_documents(documents: List[Document]) -> List[Document]:
    """Split page-level documents into smaller overlapping chunks.

    Uses ``RecursiveCharacterTextSplitter`` so splits prefer natural
    boundaries (paragraphs, then sentences, then words) and avoid breaking
    mid-sentence where possible. Source and page metadata are preserved, and
    each chunk is additionally tagged with the line number (within its page)
    where the chunk begins, for richer source citations.

    Args:
        documents: List of full-page ``Document`` objects from ``load_pdfs``.

    Returns:
        List of chunked ``Document`` objects with ``source``, ``page``, and
        ``line`` metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
        add_start_index=True,
    )

    all_chunks: List[Document] = []
    for document in documents:
        # Split one page at a time so each chunk's ``start_index`` is an offset
        # within that page's text, letting us derive a per-page line number.
        for chunk in splitter.split_documents([document]):
            start = chunk.metadata.get("start_index", 0)
            chunk.metadata["line"] = document.page_content[:start].count("\n") + 1
            page = chunk.metadata.get("page")
            if isinstance(page, int):
                chunk.metadata["page_label"] = page + 1
            all_chunks.append(chunk)

    return all_chunks


def filter_new_files(uploaded_files, already_indexed: List[str]):
    """Separate uploaded files into new ones and already-indexed duplicates.

    Detection is based on a filename match against the list of sources already
    present in the vector store, so a previously processed PDF is not embedded
    a second time.

    Args:
        uploaded_files: List of Streamlit ``UploadedFile`` objects.
        already_indexed: List of filenames already present in the vector store.

    Returns:
        Tuple of (new_files, skipped_filenames):
            - new_files: file objects whose names are not yet indexed.
            - skipped_filenames: names of files already indexed.
    """
    new_files = []
    skipped: List[str] = []
    for f in uploaded_files:
        if f.name in already_indexed:
            skipped.append(f.name)
        else:
            new_files.append(f)
    return new_files, skipped
