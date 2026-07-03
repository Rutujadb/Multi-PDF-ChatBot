"""Inject extracted image captions into RAG context without vector image storage."""

from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from config import IMAGE_EXTRACTION_ENABLED
from image_store import get_images_for_page

logger = logging.getLogger(__name__)


def build_image_context_block(
    session_id: str,
    source: str,
    page: int,
    page_label: Optional[str] = None,
) -> str:
    """Build a text block describing images on one PDF page."""
    if not IMAGE_EXTRACTION_ENABLED or not session_id:
        return ""

    images = get_images_for_page(session_id, source, page)
    if not images:
        return ""

    label = page_label or str(page + 1)
    lines = [f"[Images on {source}, page {label}]"]
    for image in images:
        caption = (image.get("caption") or "").strip()
        if caption:
            lines.append(f"- Image {int(image.get('image_index', 0)) + 1}: {caption}")
        else:
            lines.append(
                f"- Image {int(image.get('image_index', 0)) + 1}: "
                "(embedded figure; caption unavailable)"
            )
    return "\n".join(lines)


def enrich_documents_with_image_context(
    documents: List[Document],
    session_id: Optional[str],
) -> List[Document]:
    """Append page-level image caption text to retrieved document chunks."""
    if not IMAGE_EXTRACTION_ENABLED or not session_id or not documents:
        return documents

    logger.debug("Enriching %d documents with image context (session=%s)",
                 len(documents), session_id)
    cache: dict[tuple[str, int], str] = {}
    enriched: List[Document] = []
    for doc in documents:
        meta = doc.metadata or {}
        source = meta.get("source")
        page = meta.get("page")
        suffix = ""
        if source and isinstance(page, int):
            key = (source, page)
            if key not in cache:
                cache[key] = build_image_context_block(
                    session_id,
                    source,
                    page,
                    page_label=str(meta.get("page_label", page + 1)),
                )
            block = cache[key]
            if block:
                suffix = "\n\n" + block
        enriched.append(
            Document(
                page_content=(doc.page_content or "") + suffix,
                metadata=dict(meta),
            )
        )
    return enriched


class ImageEnrichingRetriever(BaseRetriever):
    """Wrap a retriever and append image captions to retrieved chunks."""

    def __init__(self, base_retriever: BaseRetriever, session_id: str):
        """Store the wrapped retriever and session scope."""
        super().__init__()
        self._base_retriever = base_retriever
        self._session_id = session_id

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> List[Document]:
        """Retrieve documents and enrich them with image caption context."""
        logger.debug("ImageEnrichingRetriever: retrieving for query '%s'", query[:80])
        docs = self._base_retriever.invoke(query)
        logger.debug("Base retriever returned %d docs; enriching with image context", len(docs))
        return enrich_documents_with_image_context(docs, self._session_id)
