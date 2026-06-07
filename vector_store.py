"""Embedding generation, ChromaDB persistence, and retrieval.

Generates local HuggingFace embeddings for text chunks, stores them in a
persistent ChromaDB collection, and exposes a retriever for the RAG chain.

SRS references: FR-VS-01, FR-VS-02, FR-VS-03, FR-VS-04.

IMPORTANT (Windows native-library ordering):
    ``torch``/``sentence-transformers`` and ``chromadb`` (which loads
    ``onnxruntime``) ship conflicting native runtimes. If chromadb is imported
    *before* the torch embedding model is loaded, the process crashes with an
    access violation. To avoid this we (a) never import ``langchain_chroma`` at
    module load time, and (b) always call ``get_embeddings()`` (which loads the
    torch model) before the lazy chromadb import inside each function.
"""

import os
from typing import TYPE_CHECKING, List, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from config import (
    EMBEDDING_MODEL_NAME,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    TOP_K_RESULTS,
)
from citation_utils import ensure_page_label

if TYPE_CHECKING:  # for type hints only; not imported at runtime
    from langchain_chroma import Chroma

# Cache the embedding model so it is loaded from disk only once per process.
_embeddings: Optional[HuggingFaceEmbeddings] = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the HuggingFace embedding model (local, no API key required).

    The model (``all-MiniLM-L6-v2``, 384-dimensional) is downloaded on first
    use (~90 MB) and cached on disk thereafter. The loaded instance is cached
    in-process so repeated calls are cheap.

    This must be called before chromadb is imported (see module docstring), so
    every function that touches Chroma calls it first.

    Returns:
        A ready-to-use ``HuggingFaceEmbeddings`` instance.
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def create_or_update_vector_store(chunks: List[Document]) -> "Chroma":
    """Create a new ChromaDB store or add chunks to the existing one.

    If a persisted store already exists at ``CHROMA_PERSIST_DIR`` the chunks are
    appended to it; otherwise a new collection is created. Data is written to
    disk so it survives application restarts.

    Args:
        chunks: List of chunked ``Document`` objects to embed and store.

    Returns:
        The ``Chroma`` vector store instance containing the chunks.
    """
    embeddings = get_embeddings()  # load torch model before importing chromadb
    from langchain_chroma import Chroma

    if os.path.exists(CHROMA_PERSIST_DIR):
        vector_store = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        if chunks:
            vector_store.add_documents(chunks)
    else:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=CHROMA_COLLECTION_NAME,
            persist_directory=CHROMA_PERSIST_DIR,
        )

    return vector_store


def load_existing_vector_store() -> Optional["Chroma"]:
    """Load a persisted ChromaDB vector store from disk if one exists.

    Returns:
        A ``Chroma`` instance if ``CHROMA_PERSIST_DIR`` exists, otherwise
        ``None`` (signalling that no PDFs have been indexed yet).
    """
    if not os.path.exists(CHROMA_PERSIST_DIR):
        return None

    embeddings = get_embeddings()  # load torch model before importing chromadb
    from langchain_chroma import Chroma

    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    return vector_store


class _PageLabelRetriever(BaseRetriever):
    """Inject 1-based ``page_label`` metadata on every retrieved chunk."""

    def __init__(self, base_retriever: BaseRetriever):
        """Wrap an existing retriever."""
        super().__init__()
        self._base_retriever = base_retriever

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> List[Document]:
        """Retrieve documents and normalize page labels for citation display."""
        docs = self._base_retriever.invoke(query)
        return [ensure_page_label(doc) for doc in docs]


def get_retriever(vector_store: "Chroma") -> BaseRetriever:
    """Create a similarity-search retriever from the vector store.

    Args:
        vector_store: A ``Chroma`` vector store instance.

    Returns:
        A retriever configured to return the top-``TOP_K_RESULTS`` most similar
        chunks for a query.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS},
    )
    return _PageLabelRetriever(retriever)


def clear_vector_store(vector_store: "Chroma") -> bool:
    """Delete all indexed data from the vector store's collection.

    Used by the "Reset Session" control so the app returns to its initial state
    and the user must re-upload PDFs (SRS FR-MEM-04). Clearing the collection
    (rather than deleting the directory) avoids Windows file-lock errors while
    the store is still open.

    Args:
        vector_store: A ``Chroma`` vector store instance (may be ``None``).

    Returns:
        ``True`` if the collection was cleared, ``False`` otherwise.
    """
    if vector_store is None:
        return False
    try:
        vector_store.delete_collection()
        return True
    except Exception:
        return False


def delete_file(vector_store: "Chroma", filename: str) -> bool:
    """Delete all chunks belonging to a single source PDF from the store.

    Lets the user remove one document (via the sidebar 🗑 control) without
    resetting the entire knowledge base.

    Args:
        vector_store: A ``Chroma`` vector store instance (may be ``None``).
        filename: The ``source`` filename whose chunks should be removed.

    Returns:
        ``True`` if the delete call succeeded, ``False`` otherwise.
    """
    if vector_store is None:
        return False
    try:
        vector_store._collection.delete(where={"source": filename})
        return True
    except Exception:
        return False


def get_page_documents(
    vector_store: "Chroma", filename: str, page_number: int
) -> List[Document]:
    """Fetch all chunks for a specific 1-based page of a given source file.

    Enables page-targeted questions ("what is on page 7 of report.pdf?"), which
    semantic similarity search alone cannot answer reliably.

    Args:
        vector_store: A ``Chroma`` vector store instance (may be ``None``).
        filename: The source PDF filename to filter on.
        page_number: 1-based page number as a human would say it. Stored page
            metadata is 0-based, so this is matched against ``page_number - 1``.

    Returns:
        List of ``Document`` chunks on that page, ordered by start position;
        empty if none are found or on error.
    """
    if vector_store is None:
        return []
    try:
        results = vector_store._collection.get(
            where={
                "$and": [
                    {"source": {"$eq": filename}},
                    {"page": {"$eq": page_number - 1}},
                ]
            },
            include=["documents", "metadatas"],
        )
        docs = [
            Document(page_content=content, metadata=meta or {})
            for content, meta in zip(results["documents"], results["metadatas"])
        ]
        docs.sort(key=lambda d: d.metadata.get("start_index", 0))
        return docs
    except Exception:
        return []


def get_indexed_filenames(vector_store: "Chroma") -> List[str]:
    """Return the unique source filenames already indexed in the store.

    Used to display the indexed-document list in the UI and to skip
    re-processing of duplicate uploads.

    Args:
        vector_store: A ``Chroma`` vector store instance.

    Returns:
        A list of unique ``source`` filenames, or an empty list if the store is
        empty or cannot be read.
    """
    try:
        collection = vector_store._collection
        results = collection.get(include=["metadatas"])
        sources = set()
        for meta in results["metadatas"]:
            if meta and "source" in meta:
                sources.add(meta["source"])
        return list(sources)
    except Exception:
        return []
