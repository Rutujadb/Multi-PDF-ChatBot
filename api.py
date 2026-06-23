"""FastAPI backend for the React dashboard.

Wraps the existing PDF, vector-store, and RAG modules so the React UI can
upload documents, chat, and manage session state. Streamlit remains available
via ``streamlit run app.py`` as an alternate UI on port 8501.

SRS references: FR-UI-01 → FR-UI-07, FR-PDF-01, FR-MEM-01 → FR-MEM-04.
"""

from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import (
    APP_NAME,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL_NAME,
    EXAMPLE_QUESTIONS,
    LLM_MAX_TOKENS,
    LLM_PROVIDER,
    LLM_REPETITION_PENALTY,
    LLM_FREQUENCY_PENALTY,
    LLM_TOP_K,
    LLM_TOP_P,
    LLM_TEMPERATURE,
    STREAMLIT_APP_URL,
    TOP_K_RESULTS,
    VECTOR_STORE,
    get_available_llm_options,
    get_default_llm_option,
    get_active_llm_name,
    get_cors_origins,
)
from api_source_preview import SourcePreviewRequest
from api_upload import (
    buffer_fastapi_uploads,
    filter_new_api_files,
    load_buffered_pdfs,
    persist_api_uploads,
    validate_api_pdf_files,
)
from pdf_processor import split_documents
from utils import (
    extract_source_items,
    format_sources,
    is_multi_document_overview,
    parse_page_reference,
)

SOURCE_COLORS = ("brand", "emerald2", "amber2")
STREAMLIT_URL = STREAMLIT_APP_URL


def _get_memory():
    """Lazy-load conversation memory to keep API startup fast on Render."""
    from rag_chain import get_memory

    return get_memory()


def _rag_chain():
    """Lazy-load RAG chain helpers."""
    from rag_chain import answer_from_documents, build_rag_chain, query_chain

    return answer_from_documents, build_rag_chain, query_chain


def _vector_store():
    """Lazy-load vector store helpers."""
    from vector_store import (
        clear_vector_store,
        create_or_update_vector_store,
        get_indexed_filenames,
        get_page_documents,
        get_retriever,
        load_existing_vector_store,
        retrieve_balanced_documents,
    )

    return (
        clear_vector_store,
        create_or_update_vector_store,
        get_indexed_filenames,
        get_page_documents,
        get_retriever,
        load_existing_vector_store,
        retrieve_balanced_documents,
    )


def session_chroma_dir(session_id: str) -> str:
    """Return the per-session Chroma persist path for the React API."""
    return str(Path(CHROMA_PERSIST_DIR) / "api_sessions" / session_id)


def _normalize_upload_files(
    files: Union[UploadFile, List[UploadFile], None],
) -> List[UploadFile]:
    """Coerce FastAPI upload input into a list of ``UploadFile`` objects."""
    if files is None:
        return []
    if isinstance(files, list):
        return files
    return [files]


@dataclass
class AppSession:
    """In-memory session state for a single browser client."""

    session_id: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    memory: Any = None
    chain: Any = None
    vector_store: Any = None
    indexed_files: List[str] = field(default_factory=list)
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None

    def ensure_memory(self) -> Any:
        """Create conversation memory on first use."""
        if self.memory is None:
            self.memory = _get_memory()
        return self.memory

    def ensure_llm_selection(self) -> None:
        """Ensure the session has a valid provider/model selection."""
        if self.llm_provider and self.llm_model:
            return
        default_llm = get_default_llm_option()
        self.llm_provider = default_llm["provider"]
        self.llm_model = default_llm["model"]

    def rebuild_chain(self) -> None:
        """Rebuild the RAG chain from the current vector store."""
        (
            _clear,
            _create,
            get_indexed_filenames,
            _get_page,
            get_retriever,
            _load,
            _retrieve,
        ) = _vector_store()
        _, build_rag_chain, _ = _rag_chain()
        self.ensure_llm_selection()
        self.indexed_files = (
            get_indexed_filenames(self.vector_store)
            if self.vector_store
            else []
        )
        if self.indexed_files:
            retriever = get_retriever(self.vector_store)
            self.chain = build_rag_chain(
                retriever,
                self.ensure_memory(),
                llm_provider=self.llm_provider,
                llm_model=self.llm_model,
            )
        else:
            self.chain = None


class ChatRequest(BaseModel):
    """Payload for a chat message."""

    message: str


class ModelSelectionRequest(BaseModel):
    """Payload for selecting the active provider/model in one session."""

    provider: str
    model: str


def _page_label(metadata: dict) -> str:
    """Return a human-friendly page label from chunk metadata."""
    page = metadata.get("page")
    if isinstance(page, int):
        return str(page + 1)
    return str(metadata.get("page_label", "?"))


def structured_sources(source_documents: List[Any]) -> List[Dict[str, Any]]:
    """Convert retrieved documents into UI-friendly source chips."""
    seen = set()
    sources: List[Dict[str, Any]] = []
    for doc in source_documents or []:
        meta = doc.metadata or {}
        filename = meta.get("source", "Unknown")
        page_raw = _page_label(meta)
        page = int(page_raw) if page_raw.isdigit() else page_raw
        key = (filename, page_raw)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "file": filename,
                "page": page,
                "color": SOURCE_COLORS[len(sources) % len(SOURCE_COLORS)],
            }
        )
    return sources


def index_stats(vector_store: Any) -> Dict[str, Any]:
    """Compute per-file and aggregate index statistics."""
    if vector_store is None:
        return {"files": [], "total_chunks": 0, "total_pages": 0}

    try:
        results = vector_store._collection.get(include=["metadatas"])
        chunk_counts: Dict[str, int] = {}
        pages_by_file: Dict[str, set] = {}
        for meta in results["metadatas"]:
            if not meta or "source" not in meta:
                continue
            source = meta["source"]
            chunk_counts[source] = chunk_counts.get(source, 0) + 1
            page = meta.get("page")
            if isinstance(page, int):
                pages_by_file.setdefault(source, set()).add(page)

        files = [
            {
                "name": name,
                "pages": len(pages_by_file.get(name, set())),
                "chunks": chunk_counts[name],
            }
            for name in sorted(chunk_counts)
        ]
        return {
            "files": files,
            "total_chunks": sum(chunk_counts.values()),
            "total_pages": sum(len(pages) for pages in pages_by_file.values()),
        }
    except Exception:
        return {"files": [], "total_chunks": 0, "total_pages": 0}


def existing_doc_references(
    indexed_files: List[Dict[str, Any]],
    filenames: List[str],
) -> List[Dict[str, Any]]:
    """Return indexed-file references matching the given filenames."""
    by_name = {item.get("name"): item for item in indexed_files}
    references: List[Dict[str, Any]] = []
    for filename in filenames:
        item = by_name.get(filename)
        if item:
            references.append(item)
        else:
            references.append(
                {
                    "name": filename,
                    "pages": 0,
                    "chunks": 0,
                }
            )
    return references


def answer_question(session: AppSession, prompt: str) -> Dict[str, Any]:
    """Route a question through page-targeted or normal RAG retrieval."""
    (
        _clear,
        _create,
        get_indexed_filenames,
        get_page_documents,
        _get_retriever,
        _load,
        retrieve_balanced_documents,
    ) = _vector_store()
    answer_from_documents, _, query_chain = _rag_chain()

    ensure_session_vector_store(session)
    session.ensure_llm_selection()
    indexed_files = session.indexed_files or (
        get_indexed_filenames(session.vector_store)
        if session.vector_store
        else []
    )

    ref_file, ref_page = parse_page_reference(prompt, indexed_files)
    if ref_file and ref_page:
        page_docs = get_page_documents(session.vector_store, ref_file, ref_page)
        if page_docs:
            result = answer_from_documents(
                prompt,
                page_docs,
                vector_store=session.vector_store,
                llm_provider=session.llm_provider,
                llm_model=session.llm_model,
            )
            try:
                session.ensure_memory().add_user_message(prompt)
                session.ensure_memory().add_ai_message(result["answer"])
            except Exception:
                pass
            return result

    if session.chain is None:
        raise HTTPException(
            status_code=400,
            detail="Upload and process PDFs before chatting.",
        )

    if is_multi_document_overview(prompt, len(indexed_files)):
        overview_docs = retrieve_balanced_documents(
            session.vector_store,
            prompt,
            per_file_k=4,
            global_k=4,
        )
        result = answer_from_documents(
            prompt,
            overview_docs,
            vector_store=session.vector_store,
            llm_provider=session.llm_provider,
            llm_model=session.llm_model,
        )
        try:
            session.ensure_memory().add_user_message(prompt)
            session.ensure_memory().add_ai_message(result["answer"])
        except Exception:
            pass
        return result

    return query_chain(
        session.chain,
        prompt,
        session.vector_store,
        chat_history=session.ensure_memory(),
    )


def create_session(session_id: Optional[str] = None) -> AppSession:
    """Create or restore a session, loading its persisted vector store."""
    _, _, _, _, _, load_existing_vector_store, _ = _vector_store()
    sid = session_id or str(uuid.uuid4())
    persist_dir = session_chroma_dir(sid)
    vector_store = load_existing_vector_store(persist_dir)
    default_llm = get_default_llm_option()
    session = AppSession(
        session_id=sid,
        vector_store=vector_store,
        llm_provider=default_llm["provider"],
        llm_model=default_llm["model"],
    )
    if vector_store is not None:
        session.rebuild_chain()
    return session


def ensure_session_vector_store(session: AppSession) -> None:
    """Load the session vector store from disk when it is not in memory."""
    _, _, _, _, _, load_existing_vector_store, _ = _vector_store()
    if session.vector_store is None:
        session.vector_store = load_existing_vector_store(
            session_chroma_dir(session.session_id)
        )
        if session.vector_store is not None:
            session.rebuild_chain()


app = FastAPI(title=f"{APP_NAME} API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_sessions: Dict[str, AppSession] = {}
_default_session: Optional[AppSession] = None


def get_or_create_default_session() -> AppSession:
    """Return the singleton default session, creating it on first use."""
    global _default_session
    if _default_session is None:
        _default_session = create_session()
        _sessions[_default_session.session_id] = _default_session
    return _default_session


def get_session(session_id: Optional[str] = None) -> AppSession:
    """Return an existing session, restoring it from disk when needed."""
    if session_id:
        if session_id not in _sessions:
            _sessions[session_id] = create_session(session_id)
        return _sessions[session_id]
    return get_or_create_default_session()


@app.get("/api/health")
def health_check():
    """Simple health probe for the API process."""
    return {"status": "ok", "app": APP_NAME}


@app.post("/api/session")
def create_api_session():
    """Start a fresh API session (used by the React dashboard on load)."""
    session = create_session()
    _sessions[session.session_id] = session
    return {"session_id": session.session_id}


@app.get("/api/status")
def get_status(session_id: Optional[str] = None):
    """Return session, index, and configuration details for the dashboard."""
    session = get_session(session_id)
    ensure_session_vector_store(session)
    session.ensure_llm_selection()
    stats = index_stats(session.vector_store)
    return {
        "session_id": session.session_id,
        "indexed_files": stats["files"],
        "stats": {
            "chunks": stats["total_chunks"],
            "pages": stats["total_pages"],
            "dims": 384,
            "top_k": TOP_K_RESULTS,
        },
        "config": {
            "llm": session.llm_model or get_active_llm_name(),
            "provider": session.llm_provider or LLM_PROVIDER,
            "store": VECTOR_STORE,
            "embeddings": EMBEDDING_MODEL_NAME,
            "temperature": LLM_TEMPERATURE,
            "top_p": LLM_TOP_P,
            "top_k": LLM_TOP_K,
            "repetition_penalty": LLM_REPETITION_PENALTY,
            "frequency_penalty": LLM_FREQUENCY_PENALTY,
            "max_tokens": LLM_MAX_TOKENS,
        },
        "messages": session.messages,
        "available_models": get_available_llm_options(),
        "selected_model": {
            "provider": session.llm_provider,
            "model": session.llm_model,
        },
        "example_questions": EXAMPLE_QUESTIONS,
        "streamlit_url": STREAMLIT_URL,
        "chat_ready": session.chain is not None,
    }


@app.post("/api/model")
def set_active_model(body: ModelSelectionRequest, session_id: Optional[str] = None):
    """Update the active provider/model for one API session."""
    session = get_session(session_id)
    available = get_available_llm_options()
    selected = next(
        (
            option
            for option in available
            if option["provider"] == body.provider and option["model"] == body.model
        ),
        None,
    )
    if selected is None:
        raise HTTPException(status_code=400, detail="Selected model is not available.")

    session.llm_provider = selected["provider"]
    session.llm_model = selected["model"]
    if session.vector_store is not None:
        session.rebuild_chain()

    return {
        "message": "Model updated.",
        "selected_model": {
            "provider": session.llm_provider,
            "model": session.llm_model,
        },
    }


@app.post("/api/upload")
async def upload_pdfs(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = None,
):
    """Validate, embed, and index uploaded PDF files."""
    session = get_session(session_id)
    upload_files = _normalize_upload_files(files)
    if not upload_files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    buffered_files = await buffer_fastapi_uploads(upload_files)
    valid_files, invalid_files = validate_api_pdf_files(buffered_files)
    ensure_session_vector_store(session)
    (
        _clear,
        create_or_update_vector_store,
        get_indexed_filenames,
        _get_page,
        _get_retriever,
        _load,
        _retrieve,
    ) = _vector_store()
    already_indexed = (
        get_indexed_filenames(session.vector_store)
        if session.vector_store
        else []
    )
    new_files, skipped = filter_new_api_files(valid_files, already_indexed)
    stats = index_stats(session.vector_store)

    if skipped:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "you cannot add same file twice",
                "existing_references": existing_doc_references(
                    stats["files"], skipped
                ),
            },
        )

    if not new_files:
        return {
            "message": "No new PDFs to process.",
            "processed": 0,
            "skipped": skipped,
            "invalid": invalid_files,
            "failed": [],
            "indexed_files": stats["files"],
        }

    documents, failed = load_buffered_pdfs(new_files)
    if not documents:
        detail = "No readable text found in the uploaded PDF(s)."
        if failed:
            detail += f" Could not read: {', '.join(failed)}."
        raise HTTPException(status_code=400, detail=detail)

    chunks = split_documents(documents)
    persist_dir = session_chroma_dir(session.session_id)
    session.vector_store = create_or_update_vector_store(
        chunks,
        persist_dir=persist_dir,
        existing_store=session.vector_store,
    )
    indexed_names = {
        doc.metadata.get("source")
        for doc in documents
        if doc.metadata.get("source")
    }
    persist_api_uploads(
        [upload for upload in new_files if upload.name in indexed_names]
    )
    session.rebuild_chain()
    stats = index_stats(session.vector_store)

    indexed_count = len(indexed_names)
    message = f"{indexed_count} PDF(s) indexed."
    if failed:
        message += f" Could not read: {', '.join(failed)}."
    if invalid_files:
        message += f" Invalid files skipped: {', '.join(invalid_files)}."

    return {
        "message": message,
        "processed": indexed_count,
        "skipped": skipped,
        "invalid": invalid_files,
        "failed": failed,
        "indexed_files": stats["files"],
    }


@app.post("/api/chat")
def chat(payload: ChatRequest, session_id: Optional[str] = None):
    """Ask a question against the indexed knowledge base."""
    session = get_session(session_id)
    ensure_session_vector_store(session)
    prompt = (payload.message or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    session.messages.append({"role": "user", "text": prompt})
    result = answer_question(session, prompt)
    answer = result["answer"]
    source_items = extract_source_items(
        result.get("source_documents", []), answer=answer
    )
    sources = [
        {**item, "color": SOURCE_COLORS[index % len(SOURCE_COLORS)]}
        for index, item in enumerate(source_items)
    ]
    sources_text = format_sources(result.get("source_documents", []))

    assistant_message = {
        "role": "assistant",
        "text": answer,
        "sources": sources,
        "sources_text": sources_text,
    }
    session.messages.append(assistant_message)

    return {
        "answer": answer,
        "sources": sources,
        "sources_text": sources_text,
        "messages": session.messages,
    }


@app.post("/api/clear-chat")
def clear_chat(session_id: Optional[str] = None):
    """Clear chat history while keeping indexed PDFs."""
    session = get_session(session_id)
    session.messages = []
    session.memory = _get_memory()
    return {"message": "Chat cleared.", "messages": []}


@app.post("/api/reset")
def reset_session(session_id: Optional[str] = None):
    """Clear chat history and wipe the indexed knowledge base."""
    (
        clear_vector_store,
        _create,
        get_indexed_filenames,
        _get_page,
        _get_retriever,
        _load,
        _retrieve,
    ) = _vector_store()
    from pdf_storage import delete_pdf

    session = get_session(session_id)
    ensure_session_vector_store(session)
    for filename in list(session.indexed_files):
        delete_pdf(filename)
    clear_vector_store(session.vector_store)
    session_dir = Path(session_chroma_dir(session.session_id))
    if session_dir.exists():
        shutil.rmtree(session_dir, ignore_errors=True)
    session.messages = []
    session.memory = _get_memory()
    session.chain = None
    session.vector_store = None
    session.indexed_files = []
    return {
        "message": "Session reset.",
        "messages": [],
        "indexed_files": [],
    }


@app.post("/api/source/preview")
def source_preview(body: SourcePreviewRequest):
    """Return a PNG preview of a cited PDF page with highlights."""
    from api_source_preview import render_source_preview_image

    return render_source_preview_image(body)


@app.post("/api/source/download")
def source_download(body: SourcePreviewRequest):
    """Download the highlighted single-page PDF for a citation."""
    from api_source_preview import render_source_download

    return render_source_download(body)
