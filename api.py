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
from pdf_image_pipeline import (
    build_caption_chunks_for_sources,
    enrich_indexed_files_with_image_counts,
    process_pdf_images,
)
from pdf_storage import get_pdf_path
from utils import (
    extract_source_items,
    format_sources,
    is_multi_document_overview,
    is_conversation_recall_question,
    parse_page_reference,
)

SOURCE_COLORS = ("brand", "emerald2", "amber2")
STREAMLIT_URL = STREAMLIT_APP_URL


def _get_memory(session_id: str):
    """Lazy-load SQLite conversation memory for one session."""
    from rag_chain import get_memory

    return get_memory(session_id)


def _rag_chain():
    """Lazy-load RAG chain helpers."""
    from rag_chain import (
        answer_from_chat_history,
        answer_from_documents,
        build_rag_chain,
        query_chain,
    )

    return answer_from_chat_history, answer_from_documents, build_rag_chain, query_chain


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
    suggested_questions: List[str] = field(default_factory=list)
    suggestions_index_key: tuple = field(default_factory=tuple)

    def ensure_memory(self) -> Any:
        """Load or create SQLite-backed conversation memory for this session."""
        if self.memory is None:
            self.memory = _get_memory(self.session_id)
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
        _, build_rag_chain, _, _ = _rag_chain()
        self.ensure_llm_selection()
        self.indexed_files = (
            get_indexed_filenames(self.vector_store)
            if self.vector_store
            else []
        )
        if self.indexed_files:
            retriever = get_retriever(self.vector_store, self.session_id)
            self.chain = build_rag_chain(
                retriever,
                self.ensure_memory(),
                llm_provider=self.llm_provider,
                llm_model=self.llm_model,
            )
        else:
            self.chain = None
        refresh_session_suggestions(self)


def refresh_session_suggestions(session: AppSession) -> None:
    """Regenerate starter questions when the indexed file set changes."""
    current_key = tuple(sorted(session.indexed_files or []))
    if not current_key:
        session.suggested_questions = []
        session.suggestions_index_key = ()
        return
    if (
        session.suggestions_index_key == current_key
        and session.suggested_questions
    ):
        return
    from rag_chain import generate_suggested_questions

    session.suggested_questions = generate_suggested_questions(
        session.vector_store,
        list(session.indexed_files),
        llm_provider=session.llm_provider,
        llm_model=session.llm_model,
    )
    session.suggestions_index_key = current_key


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
    for filename in dict.fromkeys(filenames):
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
    answer_from_chat_history, answer_from_documents, _, query_chain = _rag_chain()

    ensure_session_vector_store(session)
    session.ensure_llm_selection()
    indexed_files = session.indexed_files or (
        get_indexed_filenames(session.vector_store)
        if session.vector_store
        else []
    )

    if is_conversation_recall_question(prompt):
        return answer_from_chat_history(
            prompt,
            session.ensure_memory(),
            llm_provider=session.llm_provider,
            llm_model=session.llm_model,
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
                session_id=session.session_id,
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
            session_id=session.session_id,
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
    from sqlite_memory import create_session as db_create_session, messages_to_api_format

    _, _, _, _, _, load_existing_vector_store, _ = _vector_store()
    sid = session_id or str(uuid.uuid4())
    db_create_session(sid)
    persist_dir = session_chroma_dir(sid)
    vector_store = load_existing_vector_store(persist_dir)
    default_llm = get_default_llm_option()
    session = AppSession(
        session_id=sid,
        vector_store=vector_store,
        llm_provider=default_llm["provider"],
        llm_model=default_llm["model"],
        messages=messages_to_api_format(sid),
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
    from sqlite_memory import create_session as db_create_session, messages_to_api_format

    if session_id:
        if session_id not in _sessions:
            _sessions[session_id] = create_session(session_id)
        session = _sessions[session_id]
        if not session.messages:
            session.messages = messages_to_api_format(session.session_id)
        return session
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
    files = enrich_indexed_files_with_image_counts(session.session_id, stats["files"])
    return {
        "session_id": session.session_id,
        "indexed_files": files,
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
        "suggested_questions": session.suggested_questions or list(EXAMPLE_QUESTIONS),
        "example_questions": session.suggested_questions or list(EXAMPLE_QUESTIONS),
        "streamlit_url": STREAMLIT_URL,
        "chat_ready": session.chain is not None,
    }


@app.post("/api/model")
def set_active_model(body: ModelSelectionRequest, session_id: Optional[str] = None):
    """Update the active provider/model for one API session."""
    session = get_session(session_id)
    available = get_available_llm_options()
    available_by_key = {
        (option["provider"], option["model"]): option for option in available
    }
    selected = available_by_key.get((body.provider, body.model))
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

    persist_api_uploads(new_files)
    image_summary = {"extracted": 0, "captioned": 0}
    for upload in new_files:
        pdf_path = get_pdf_path(upload.name)
        if pdf_path is None:
            continue
        image_stats = process_pdf_images(session.session_id, upload.name, pdf_path)
        image_summary["extracted"] += image_stats.get("extracted", 0)
        image_summary["captioned"] += image_stats.get("captioned", 0)

    documents, failed = load_buffered_pdfs(new_files)
    source_names = [upload.name for upload in new_files]

    if not documents:
        chunks = build_caption_chunks_for_sources(session.session_id, source_names)
        if not chunks:
            detail = "No readable text or image captions found in the uploaded PDF(s)."
            if failed:
                detail += f" Could not read: {', '.join(failed)}."
            raise HTTPException(status_code=400, detail=detail)
    else:
        chunks = split_documents(documents)
        chunks = [chunk for chunk in chunks if (chunk.page_content or "").strip()]
        if not chunks:
            chunks = build_caption_chunks_for_sources(session.session_id, source_names)

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="Could not index the uploaded PDF(s). Add selectable text or enable image captioning.",
        )

    persist_dir = session_chroma_dir(session.session_id)
    session.vector_store = create_or_update_vector_store(
        chunks,
        persist_dir=persist_dir,
        existing_store=session.vector_store,
    )
    indexed_names = {
        chunk.metadata.get("source")
        for chunk in chunks
        if chunk.metadata.get("source")
    }

    session.rebuild_chain()
    if not session.indexed_files or session.chain is None:
        raise HTTPException(
            status_code=500,
            detail="Indexing did not complete. Reset the session and try again.",
        )

    stats = index_stats(session.vector_store)
    indexed_files = enrich_indexed_files_with_image_counts(
        session.session_id, stats["files"]
    )

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
        "indexed_files": indexed_files,
        "image_summary": image_summary,
        "suggested_questions": session.suggested_questions,
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
    session.ensure_memory().clear()
    session.messages = []
    session.memory = None
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
    from image_store import delete_images_for_session
    from sqlite_memory import delete_session as db_delete_session

    session = get_session(session_id)
    ensure_session_vector_store(session)
    for filename in list(session.indexed_files):
        delete_pdf(filename)
    clear_vector_store(session.vector_store)
    session_dir = Path(session_chroma_dir(session.session_id))
    if session_dir.exists():
        shutil.rmtree(session_dir, ignore_errors=True)
    db_delete_session(session.session_id)
    delete_images_for_session(session.session_id)
    from sqlite_memory import create_session as db_create_session

    db_create_session(session.session_id)
    session.messages = []
    session.memory = None
    session.chain = None
    session.vector_store = None
    session.indexed_files = []
    session.suggested_questions = []
    session.suggestions_index_key = ()
    return {
        "message": "Session reset.",
        "messages": [],
        "indexed_files": [],
    }


@app.get("/api/images")
def list_pdf_images(
    session_id: Optional[str] = None,
    source: Optional[str] = None,
    page: Optional[int] = None,
):
    """List extracted PDF images and captions for one session."""
    from image_store import list_images

    session = get_session(session_id)
    images = list_images(session.session_id, source=source, page=page)
    return {"images": images}


@app.get("/api/images/{image_id}/file")
def get_pdf_image_file(image_id: str, session_id: Optional[str] = None):
    """Return the PNG/JPEG bytes for one extracted PDF image."""
    from fastapi.responses import FileResponse
    from image_store import get_image

    session = get_session(session_id)
    record = get_image(image_id)
    if record is None or record.get("session_id") != session.session_id:
        raise HTTPException(status_code=404, detail="Image not found.")

    path = Path(record["file_path"])
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Image file missing on disk.")
    return FileResponse(path)


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
