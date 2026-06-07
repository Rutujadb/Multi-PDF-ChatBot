"""FastAPI backend for the React dashboard.

Wraps the existing PDF, vector-store, and RAG modules so the React UI can
upload documents, chat, and manage session state. Streamlit remains available
via ``streamlit run app.py`` as an alternate UI on port 8501.

SRS references: FR-UI-01 → FR-UI-07, FR-PDF-01, FR-MEM-01 → FR-MEM-04.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import (
    APP_NAME,
    EMBEDDING_MODEL_NAME,
    EXAMPLE_QUESTIONS,
    GEMINI_MODEL_NAME,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    TOP_K_RESULTS,
    VECTOR_STORE,
)
from pdf_processor import filter_new_files, load_pdfs, split_documents
from rag_chain import (
    answer_from_documents,
    build_rag_chain,
    get_memory,
    query_chain,
)
from utils import format_sources, parse_page_reference, validate_pdf_files
from vector_store import (
    clear_vector_store,
    create_or_update_vector_store,
    get_indexed_filenames,
    get_page_documents,
    get_retriever,
    load_existing_vector_store,
)

SOURCE_COLORS = ("brand", "emerald2", "amber2")
STREAMLIT_URL = "http://localhost:8501"


@dataclass
class AppSession:
    """In-memory session state for a single browser client."""

    session_id: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    memory: Any = field(default_factory=get_memory)
    chain: Any = None
    vector_store: Any = None
    indexed_files: List[str] = field(default_factory=list)

    def rebuild_chain(self) -> None:
        """Rebuild the RAG chain from the current vector store."""
        self.indexed_files = (
            get_indexed_filenames(self.vector_store)
            if self.vector_store
            else []
        )
        if self.indexed_files:
            retriever = get_retriever(self.vector_store)
            self.chain = build_rag_chain(retriever, self.memory)
        else:
            self.chain = None


class ChatRequest(BaseModel):
    """Payload for a chat message."""

    message: str


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


def answer_question(session: AppSession, prompt: str) -> Dict[str, Any]:
    """Route a question through page-targeted or normal RAG retrieval."""
    ref_file, ref_page = parse_page_reference(prompt, session.indexed_files)
    if ref_file and ref_page:
        page_docs = get_page_documents(session.vector_store, ref_file, ref_page)
        if page_docs:
            result = answer_from_documents(prompt, page_docs)
            try:
                session.memory.save_context(
                    {"question": prompt}, {"answer": result["answer"]}
                )
            except Exception:
                pass
            return result

    if session.chain is None:
        raise HTTPException(
            status_code=400,
            detail="Upload and process PDFs before chatting.",
        )
    return query_chain(session.chain, prompt, session.vector_store)


def create_session() -> AppSession:
    """Create a new session, loading any persisted vector store."""
    session = AppSession(session_id=str(uuid.uuid4()))
    vector_store = load_existing_vector_store()
    session.vector_store = vector_store
    if vector_store is not None:
        session.rebuild_chain()
    return session


app = FastAPI(title=f"{APP_NAME} API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_sessions: Dict[str, AppSession] = {}
_default_session = create_session()
_sessions[_default_session.session_id] = _default_session


def get_session(session_id: Optional[str] = None) -> AppSession:
    """Return an existing session or fall back to the default one."""
    if session_id and session_id in _sessions:
        return _sessions[session_id]
    return _default_session


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
            "llm": GEMINI_MODEL_NAME,
            "store": VECTOR_STORE,
            "embeddings": EMBEDDING_MODEL_NAME,
            "temperature": LLM_TEMPERATURE,
            "max_tokens": LLM_MAX_TOKENS,
        },
        "messages": session.messages,
        "example_questions": EXAMPLE_QUESTIONS,
        "streamlit_url": STREAMLIT_URL,
        "chat_ready": session.chain is not None,
    }


@app.post("/api/upload")
async def upload_pdfs(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = None,
):
    """Validate, embed, and index uploaded PDF files."""
    session = get_session(session_id)
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    valid_files, invalid_files = validate_pdf_files(files)
    new_files, skipped = filter_new_files(valid_files, session.indexed_files)

    if not new_files and not invalid_files and skipped:
        return {
            "message": f"{len(skipped)} file(s) already indexed, skipped.",
            "processed": 0,
            "skipped": skipped,
            "invalid": invalid_files,
        }

    if not new_files:
        return {
            "message": "No new PDFs to process.",
            "processed": 0,
            "skipped": skipped,
            "invalid": invalid_files,
        }

    for upload in new_files:
        await upload.seek(0)

    documents, failed = load_pdfs(new_files)
    if failed:
        pass
    if not documents:
        raise HTTPException(
            status_code=400,
            detail="No readable text found in the uploaded PDF(s).",
        )

    chunks = split_documents(documents)
    session.vector_store = create_or_update_vector_store(chunks)
    session.rebuild_chain()

    return {
        "message": f"{len(new_files)} PDF(s) processed and indexed.",
        "processed": len(new_files),
        "skipped": skipped,
        "invalid": invalid_files,
        "failed": failed,
        "indexed_files": index_stats(session.vector_store)["files"],
    }


@app.post("/api/chat")
def chat(payload: ChatRequest, session_id: Optional[str] = None):
    """Ask a question against the indexed knowledge base."""
    session = get_session(session_id)
    prompt = (payload.message or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    session.messages.append({"role": "user", "text": prompt})
    result = answer_question(session, prompt)
    answer = result["answer"]
    sources = structured_sources(result.get("source_documents", []))
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
    session.memory = get_memory()
    if session.chain is not None:
        session.chain.memory = session.memory
    return {"message": "Chat cleared.", "messages": []}


@app.post("/api/reset")
def reset_session(session_id: Optional[str] = None):
    """Clear chat history and wipe the indexed knowledge base."""
    session = get_session(session_id)
    clear_vector_store(session.vector_store)
    session.messages = []
    session.memory = get_memory()
    session.chain = None
    session.vector_store = None
    session.indexed_files = []
    return {
        "message": "Session reset.",
        "messages": [],
        "indexed_files": [],
    }
