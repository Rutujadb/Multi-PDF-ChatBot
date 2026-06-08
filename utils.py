"""Helper utilities: source-citation formatting, validation, chat export.

SRS references: FR-UI-03 (source citation display), FR-PDF-01 (PDF-only upload),
NFR-USE-02 (plain-English messaging).
"""

import re
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from citation_utils import best_excerpt_fragments, extract_answer_phrases

_PAGE_RE = re.compile(r"page\s*(\d+)", re.IGNORECASE)

_MULTI_DOC_OVERVIEW_PATTERNS = (
    re.compile(r"\beach\s+(pdf|pdfs|document|documents|file|files)\b", re.I),
    re.compile(r"\ball\s+(pdf|pdfs|document|documents|file|files)\b", re.I),
    re.compile(r"\bevery\s+(pdf|pdfs|document|documents|file|files)\b", re.I),
    re.compile(r"\bboth\s+(pdf|pdfs|document|documents|file|files)\b", re.I),
    re.compile(r"^summar(?:y|ise|ize)(?:\s+all)?[.!?]?$", re.I),
    re.compile(r"\bwhat\s+(?:is|are)\s+each\b", re.I),
    re.compile(r"\btell\s+me\s+what\s+each\b", re.I),
    re.compile(r"\boverview\s+of\s+(?:all|each|every)\b", re.I),
    re.compile(r"\bwhat\s+(?:is|are)\s+(?:the\s+)?(?:pdf|pdfs|document|documents)\b", re.I),
)


def is_multi_document_overview(question: str, indexed_file_count: int) -> bool:
    """Return True when the user is asking for a summary across all PDFs.

    Args:
        question: The user's question text.
        indexed_file_count: Number of files currently indexed.

    Returns:
        ``True`` if the question should retrieve context from every indexed PDF.
    """
    if indexed_file_count < 2:
        return False
    text = (question or "").strip()
    if not text:
        return False
    return any(pattern.search(text) for pattern in _MULTI_DOC_OVERVIEW_PATTERNS)


def parse_page_reference(
    question: str, indexed_files: List[str]
) -> Tuple[Optional[str], Optional[int]]:
    """Detect a "page N of <file>" reference in a question.

    Resolves the filename by matching any indexed file name (or its stem)
    mentioned in the question; if exactly one file is indexed, that file is
    assumed when none is named explicitly.

    Args:
        question: The user's question text.
        indexed_files: Filenames currently in the vector store.

    Returns:
        Tuple of (filename, page_number). Both are ``None`` if no resolvable
        page+file reference is found.
    """
    match = _PAGE_RE.search(question or "")
    if not match:
        return None, None
    page = int(match.group(1))

    lowered = question.lower()
    filename = None
    for name in indexed_files or []:
        stem = name.lower().rsplit(".", 1)[0]
        if name.lower() in lowered or stem in lowered:
            filename = name
            break
    if filename is None and indexed_files and len(indexed_files) == 1:
        filename = indexed_files[0]

    if filename is None:
        return None, None
    return filename, page


def _page_number(metadata: dict) -> str:
    """Return a human-friendly 1-based page label from chunk metadata.

    Args:
        metadata: A chunk's metadata dictionary.

    Returns:
        The page label string (e.g. ``"3"``), or ``"?"`` if unknown.
    """
    page = metadata.get("page")
    if isinstance(page, int):
        return str(page + 1)  # PyPDFLoader pages are 0-based
    return str(metadata.get("page_label", "?"))


def _source_label(source: str, page: str, line) -> str:
    """Build a compact human-readable source label."""
    label = f"{source} - p.{page}"
    if line:
        label += f", line {line}"
    return label


def extract_source_items(
    source_documents: List[Document],
    answer: Optional[str] = None,
) -> List[dict]:
    """Convert retrieved documents into structured, clickable source entries.

    Args:
        source_documents: Retrieved chunks from the RAG chain.
        answer: Optional assistant answer used to pick PDF highlight phrases.

    Returns:
        List of dicts with ``file``, ``page``, ``line``, ``excerpt``, ``label``.
    """
    items: List[dict] = []
    seen = set()
    for doc in source_documents or []:
        meta = doc.metadata or {}
        source = meta.get("source", "Unknown")
        page = _page_number(meta)
        line = meta.get("line")
        excerpt = (doc.page_content or "").strip()
        key = (source, page, line, excerpt[:160])
        if key in seen:
            continue
        seen.add(key)

        highlight_phrases: List[str] = []
        if answer:
            phrase_seen = set()
            for phrase in extract_answer_phrases(answer):
                key_phrase = phrase.lower()
                if key_phrase not in phrase_seen:
                    phrase_seen.add(key_phrase)
                    highlight_phrases.append(phrase)
            for fragment in best_excerpt_fragments(answer, excerpt):
                key_phrase = fragment.lower()
                if key_phrase not in phrase_seen:
                    phrase_seen.add(key_phrase)
                    highlight_phrases.append(fragment)

        items.append(
            {
                "file": source,
                "page": int(page) if str(page).isdigit() else page,
                "line": line,
                "excerpt": excerpt,
                "label": _source_label(source, page, line),
                "highlight_phrases": highlight_phrases,
            }
        )
    return items


def format_sources(source_documents: List[Document]) -> str:
    """Format retrieved-document metadata into a readable citation string.

    Each unique (file, page, line) location is listed once, so the user can see
    exactly where in which PDF an answer was grounded.

    Args:
        source_documents: List of retrieved ``Document`` objects.

    Returns:
        A markdown citation string such as::

            📄 **Sources:**
            • report.pdf - p.3, line 12
            • manual.pdf - p.1, line 5

        or an empty string if no source documents were provided.
    """
    if not source_documents:
        return ""

    citations: List[str] = []
    for item in extract_source_items(source_documents):
        citations.append(item["label"])

    return "📄 **Sources:**  \n" + "  \n".join(f"• {c}" for c in citations)


def validate_pdf_files(uploaded_files) -> Tuple[list, List[str]]:
    """Split uploaded files into valid PDFs and invalid ones.

    A file is valid only if its name ends with ``.pdf`` (case insensitive) and
    it is not empty.

    Args:
        uploaded_files: List of Streamlit ``UploadedFile`` objects.

    Returns:
        Tuple of (valid_files, invalid_filenames).
    """
    valid = []
    invalid: List[str] = []
    for f in uploaded_files:
        if f.name.lower().endswith(".pdf") and getattr(f, "size", 1) > 0:
            valid.append(f)
        else:
            invalid.append(f.name)
    return valid, invalid


def build_chat_export(messages: List[dict], fmt: str = "md") -> str:
    """Render the conversation as a downloadable Markdown or plain-text string.

    Args:
        messages: List of message dicts with ``role``, ``content``, and an
            optional ``sources`` string.
        fmt: ``"md"`` for Markdown, ``"txt"`` for plain text.

    Returns:
        The formatted conversation as a single string.
    """
    lines: List[str] = []
    is_md = fmt == "md"

    if is_md:
        lines.append("# Multi-PDF ChatBot - Conversation\n")

    for message in messages:
        role = "You" if message["role"] == "user" else "Assistant"
        content = message.get("content", "")
        # Flatten any multi-line source citation onto a single line.
        sources = (message.get("sources") or "").replace("\n", " ").strip()

        if is_md:
            lines.append(f"**{role}:** {content}\n")
            if sources:
                lines.append(f"_{sources}_\n")
        else:
            lines.append(f"{role}: {content}")
            if sources:
                lines.append(sources)
            lines.append("")

    return "\n".join(lines).strip() + "\n"
