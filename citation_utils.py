"""Align displayed citations with the generated answer text."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from langchain_core.documents import Document

_STOPWORDS = {
    "about",
    "after",
    "also",
    "been",
    "being",
    "could",
    "from",
    "have",
    "into",
    "more",
    "other",
    "should",
    "such",
    "than",
    "that",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "using",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
}


_REFUSAL_PHRASE = "i don't have enough information in the uploaded documents to answer this."


def is_refusal_answer(answer: str) -> bool:
    """Return True when the model declined to answer from the retrieved context."""
    normalized = " ".join((answer or "").strip().lower().split())
    if not normalized:
        return False
    if _REFUSAL_PHRASE in normalized:
        return True
    return normalized.startswith("i don't have enough information") or normalized.startswith(
        "i do not have enough information"
    )


def ensure_page_label(doc: Document) -> Document:
    """Ensure a chunk carries a human 1-based ``page_label`` in metadata."""
    meta = dict(doc.metadata or {})
    if "page_label" not in meta:
        page = meta.get("page")
        if isinstance(page, int):
            meta["page_label"] = page + 1
        else:
            meta["page_label"] = meta.get("page_label", "?")
    doc.metadata = meta
    return doc


def _doc_key(doc: Document) -> Tuple[Any, ...]:
    """Build a deduplication key for a chunk."""
    meta = doc.metadata or {}
    return (
        meta.get("source"),
        meta.get("page"),
        meta.get("start_index"),
        (doc.page_content or "")[:100],
    )


def _significant_terms(text: str) -> Set[str]:
    """Return lower-case terms (4+ letters) useful for overlap scoring."""
    words = set(re.findall(r"\b[a-z][a-z0-9]{3,}\b", (text or "").lower()))
    return words - _STOPWORDS


def _lexical_overlap_score(answer: str, chunk: str) -> float:
    """Score how many answer terms appear in a candidate chunk."""
    answer_terms = _significant_terms(answer)
    if not answer_terms:
        return 0.0
    chunk_terms = _significant_terms(chunk)
    return len(answer_terms & chunk_terms) / len(answer_terms)


def extract_answer_phrases(answer: str) -> List[str]:
    """Return distinctive phrases from an answer for PDF highlighting."""
    phrases: List[str] = []
    seen: Set[str] = set()
    text = re.sub(r"\[From[^\]]+\]", "", answer or "", flags=re.IGNORECASE)
    text = re.sub(r"\[[^\]]+\]", "", text)

    for match in re.finditer(r'"([^"]{3,120})"|\'([^\']{3,120})\'', text):
        phrase = (match.group(1) or match.group(2) or "").strip()
        key = phrase.lower()
        if phrase and key not in seen:
            seen.add(key)
            phrases.append(phrase)

    for match in re.finditer(r"\b[A-Z][\w\-']+(?:\s+[A-Z][\w\-']+)+\b", text):
        phrase = match.group().strip()
        key = phrase.lower()
        if len(phrase) >= 4 and key not in seen and not key.startswith("from "):
            seen.add(key)
            phrases.append(phrase)

    for term in sorted(_significant_terms(text), key=len, reverse=True):
        if len(term) >= 5 and term not in seen:
            seen.add(term)
            phrases.append(term)

    return phrases


def best_excerpt_fragments(answer: str, excerpt: str, limit: int = 5) -> List[str]:
    """Return excerpt substrings that best match the answer for PDF search."""
    if not excerpt:
        return []

    fragments: List[str] = []
    seen: Set[str] = set()

    def _add(fragment: str) -> None:
        cleaned = " ".join((fragment or "").split())
        key = cleaned.lower()
        if len(cleaned) >= 8 and key not in seen:
            seen.add(key)
            fragments.append(cleaned)

    for phrase in extract_answer_phrases(answer):
        if phrase.lower() in excerpt.lower():
            _add(phrase)
            for sentence in re.split(r"(?<=[.!?])\s+|\n", excerpt):
                if phrase.lower() in sentence.lower():
                    _add(sentence)

    scored: List[Tuple[float, str]] = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n", excerpt):
        cleaned = " ".join(sentence.split())
        if len(cleaned) < 12:
            continue
        overlap = _lexical_overlap_score(answer, cleaned)
        if overlap > 0:
            scored.append((overlap, cleaned))
    scored.sort(key=lambda item: item[0], reverse=True)
    for _, sentence in scored[:3]:
        _add(sentence)

    return fragments[:limit]


def count_answer_phrase_hits(answer: str, content: str) -> int:
    """Count how many answer phrases appear in a chunk."""
    lowered = (content or "").lower()
    hits = 0
    for phrase in extract_answer_phrases(answer):
        if phrase.lower() in lowered:
            hits += 1
    return hits


def sources_mentioned_in_answer(answer: str) -> List[str]:
    """Return PDF filenames explicitly referenced in an assistant answer."""
    found: List[str] = []
    seen: Set[str] = set()
    patterns = (
        r"\[From\s+([^\],\]]+\.pdf)\b",
        r"\*\*([^\*\n]+\.pdf)\*\*",
        r"`([^`\n]+\.pdf)`",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, answer, re.IGNORECASE):
            name = match.group(1).strip()
            key = name.lower()
            if name and key not in seen:
                seen.add(key)
                found.append(name)
    return found


def _match_source_filename(candidate: str, available_sources: List[str]) -> Optional[str]:
    """Map a filename mentioned in text to an indexed source name."""
    if not candidate:
        return None
    candidate_lower = candidate.lower()
    for source in available_sources:
        if source.lower() == candidate_lower:
            return source
    for source in available_sources:
        if candidate_lower in source.lower() or source.lower() in candidate_lower:
            return source
    return None


def _chunk_supports_answer(
    combined: float,
    answer_sim: float,
    lexical: float,
    phrase_hits: int,
    top_combined: float,
    *,
    relaxed: bool = False,
) -> bool:
    """Return True when a chunk is a reasonable citation for the answer."""
    margin = 0.22 if relaxed else 0.18
    close_to_top = combined >= max(0.28 if relaxed else 0.30, top_combined - margin)
    supports_answer = (
        answer_sim >= (0.22 if relaxed else 0.26)
        or lexical >= (0.05 if relaxed else 0.08)
        or phrase_hits >= 1
    )
    return close_to_top and supports_answer


def _cosine_similarity(left: List[float], right: List[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def resolve_citation_sources(
    answer: str,
    question: str,
    retrieved_docs: List[Document],
    vector_store=None,
    max_sources: int = 3,
) -> List[Document]:
    """Pick source chunks that actually support the generated answer.

    The RAG chain returns every chunk passed to the LLM, but the UI should
    only show excerpts that align with the answer. This re-ranks retrieved
    chunks against the answer and runs a follow-up vector search so the
    correct page (e.g. p.3) is not missed when question retrieval skews
    toward neighbouring sections.

    Args:
        answer: Generated assistant answer.
        question: Original user question.
        retrieved_docs: Chunks returned by the retrieval chain.
        vector_store: Optional Chroma store for answer-focused search.
        max_sources: Maximum citations to display.

    Returns:
        Filtered, ranked list of supporting ``Document`` objects.
    """
    if is_refusal_answer(answer):
        return []

    candidates: List[Document] = [
        ensure_page_label(doc) for doc in (retrieved_docs or [])
    ]

    if vector_store is not None and answer.strip():
        search_query = f"{question.strip()}\n{answer.strip()[:800]}"
        try:
            from vector_store import retrieve_balanced_documents

            extra = retrieve_balanced_documents(
                vector_store,
                search_query,
                global_k=6,
                per_file_k=2,
            )
            candidates.extend(extra)
        except Exception:
            pass

    seen = set()
    unique: List[Document] = []
    for doc in candidates:
        key = _doc_key(doc)
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)

    if not unique:
        return []

    from vector_store import get_embeddings

    embeddings = get_embeddings()
    answer_vec = embeddings.embed_query(answer[:2000])
    question_vec = embeddings.embed_query(question[:500])
    answer_phrases = extract_answer_phrases(answer)
    has_distinctive_phrases = any(
        " " in phrase or len(phrase) >= 6 for phrase in answer_phrases
    )

    scored: List[Tuple[float, float, float, int, Document]] = []
    for doc in unique:
        content = doc.page_content or ""
        chunk_vec = embeddings.embed_query(content[:2000])
        answer_sim = _cosine_similarity(answer_vec, chunk_vec)
        question_sim = _cosine_similarity(question_vec, chunk_vec)
        lexical = _lexical_overlap_score(answer, content)
        phrase_hits = count_answer_phrase_hits(answer, content)
        phrase_boost = min(phrase_hits, 3) * 0.12
        combined = (
            0.50 * answer_sim
            + 0.22 * question_sim
            + 0.18 * lexical
            + phrase_boost
        )
        scored.append((combined, answer_sim, lexical, phrase_hits, doc))

    scored.sort(key=lambda item: item[0], reverse=True)
    top_combined = scored[0][0]

    selected: List[Document] = []
    selected_keys: Set[Tuple[Any, ...]] = set()

    def _add_doc(doc: Document) -> bool:
        key = _doc_key(doc)
        if key in selected_keys:
            return False
        selected_keys.add(key)
        selected.append(doc)
        return True

    unique_sources = sorted(
        {
            (doc.metadata or {}).get("source")
            for doc in unique
            if (doc.metadata or {}).get("source")
        }
    )
    mentioned = sources_mentioned_in_answer(answer)
    target_files: List[str] = []
    seen_targets: Set[str] = set()
    for name in mentioned:
        matched = _match_source_filename(name, unique_sources)
        if matched and matched not in seen_targets:
            seen_targets.add(matched)
            target_files.append(matched)

    multi_doc_answer = len(unique_sources) >= 2 and (
        len(target_files) >= 2 or len(unique_sources) >= 2
    )
    effective_max = (
        max(max_sources, min(max(len(target_files), len(unique_sources)) * 2, 8))
        if multi_doc_answer
        else max_sources
    )

    if multi_doc_answer:
        files_to_cover = target_files if len(target_files) >= 2 else unique_sources
        for filename in files_to_cover:
            for combined, answer_sim, lexical, phrase_hits, doc in scored:
                if (doc.metadata or {}).get("source") != filename:
                    continue
                if _chunk_supports_answer(
                    combined,
                    answer_sim,
                    lexical,
                    phrase_hits,
                    top_combined,
                    relaxed=True,
                ):
                    _add_doc(doc)
                    break

    for combined, answer_sim, lexical, phrase_hits, doc in scored:
        if len(selected) >= effective_max:
            break
        if _chunk_supports_answer(
            combined, answer_sim, lexical, phrase_hits, top_combined
        ):
            _add_doc(doc)

    if not selected:
        for combined, answer_sim, lexical, phrase_hits, doc in scored:
            if len(selected) >= effective_max:
                break
            question_overlap = _lexical_overlap_score(
                question, doc.page_content or ""
            )
            if question_overlap >= 0.20 and combined >= max(0.30, top_combined - 0.12):
                _add_doc(doc)

    return selected
