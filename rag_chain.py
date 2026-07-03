"""Retrieval-Augmented Generation chain: LLM, chat history, and orchestration.

Wires the vector-store retriever to the configured LLM (OpenRouter, Groq,
Nvidia, or Gemini) with conversation memory, using a grounding prompt so
answers come only from the uploaded documents.

SRS references: FR-RAG-01, FR-RAG-02, FR-RAG-03, FR-RAG-04, FR-MEM-01, FR-MEM-02.

Note on versions: this project runs LangChain 1.x. ``ConversationalRetrievalChain``
now lives in the ``langchain_classic`` package (it was in ``langchain.*`` in the
0.2.x era the PLAN was written against).
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_classic.chains import ConversationalRetrievalChain

from config import (
    LLM_PROVIDER,
    GOOGLE_API_KEY,
    GEMINI_MODEL_NAME,
    GROQ_API_KEY,
    GROQ_MODEL,
    NVIDIA_API_KEY,
    NVIDIA_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_APP_NAME,
    OPENROUTER_HTTP_REFERER,
    OPENROUTER_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_TOP_P,
    LLM_TOP_K,
    LLM_REPETITION_PENALTY,
    LLM_FREQUENCY_PENALTY,
    CITATION_MAX_SOURCES,
    EXAMPLE_QUESTIONS,
    SUGGESTED_QUESTION_COUNT,
    SUGGESTED_QUESTION_RETRIEVAL_QUERY,
)
from citation_utils import ensure_page_label, is_refusal_answer, resolve_citation_sources
from image_rag import enrich_documents_with_image_context
from utils import format_llm_error
from vector_store import get_indexed_filenames, retrieve_balanced_documents

logger = logging.getLogger(__name__)


def get_llm(
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> BaseLanguageModel:
    """Load the configured chat LLM (OpenRouter, Groq, Nvidia, or Gemini).

    The provider is selected by ``LLM_PROVIDER`` in ``config.py`` / ``.env``.
    API keys are read from configuration and never hardcoded.

    Returns:
        A LangChain chat-model instance.
    """
    provider = (llm_provider or LLM_PROVIDER).strip().lower()
    model_name = llm_model or {
        "openrouter": OPENROUTER_MODEL,
        "groq": GROQ_MODEL,
        "nvidia": NVIDIA_MODEL,
        "gemini": GEMINI_MODEL_NAME,
    }.get(provider, "unknown")
    logger.info("Initialising LLM: provider=%s, model=%s", provider, model_name)

    if provider == "openrouter":
        if not OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. Add it to your .env file."
            )
        return ChatOpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            model=llm_model or OPENROUTER_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            model_kwargs={
                "top_p": LLM_TOP_P,
                "frequency_penalty": LLM_FREQUENCY_PENALTY,
            },
            default_headers={
                "HTTP-Referer": OPENROUTER_HTTP_REFERER,
                "X-Title": OPENROUTER_APP_NAME,
            },
        )

    if provider == "groq":
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to your .env file."
            )
        return ChatOpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            model=llm_model or GROQ_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            top_p=LLM_TOP_P,
        )

    if provider == "nvidia":
        if not NVIDIA_API_KEY:
            raise ValueError(
                "NVIDIA_API_KEY is not set. Add it to your .env file."
            )
        return ChatOpenAI(
            api_key=NVIDIA_API_KEY,
            base_url="https://integrate.api.nvidia.com/v1",
            model=llm_model or NVIDIA_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            top_p=LLM_TOP_P,
        )

    if provider == "gemini":
        if not GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY is not set. Add it to your .env file or set "
                "LLM_PROVIDER to openrouter, groq, or nvidia with the matching "
                "API key."
            )
        return ChatGoogleGenerativeAI(
            model=llm_model or GEMINI_MODEL_NAME,
            google_api_key=GOOGLE_API_KEY,
            temperature=LLM_TEMPERATURE,
            top_p=LLM_TOP_P,
            top_k=LLM_TOP_K,
            max_output_tokens=LLM_MAX_TOKENS,
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER: {provider!r}. "
        "Use openrouter, groq, nvidia, or gemini."
    )


def get_memory(session_id: str) -> BaseChatMessageHistory:
    """Return SQLite-backed chat history for one session."""
    from sqlite_memory import SqliteChatMessageHistory, create_session

    sid = (session_id or "").strip()
    if not sid:
        raise ValueError("session_id is required for persistent chat memory.")
    create_session(sid)
    return SqliteChatMessageHistory(sid)


def build_rag_chain(
    retriever: BaseRetriever,
    chat_history: Optional[BaseChatMessageHistory] = None,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> ConversationalRetrievalChain:
    """Build the conversational RAG chain from retriever, LLM, and memory.

    The chain condenses each follow-up question with the chat history into a
    standalone question, retrieves the most relevant chunks, and answers using
    a grounding prompt. Source documents are returned with every answer.

    Args:
        retriever: Vector-store retriever supplying context chunks.
        chat_history: Session chat history, managed outside the chain.

    Returns:
        A configured ``ConversationalRetrievalChain``.
    """
    logger.info("Building RAG chain (provider=%s, model=%s)", llm_provider, llm_model)
    llm = get_llm(llm_provider=llm_provider, llm_model=llm_model)

    # The document-combining step only receives ``context`` and ``question``;
    # chat history is handled separately by the question-condensing step, so the
    # combine prompt must not require a ``chat_history`` variable.
    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=_qa_template(),
    )

    # Label each retrieved chunk with its source filename and page inside the
    # context, so the model can name documents and summarise them per file.
    document_prompt = PromptTemplate(
        input_variables=["page_content", "source", "page_label"],
        template="[From {source}, page {page_label}]\n{page_content}",
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={
            "prompt": qa_prompt,
            "document_prompt": document_prompt,
        },
        verbose=False,
    )
    logger.info("RAG chain built successfully")
    return chain


def query_chain(
    chain: ConversationalRetrievalChain,
    question: str,
    vector_store=None,
    chat_history: Optional[BaseChatMessageHistory] = None,
) -> Dict[str, Any]:
    """Invoke the RAG chain with a user question.

    All errors are caught and returned as a user-facing message so the UI never
    sees a raw stack trace (NFR-REL-03). Returned ``source_documents`` are
    filtered to chunks that best support the generated answer.

    Args:
        chain: A built ``ConversationalRetrievalChain``.
        question: The user's question.
        vector_store: Optional vector store for answer-aligned citation search.

    Returns:
        Dict with ``answer`` (str) and ``source_documents`` (list of Documents).
    """
    try:
        logger.info("RAG query: %s", question[:120])
        history_messages: list[BaseMessage] = (
            list(chat_history.messages) if chat_history is not None else []
        )
        logger.debug("Chat history contains %d messages", len(history_messages))
        result = chain.invoke(
            {"question": question, "chat_history": history_messages}
        )
        answer = result.get(
            "answer", "Sorry, I could not generate an answer."
        )
        logger.info("LLM answered (%d chars), %d source docs returned",
                     len(answer), len(result.get("source_documents", [])))
        if chat_history is not None:
            chat_history.add_messages(
                [HumanMessage(content=question), AIMessage(content=answer)]
            )
        raw_sources = result.get("source_documents", [])
        for doc in raw_sources:
            ensure_page_label(doc)
        if is_refusal_answer(answer):
            cited_sources = []
        else:
            cited_sources = resolve_citation_sources(
                answer,
                question,
                raw_sources,
                vector_store=vector_store,
                max_sources=CITATION_MAX_SOURCES,
            )
        logger.info("Citations resolved: %d sources", len(cited_sources))
        return {
            "answer": answer,
            "source_documents": cited_sources,
        }
    except Exception as e:
        logger.error("RAG query failed: %s", e, exc_info=True)
        return {
            "answer": format_llm_error(e),
            "source_documents": [],
        }


def answer_from_chat_history(
    question: str,
    chat_history: Optional[BaseChatMessageHistory] = None,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Answer questions about the current chat session using stored turns.

    Meta questions such as "what was my previous question?" are not answerable
    from PDF retrieval context, so this path reads SQLite-backed chat history
    instead of running document RAG.

    Args:
        question: The user's question.
        chat_history: Session chat history with prior turns.
        llm_provider: Optional LLM provider override.
        llm_model: Optional LLM model override.

    Returns:
        Dict with ``answer`` (str) and empty ``source_documents``.
    """
    logger.info("Answering from chat history (conversation recall): %s", question[:120])
    messages = list(chat_history.messages) if chat_history is not None else []
    human_turns = [
        str(message.content)
        for message in messages
        if isinstance(message, HumanMessage)
    ]

    if not human_turns:
        answer = "We have not exchanged any messages yet in this session."
    else:
        lowered = (question or "").lower()
        previous_markers = (
            "previous question",
            "last question",
            "what did i ask",
            "what was my question",
        )
        if any(marker in lowered for marker in previous_markers):
            answer = f'Your previous question was: "{human_turns[-1]}"'
        else:
            history_lines = []
            for message in messages:
                role = "You" if isinstance(message, HumanMessage) else "Assistant"
                history_lines.append(f"{role}: {message.content}")
            history_text = "\n".join(history_lines)
            prompt = (
                "You are helping the user understand their current chat session. "
                "Answer using only the conversation history below. "
                "Do not invent messages that are not shown.\n\n"
                f"Conversation history:\n{history_text}\n\n"
                f"Question: {question}\nAnswer:"
            )
            try:
                response = get_llm(
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                ).invoke(prompt)
                answer = getattr(response, "content", str(response))
            except Exception as exc:
                answer = format_llm_error(exc)

    if chat_history is not None:
        chat_history.add_messages(
            [HumanMessage(content=question), AIMessage(content=answer)]
        )
    return {"answer": answer, "source_documents": []}


def _parse_suggested_questions(raw_text: str, count: int) -> List[str]:
    """Parse an LLM response into a short list of user-facing questions."""
    text = (raw_text or "").strip()
    if not text:
        return []

    candidates: List[str] = []
    seen_candidates: set[str] = set()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            for item in parsed:
                cleaned = str(item).strip()
                if not cleaned:
                    continue
                key = cleaned.lower()
                if key in seen_candidates:
                    continue
                seen_candidates.add(key)
                candidates.append(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    for item in parsed:
                        cleaned = str(item).strip()
                        if not cleaned:
                            continue
                        key = cleaned.lower()
                        if key in seen_candidates:
                            continue
                        seen_candidates.add(key)
                        candidates.append(cleaned)
            except json.JSONDecodeError:
                pass

    if not candidates:
        for line in text.splitlines():
            cleaned = re.sub(r"^[\s\d\-\*\.\)]+", "", line).strip().strip("\"'")
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen_candidates:
                continue
            seen_candidates.add(key)
            candidates.append(cleaned)

    questions: List[str] = []
    seen_questions: set[str] = set()
    for item in candidates:
        question = item.strip().strip("\"'")
        if not question:
            continue
        if not question.endswith("?"):
            question = f"{question.rstrip('.')}?"
        if len(question) < 12 or len(question) > 160:
            continue
        key = question.lower()
        if key in seen_questions:
            continue
        seen_questions.add(key)
        questions.append(question)
        if len(questions) >= count:
            break
    return questions


def generate_suggested_questions(
    vector_store,
    indexed_files: Optional[List[str]] = None,
    *,
    count: int = SUGGESTED_QUESTION_COUNT,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> List[str]:
    """Generate starter questions grounded in indexed PDF content.

    Retrieves representative chunks from every indexed file, then asks the
    configured LLM for practical questions a user could ask about that content.

    Args:
        vector_store: Chroma vector store with indexed chunks.
        indexed_files: Optional list of indexed filenames; inferred if omitted.
        count: Number of questions to return.
        llm_provider: Optional LLM provider override.
        llm_model: Optional LLM model override.

    Returns:
        A list of question strings, or ``EXAMPLE_QUESTIONS`` on failure.
    """
    filenames = list(indexed_files or [])
    if not filenames and vector_store is not None:
        filenames = get_indexed_filenames(vector_store)
    if not vector_store or not filenames:
        return list(EXAMPLE_QUESTIONS[:count])

    documents = retrieve_balanced_documents(
        vector_store,
        SUGGESTED_QUESTION_RETRIEVAL_QUERY,
        per_file_k=2,
    )
    if not documents:
        return list(EXAMPLE_QUESTIONS[:count])

    labeled_docs = [ensure_page_label(doc) for doc in documents]
    context = "\n\n".join(
        f"[From {doc.metadata.get('source', 'Unknown')}, "
        f"page {doc.metadata.get('page_label', '?')}]\n{doc.page_content}"
        for doc in labeled_docs
    )
    file_list = ", ".join(filenames)
    prompt = (
        "You help users explore uploaded PDF documents. Based ONLY on the "
        "context below, suggest "
        f"{count} specific, practical questions a user could ask about this "
        "content. Each question must be answerable from the documents. "
        "Use concrete wording from the material (for example, if the PDF "
        "describes paid leave, ask 'How many paid leave days are available?' "
        "or 'How do you apply for paid leave?'). Avoid generic prompts like "
        "'What are the key points?' or 'Summarise the documents.'\n\n"
        f"Indexed files: {file_list}\n\n"
        "Return ONLY a JSON array of strings. Example:\n"
        '["How many paid leave days are available?", '
        '"How do you apply for paid leave?"]\n\n'
        f"Context:\n{context}"
    )

    try:
        logger.info("Generating %d suggested questions from indexed content", count)
        response = get_llm(
            llm_provider=llm_provider,
            llm_model=llm_model,
        ).invoke(prompt)
        raw = getattr(response, "content", str(response))
        questions = _parse_suggested_questions(raw, count)
        if questions:
            logger.info("Generated %d suggested questions", len(questions))
            return questions
    except Exception:
        logger.warning("Failed to generate suggested questions, using defaults", exc_info=True)
    return list(EXAMPLE_QUESTIONS[:count])


def answer_from_documents(
    question: str,
    documents,
    vector_store=None,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Answer a question using an explicit set of documents (no retrieval).

    Used for page-targeted questions, where the relevant chunks are fetched by
    metadata filter rather than semantic similarity. Each chunk is labelled with
    its source and page in the context, and the same grounding prompt is used.

    Args:
        question: The user's question.
        documents: List of ``Document`` chunks to answer from.
        vector_store: Optional vector store for citation resolution.
        llm_provider: Optional LLM provider override.
        llm_model: Optional LLM model override.
        session_id: Optional session id for image caption context.

    Returns:
        Dict with ``answer`` (str) and ``source_documents`` (the given list).
    """
    if not documents:
        return {
            "answer": "I don't have enough information in the uploaded "
                      "documents to answer this.",
            "source_documents": [],
        }

    logger.info("Answering from %d explicit documents (provider=%s, model=%s)",
                len(documents), llm_provider, llm_model)
    labeled_docs = [ensure_page_label(doc) for doc in documents]
    context_docs = enrich_documents_with_image_context(labeled_docs, session_id)
    context = "\n\n".join(
        f"[From {doc.metadata.get('source', 'Unknown')}, "
        f"page {doc.metadata.get('page_label', '?')}]\n{doc.page_content}"
        for doc in context_docs
    )
    prompt = _qa_template().format(context=context, question=question)

    try:
        logger.info("Invoking LLM for document-targeted answer…")
        response = get_llm(
            llm_provider=llm_provider,
            llm_model=llm_model,
        ).invoke(prompt)
        answer = getattr(response, "content", str(response))
        logger.info("LLM response received (%d chars)", len(answer))
        if is_refusal_answer(answer):
            cited = []
        else:
            cited = resolve_citation_sources(
                answer,
                question,
                labeled_docs,
                vector_store=vector_store,
                max_sources=CITATION_MAX_SOURCES,
            )
        return {"answer": answer, "source_documents": cited}
    except Exception as e:
        logger.error("LLM call failed for document-targeted answer: %s", e, exc_info=True)
        return {
            "answer": format_llm_error(e),
            "source_documents": [],
        }


def _qa_template() -> str:
    """Return the grounding prompt for the answer-generation step.

    Derived from ``SYSTEM_PROMPT_TEMPLATE`` in ``config.py`` but with the
    ``chat_history`` block removed, because the combine-documents step in
    ``ConversationalRetrievalChain`` is only given ``context`` and ``question``.
    History is still used - it drives the separate question-condensing step.

    Returns:
        A prompt template string using only ``{context}`` and ``{question}``.
    """
    return (
        "You are a helpful assistant that answers questions about the user's "
        "uploaded PDF documents. Base your answer only on the provided context "
        "below. Each excerpt is prefixed with its source as "
        "'[From <filename>, page <n>]', so you can refer to documents by their "
        "filename and summarise each one. When multiple documents appear in the "
        "context and the question asks about each PDF, all PDFs, or a general "
        "summary, provide a separate summary for every document filename you "
        "see in the context. You may summarise and synthesise across the "
        "context - for example, to describe the topics, themes, or main points "
        "covered. Treat synonyms, abbreviations, and related phrasing "
        "as relevant (e.g. COVID, COVID-19, coronavirus). "
        "Only if the context contains nothing relevant to the question, reply "
        'exactly: "I don\'t have enough information in the uploaded documents '
        'to answer this." Do not use any outside knowledge.'
        "\n\nContext:\n{context}\n\n"
        "Question: {question}\nAnswer:"
    )
