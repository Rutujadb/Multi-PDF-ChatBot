"""Retrieval-Augmented Generation chain: LLM, chat history, and orchestration.

Wires the vector-store retriever to the configured LLM (OpenRouter, Groq,
Nvidia, or Gemini) with conversation memory, using a grounding prompt so
answers come only from the uploaded documents.

SRS references: FR-RAG-01, FR-RAG-02, FR-RAG-03, FR-RAG-04, FR-MEM-01, FR-MEM-02.

Note on versions: this project runs LangChain 1.x. ``ConversationalRetrievalChain``
now lives in the ``langchain_classic`` package (it was in ``langchain.*`` in the
0.2.x era the PLAN was written against).
"""

from typing import Any, Dict, Optional

from langchain_core.chat_history import InMemoryChatMessageHistory
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
)
from citation_utils import ensure_page_label, is_refusal_answer, resolve_citation_sources


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


def get_memory() -> InMemoryChatMessageHistory:
    """Create a fresh in-memory chat history for one isolated session."""
    return InMemoryChatMessageHistory()


def build_rag_chain(
    retriever: BaseRetriever,
    chat_history: Optional[InMemoryChatMessageHistory] = None,
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
    return chain


def query_chain(
    chain: ConversationalRetrievalChain,
    question: str,
    vector_store=None,
    chat_history: Optional[InMemoryChatMessageHistory] = None,
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
        history_messages: list[BaseMessage] = (
            list(chat_history.messages) if chat_history is not None else []
        )
        result = chain.invoke(
            {"question": question, "chat_history": history_messages}
        )
        answer = result.get(
            "answer", "Sorry, I could not generate an answer."
        )
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
        return {
            "answer": answer,
            "source_documents": cited_sources,
        }
    except Exception as e:
        return {
            "answer": f"Error generating answer: {e}",
            "source_documents": [],
        }


def answer_from_documents(
    question: str,
    documents,
    vector_store=None,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Answer a question using an explicit set of documents (no retrieval).

    Used for page-targeted questions, where the relevant chunks are fetched by
    metadata filter rather than semantic similarity. Each chunk is labelled with
    its source and page in the context, and the same grounding prompt is used.

    Args:
        question: The user's question.
        documents: List of ``Document`` chunks to answer from.

    Returns:
        Dict with ``answer`` (str) and ``source_documents`` (the given list).
    """
    if not documents:
        return {
            "answer": "I don't have enough information in the uploaded "
                      "documents to answer this.",
            "source_documents": [],
        }

    labeled_docs = [ensure_page_label(doc) for doc in documents]
    context = "\n\n".join(
        f"[From {doc.metadata.get('source', 'Unknown')}, "
        f"page {doc.metadata.get('page_label', '?')}]\n{doc.page_content}"
        for doc in labeled_docs
    )
    prompt = _qa_template().format(context=context, question=question)

    try:
        response = get_llm(
            llm_provider=llm_provider,
            llm_model=llm_model,
        ).invoke(prompt)
        answer = getattr(response, "content", str(response))
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
        return {
            "answer": f"Error generating answer: {e}",
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
