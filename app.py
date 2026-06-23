"""Multi-PDF ChatBot - Streamlit entry point.

Renders the web UI and orchestrates the PDF-processing, vector-store, and RAG
modules. All stateful data lives in ``st.session_state``.

SRS references: FR-UI-01 → FR-UI-07, FR-PDF-01, FR-MEM-01, FR-MEM-03, FR-MEM-04.
"""
import os
import streamlit as st


def _apply_streamlit_secrets() -> None:
    """Map Streamlit Cloud secrets to os.environ for config.py."""
    try:
        for key, value in st.secrets.items():
            if isinstance(value, (str, int, float, bool)):
                os.environ.setdefault(str(key), str(value))
    except Exception:
        pass


_apply_streamlit_secrets()


import streamlit as st

from config import (
    APP_NAME,
    EXAMPLE_QUESTIONS,
    get_available_llm_options,
    get_default_llm_option,
)
from pdf_processor import load_pdfs, split_documents, filter_new_files
from pdf_storage import clear_all_pdfs, delete_pdf, save_uploaded_pdf
from source_viewer import dismiss_pdf_viewer, render_pdf_viewer_modal, render_source_citations
from vector_store import (
    create_or_update_vector_store,
    load_existing_vector_store,
    get_retriever,
    get_indexed_filenames,
    get_page_documents,
    clear_vector_store,
    delete_file,
    retrieve_balanced_documents,
)
from rag_chain import (
    get_memory,
    build_rag_chain,
    query_chain,
    answer_from_documents,
)
from utils import (
    validate_pdf_files,
    format_sources,
    extract_source_items,
    build_chat_export,
    is_multi_document_overview,
    parse_page_reference,
)

USER_AVATAR = "🧑"
ASSISTANT_AVATAR = "🤖"

st.set_page_config(
    page_title="Multi-PDF ChatBot",
    page_icon="📚",
    layout="wide",
)


def apply_theme(mode: str):
    """Apply a light/dark colour override via injected CSS.

    "Default" applies nothing, so the app follows the user's Streamlit/browser
    theme. "Light" and "Dark" override the main and sidebar colours.

    Args:
        mode: One of ``"Default"``, ``"Light"``, ``"Dark"``.
    """
    if mode == "Dark":
        css = """
        <style>
        .stApp { background-color: #0e1117; color: #fafafa; }
        [data-testid="stSidebar"] { background-color: #1a1d24; }
        [data-testid="stSidebar"] * { color: #fafafa; }
        </style>
        """
    elif mode == "Light":
        css = """
        <style>
        .stApp { background-color: #ffffff; color: #1a1a1a; }
        [data-testid="stSidebar"] { background-color: #f3f4f6; }
        [data-testid="stSidebar"] * { color: #1a1a1a; }
        </style>
        """
    else:
        css = ""

    base = """
    <style>
    .stChatMessage { border-radius: 8px; margin-bottom: 8px; }
    .stCaption { font-size: 0.8rem; margin-top: 4px; }
    div[data-testid="stChatMessage"] div.stButton > button {
        border: 1px solid #dbeafe;
        background: #eff6ff;
        color: #2563eb;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 0.35rem 0.7rem;
        min-height: 2rem;
    }
    div[data-testid="stChatMessage"] div.stButton > button:hover {
        border-color: #2563eb;
        background: #dbeafe;
        color: #1d4ed8;
    }
    </style>
    """
    st.markdown(base, unsafe_allow_html=True)
    if css:
        st.markdown(css, unsafe_allow_html=True)


def initialise_session_state():
    """Initialise all session-state keys, loading any persisted vector store.

    On first load, an existing ChromaDB store (from a previous session) is
    reloaded from disk so the user does not need to re-upload PDFs.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = get_memory()
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    if "uploader_key" not in st.session_state:
        # Part of the file_uploader's widget key; bumping it forces Streamlit
        # to render a fresh, empty uploader (used by "Reset All").
        st.session_state.uploader_key = 0
    if "pdf_viewer" not in st.session_state:
        st.session_state.pdf_viewer = None
    if "pdf_viewer_token" not in st.session_state:
        st.session_state.pdf_viewer_token = 0
    if "pdf_viewer_item" not in st.session_state:
        st.session_state.pdf_viewer_item = None
    if "pdf_viewer_payload" not in st.session_state:
        st.session_state.pdf_viewer_payload = None
    if "pdf_viewer_stage" not in st.session_state:
        st.session_state.pdf_viewer_stage = "idle"
    if "selected_llm_provider" not in st.session_state:
        default_llm = get_default_llm_option()
        st.session_state.selected_llm_provider = default_llm["provider"]
        st.session_state.selected_llm_model = default_llm["model"]
        st.session_state.selected_llm_label = default_llm["label"]
    if "vector_store" not in st.session_state:
        vector_store = load_existing_vector_store()
        st.session_state.vector_store = vector_store
        if vector_store is not None:
            st.session_state.indexed_files = get_indexed_filenames(vector_store)
            if st.session_state.indexed_files:
                retriever = get_retriever(vector_store)
                st.session_state.chain = build_rag_chain(
                    retriever,
                    st.session_state.memory,
                    llm_provider=st.session_state.selected_llm_provider,
                    llm_model=st.session_state.selected_llm_model,
                )


def rebuild_chain():
    """Rebuild the RAG chain from the current vector store, or disable chat.

    If no documents remain indexed, the chain is set to ``None`` so the chat
    input is disabled.
    """
    vector_store = st.session_state.vector_store
    st.session_state.indexed_files = (
        get_indexed_filenames(vector_store) if vector_store else []
    )
    if st.session_state.indexed_files:
        retriever = get_retriever(vector_store)
        st.session_state.chain = build_rag_chain(
            retriever,
            st.session_state.memory,
            llm_provider=st.session_state.selected_llm_provider,
            llm_model=st.session_state.selected_llm_model,
        )
    else:
        st.session_state.chain = None


def _apply_selected_model(label: str) -> None:
    """Persist the selected model option and rebuild the chain if needed."""
    options = get_available_llm_options()
    selected = next(
        (option for option in options if option["label"] == label),
        None,
    )
    if selected is None:
        return

    changed = (
        st.session_state.get("selected_llm_provider") != selected["provider"]
        or st.session_state.get("selected_llm_model") != selected["model"]
    )
    st.session_state.selected_llm_provider = selected["provider"]
    st.session_state.selected_llm_model = selected["model"]
    st.session_state.selected_llm_label = selected["label"]

    if changed and st.session_state.get("indexed_files"):
        rebuild_chain()


def process_uploaded_pdfs(uploaded_files):
    """Validate, de-duplicate, embed, and index the uploaded PDFs.

    Args:
        uploaded_files: Files from the sidebar uploader.
    """
    valid_files, invalid_files = validate_pdf_files(uploaded_files)
    new_files, skipped = filter_new_files(
        valid_files, st.session_state.indexed_files
    )

    if invalid_files:
        st.error(f"Invalid files skipped: {', '.join(invalid_files)}")
    if skipped:
        st.info(f"{len(skipped)} file(s) already indexed, skipped.")

    if not new_files:
        st.info("No new PDFs to process.")
        return

    with st.spinner(f"Processing {len(new_files)} PDF(s)..."):
        for uploaded in new_files:
            uploaded.seek(0)
            save_uploaded_pdf(uploaded.name, uploaded.read())
            uploaded.seek(0)

        documents, failed = load_pdfs(new_files)
        if failed:
            st.warning(f"Could not read (skipped): {', '.join(failed)}")
        if not documents:
            st.error("No readable text found in the uploaded PDF(s).")
            return

        chunks = split_documents(documents)
        vector_store = create_or_update_vector_store(chunks)
        st.session_state.vector_store = vector_store
        rebuild_chain()

    st.success(f"✅ {len(new_files)} PDF(s) processed and indexed!")


def remove_file(filename: str):
    """Remove a single indexed PDF and refresh the chain (FR-MEM, doc mgmt)."""
    delete_file(st.session_state.vector_store, filename)
    delete_pdf(filename)
    viewer_item = st.session_state.get("pdf_viewer_item") or {}
    if viewer_item.get("file") == filename:
        dismiss_pdf_viewer()
        st.session_state.pdf_viewer_stage = "cleanup"
    rebuild_chain()


def clear_chat():
    """Clear the conversation while keeping the indexed PDFs (FR-MEM-03)."""
    dismiss_pdf_viewer()
    st.session_state.pdf_viewer_stage = "cleanup"
    st.session_state.messages = []
    st.session_state.memory = get_memory()
    if st.session_state.chain is not None:
        st.session_state.chain.memory = st.session_state.memory


def reset_session():
    """Clear chat history and the indexed knowledge base (FR-MEM-04)."""
    clear_vector_store(st.session_state.get("vector_store"))
    clear_all_pdfs()
    dismiss_pdf_viewer()
    st.session_state.pdf_viewer_stage = "cleanup"
    st.session_state.messages = []
    st.session_state.memory = get_memory()
    st.session_state.chain = None
    st.session_state.vector_store = None
    st.session_state.indexed_files = []
    # Bump the uploader key so the file_uploader is re-created empty.
    st.session_state.uploader_key += 1


def render_sidebar():
    """Render the sidebar: uploader, process button, indexed list, controls."""
    with st.sidebar:
        st.title("📚 Multi-PDF ChatBot")
        st.markdown("---")

        model_options = get_available_llm_options()
        if model_options:
            labels = [option["label"] for option in model_options]
            current_label = st.session_state.get("selected_llm_label", labels[0])
            if current_label not in labels:
                current_label = labels[0]
            selected_label = st.selectbox(
                "Models",
                labels,
                index=labels.index(current_label),
                help="Choose which configured LLM the chatbot should use.",
            )
            _apply_selected_model(selected_label)
        else:
            st.warning("No usable LLM models found. Add at least one API key.")

        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF files to chat with",
            key=f"uploader_{st.session_state.uploader_key}",
        )

        if st.button("⚡ Process PDFs", use_container_width=True):
            if not uploaded_files:
                st.warning("Please upload at least one PDF file first.")
            else:
                process_uploaded_pdfs(uploaded_files)

        if st.session_state.indexed_files:
            st.markdown("---")
            count = len(st.session_state.indexed_files)
            st.markdown(f"**Indexed Documents ({count}):**")
            for filename in st.session_state.indexed_files:
                col_name, col_del = st.columns([5, 1])
                col_name.markdown(f"✅ {filename}")
                if col_del.button("🗑", key=f"del_{filename}",
                                  help=f"Remove {filename}"):
                    remove_file(filename)
                    st.rerun()

        if st.session_state.messages:
            st.markdown("---")
            st.markdown("**Export conversation:**")
            col_md, col_txt = st.columns(2)
            col_md.download_button(
                "Export .md",
                data=build_chat_export(st.session_state.messages, "md"),
                file_name="chat_history.md",
                mime="text/markdown",
                use_container_width=True,
            )
            col_txt.download_button(
                "Export .txt",
                data=build_chat_export(st.session_state.messages, "txt"),
                file_name="chat_history.txt",
                mime="text/plain",
                use_container_width=True,
            )

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑 Clear Chat", use_container_width=True):
                clear_chat()
                st.rerun()
        with col2:
            if st.button("🔄 Reset All", use_container_width=True):
                reset_session()
                st.rerun()


def render_header():
    """Render the page title and the top-right theme selector."""
    col_title, col_theme = st.columns([5, 1])
    with col_theme:
        mode = st.selectbox(
            "Theme",
            ["Default", "Light", "Dark"],
            key="theme_mode",
            label_visibility="collapsed",
        )
    apply_theme(mode)
    with col_title:
        st.title("💬 Chat with your PDFs")
        st.caption("Ask questions across all of your uploaded PDFs.")


def render_starter_questions():
    """Show clickable example questions before any conversation has started."""
    st.markdown("**Try asking:**")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, question in zip(cols, EXAMPLE_QUESTIONS):
        if col.button(question, use_container_width=True):
            st.session_state.pending_question = question
            st.rerun()


def handle_question(prompt: str):
    """Run a question through the RAG chain and render the exchange."""
    dismiss_pdf_viewer()
    st.session_state.pdf_viewer_stage = "cleanup"

    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.spinner("Searching your documents..."):
            result = answer_prompt(prompt)
            answer = result["answer"]
            source_items = extract_source_items(
                result["source_documents"], answer=answer
            )
            sources = format_sources(result["source_documents"])
        st.markdown(answer)
        if source_items:
            render_source_citations(
                source_items, f"live_{len(st.session_state.messages)}"
            )
        elif sources:
            st.caption(sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "source_items": source_items,
        }
    )


def answer_prompt(prompt: str) -> dict:
    """Route a question to page-targeted retrieval or the normal RAG chain.

    If the question references a specific page of an indexed PDF, the exact
    chunks for that page are fetched by metadata and answered directly;
    otherwise the conversational retrieval chain handles it. Page answers are
    also written to memory so follow-up questions stay context-aware.

    Args:
        prompt: The user's question.

    Returns:
        Dict with ``answer`` and ``source_documents``.
    """
    indexed_files = st.session_state.indexed_files
    ref_file, ref_page = parse_page_reference(prompt, indexed_files)
    if ref_file and ref_page:
        page_docs = get_page_documents(
            st.session_state.vector_store, ref_file, ref_page
        )
        if page_docs:
            result = answer_from_documents(
                prompt,
                page_docs,
                vector_store=st.session_state.vector_store,
                llm_provider=st.session_state.selected_llm_provider,
                llm_model=st.session_state.selected_llm_model,
            )
            try:
                st.session_state.memory.save_context(
                    {"question": prompt}, {"answer": result["answer"]}
                )
            except Exception:
                pass
            return result
        # No text on that page → fall through to normal retrieval.

    if is_multi_document_overview(prompt, len(indexed_files)):
        overview_docs = retrieve_balanced_documents(
            st.session_state.vector_store,
            prompt,
            per_file_k=4,
            global_k=4,
        )
        result = answer_from_documents(
            prompt,
            overview_docs,
            vector_store=st.session_state.vector_store,
            llm_provider=st.session_state.selected_llm_provider,
            llm_model=st.session_state.selected_llm_model,
        )
        try:
            st.session_state.memory.save_context(
                {"question": prompt}, {"answer": result["answer"]}
            )
        except Exception:
            pass
        return result

    return query_chain(st.session_state.chain, prompt, st.session_state.vector_store)


def render_chat():
    """Render the main chat area: history, empty state, input, and footer."""
    if not st.session_state.indexed_files:
        st.info(
            "👈 Upload PDF files in the sidebar and click "
            "'Process PDFs' to get started."
        )

    for index, message in enumerate(st.session_state.messages):
        avatar = USER_AVATAR if message["role"] == "user" else ASSISTANT_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if message.get("source_items"):
                render_source_citations(message["source_items"], f"msg_{index}")
            elif message.get("sources"):
                st.caption(message["sources"])

    chat_disabled = st.session_state.chain is None

    # Starter questions: only when PDFs are ready and no chat has begun.
    if not chat_disabled and not st.session_state.messages:
        render_starter_questions()

    typed = st.chat_input(
        "Ask a question about your PDFs..."
        if not chat_disabled
        else "Upload and process PDFs first...",
        disabled=chat_disabled,
    )

    # A question can come from the input box or from a starter-question click.
    prompt = typed or st.session_state.pending_question
    st.session_state.pending_question = None
    if prompt:
        handle_question(prompt)

    render_pdf_viewer_modal()

    st.markdown("---")
    st.caption(f"🤖 {APP_NAME}")


def main():
    """Application entry point."""
    initialise_session_state()
    render_sidebar()
    render_header()
    render_chat()


if __name__ == "__main__":
    main()
