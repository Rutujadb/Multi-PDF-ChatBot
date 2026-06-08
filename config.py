import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Chunking ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- Retrieval ---
# Number of chunks fetched per query. Higher gives broad/summary questions
# ("what topics are covered?") more context to synthesise from.
TOP_K_RESULTS = 6

# Maximum source cards shown after answer-aligned citation filtering.
CITATION_MAX_SOURCES = 3

# --- ChromaDB ---
CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "multi_pdf_store"

# --- Uploaded PDF storage (for source preview) ---
UPLOADED_PDF_DIR = Path("./uploaded_pdfs")

# --- Embedding model ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- LLM ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL_NAME = "gemini-2.0-flash"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 1024

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct")
OPENROUTER_HTTP_REFERER = os.getenv(
    "OPENROUTER_HTTP_REFERER", "http://localhost:5173"
)
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "Multi-PDF ChatBot")

_explicit_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
if _explicit_provider in ("gemini", "openrouter"):
    LLM_PROVIDER = _explicit_provider
elif OPENROUTER_API_KEY:
    LLM_PROVIDER = "openrouter"
elif GOOGLE_API_KEY:
    LLM_PROVIDER = "gemini"
else:
    LLM_PROVIDER = "openrouter"

if LLM_PROVIDER == "gemini" and not GOOGLE_API_KEY and OPENROUTER_API_KEY:
    LLM_PROVIDER = "openrouter"


def get_active_llm_name() -> str:
    """Return the model name for the configured LLM provider."""
    if LLM_PROVIDER == "openrouter":
        return OPENROUTER_MODEL
    return GEMINI_MODEL_NAME

# --- Vector store ---
VECTOR_STORE = os.getenv("VECTOR_STORE", "chroma")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "multi-pdf-chatbot")

# --- Prompt template ---
SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions
about the user's uploaded PDF documents. Base your answer only on the provided
context below. You may summarise and synthesise across the context - for
example, to describe the topics, themes, or main points covered. Only if the
context contains nothing relevant to the question, reply exactly: "I don't have
enough information in the uploaded documents to answer this." Do not use any
outside knowledge.

Context:
{context}

Chat History:
{chat_history}

Question: {question}
Answer:"""

# --- UI: starter/example questions shown when no chat has begun yet ---
EXAMPLE_QUESTIONS = [
    "Summarise the uploaded documents.",
    "What are the key points?",
    "What topics are covered?",
]

# --- UI: app display name (shown in the footer) ---
APP_NAME = "Multi-PDF ChatBot"

# --- UI: classic Streamlit app URL (linked from React dashboard) ---
STREAMLIT_APP_URL = os.getenv(
    "STREAMLIT_APP_URL",
    "https://multi-pdf-chatbot-rb.streamlit.app/",
)
