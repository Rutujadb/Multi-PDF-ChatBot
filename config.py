import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _read_secret(name: str, default: str = "") -> str:
    """Read an environment variable and treat template placeholders as unset."""
    raw = os.getenv(name, default).strip()
    if not raw:
        return ""
    lowered = raw.lower()
    if lowered.startswith("your_") and lowered.endswith("_here"):
        return ""
    return raw

# --- Chunking ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- Retrieval ---
# Number of chunks fetched per query. Higher gives broad/summary questions
# ("what topics are covered?") more context to synthesise from.
TOP_K_RESULTS = 8

# Maximum source cards shown after answer-aligned citation filtering.
CITATION_MAX_SOURCES = 4

# --- ChromaDB ---
CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "multi_pdf_store"

# --- Uploaded PDF storage (for source preview) ---
UPLOADED_PDF_DIR = Path("./uploaded_pdfs")

# --- SQLite chat memory (persistent conversation history) ---
CHAT_DB_PATH = Path(os.getenv("CHAT_DB_PATH", "./data/chat_memory.db"))

# --- Embedding model ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- LLM ---
GOOGLE_API_KEY = _read_secret("GOOGLE_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 1024
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.85"))
LLM_TOP_K = int(os.getenv("LLM_TOP_K", "40"))
LLM_REPETITION_PENALTY = float(os.getenv("LLM_REPETITION_PENALTY", "1.2"))
LLM_FREQUENCY_PENALTY = float(os.getenv("LLM_FREQUENCY_PENALTY", "0.3"))

OPENROUTER_API_KEY = _read_secret("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-2-9b-it:free")
OPENROUTER_HTTP_REFERER = os.getenv(
    "OPENROUTER_HTTP_REFERER", "http://localhost:5173"
)
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "Multi-PDF ChatBot")

GROQ_API_KEY = _read_secret("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

NVIDIA_API_KEY = _read_secret("NVIDIA_API_KEY")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")

_explicit_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
if _explicit_provider in ("gemini", "openrouter", "groq", "nvidia"):
    LLM_PROVIDER = _explicit_provider
elif OPENROUTER_API_KEY:
    LLM_PROVIDER = "openrouter"
elif GROQ_API_KEY:
    LLM_PROVIDER = "groq"
elif NVIDIA_API_KEY:
    LLM_PROVIDER = "nvidia"
elif GOOGLE_API_KEY:
    LLM_PROVIDER = "gemini"
else:
    LLM_PROVIDER = "openrouter"

if LLM_PROVIDER == "gemini" and not GOOGLE_API_KEY and OPENROUTER_API_KEY:
    LLM_PROVIDER = "openrouter"

if LLM_PROVIDER == "groq" and not GROQ_API_KEY and OPENROUTER_API_KEY:
    LLM_PROVIDER = "openrouter"

if LLM_PROVIDER == "nvidia" and not NVIDIA_API_KEY and OPENROUTER_API_KEY:
    LLM_PROVIDER = "openrouter"


def get_active_llm_name() -> str:
    """Return the model name for the configured LLM provider."""
    if LLM_PROVIDER == "openrouter":
        return OPENROUTER_MODEL
    if LLM_PROVIDER == "groq":
        return GROQ_MODEL
    if LLM_PROVIDER == "nvidia":
        return NVIDIA_MODEL
    return GEMINI_MODEL_NAME


def get_available_llm_options() -> list[dict[str, str]]:
    """Return the provider/model options that are usable with current keys."""
    options: list[dict[str, str]] = []
    provider_labels = {
        "openrouter": "OpenRouter",
        "groq": "Groq",
        "nvidia": "Nvidia",
        "gemini": "Gemini",
    }
    provider_models = {
        "openrouter": OPENROUTER_MODEL,
        "groq": GROQ_MODEL,
        "nvidia": NVIDIA_MODEL,
        "gemini": GEMINI_MODEL_NAME,
    }
    provider_keys = {
        "openrouter": OPENROUTER_API_KEY,
        "groq": GROQ_API_KEY,
        "nvidia": NVIDIA_API_KEY,
        "gemini": GOOGLE_API_KEY,
    }

    for provider in ("openrouter", "groq", "nvidia", "gemini"):
        if not provider_keys[provider]:
            continue
        model = provider_models[provider]
        options.append(
            {
                "provider": provider,
                "model": model,
                "label": f"{provider_labels[provider]} - {model}",
            }
        )
    return options


def get_default_llm_option() -> dict[str, str]:
    """Return the default UI selection for the active provider/model."""
    options = get_available_llm_options()
    active_model = get_active_llm_name()
    options_by_key = {
        (option["provider"], option["model"]): option for option in options
    }
    default = options_by_key.get((LLM_PROVIDER, active_model))
    if default is not None:
        return default
    if options:
        return options[0]
    return {
        "provider": LLM_PROVIDER,
        "model": active_model,
        "label": active_model,
    }

# --- Vector store ---
VECTOR_STORE = os.getenv("VECTOR_STORE", "chroma")
PINECONE_API_KEY = _read_secret("PINECONE_API_KEY")
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

# --- UI: context-aware suggested questions (generated after indexing) ---
SUGGESTED_QUESTION_COUNT = 4
SUGGESTED_QUESTION_RETRIEVAL_QUERY = (
    "main topics policies procedures requirements eligibility benefits "
    "rules how to apply overview summary"
)

# --- UI: app display name (shown in the footer) ---
APP_NAME = "Multi-PDF ChatBot"

# --- UI: classic Streamlit app URL (linked from React dashboard) ---
STREAMLIT_APP_URL = os.getenv(
    "STREAMLIT_APP_URL",
    "https://multi-pdf-chatbot-rb.streamlit.app/",
)

# Comma-separated deployed React origins for FastAPI CORS (e.g. Vercel URL).
_LOCAL_CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
)


def get_cors_origins() -> list[str]:
    """Return allowed browser origins for the FastAPI CORS middleware."""
    origins: list[str] = list(_LOCAL_CORS_ORIGINS)
    seen: set[str] = set(origins)
    extra = os.getenv("FRONTEND_ALLOWED_ORIGINS", "")
    for origin in extra.split(","):
        cleaned = origin.strip().rstrip("/")
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            origins.append(cleaned)
    return origins
