import os
from dotenv import load_dotenv

load_dotenv()

# --- Chunking ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- Retrieval ---
# Number of chunks fetched per query. Higher gives broad/summary questions
# ("what topics are covered?") more context to synthesise from.
TOP_K_RESULTS = 6

# --- ChromaDB ---
CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "multi_pdf_store"

# --- Embedding model ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- LLM ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL_NAME = "gemini-2.0-flash"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 1024

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct")

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
