# 📚 Multi-PDF ChatBot

A Streamlit web application that lets you upload multiple PDF documents and chat
across all of them using a Retrieval-Augmented Generation (RAG) pipeline. Answers
are grounded strictly in your uploaded documents and include source citations.

---

## What it does

- Upload one or more PDF files into a shared knowledge base
- Ask natural-language questions answered from the PDF content
- See which PDF each answer came from (source citation)
- Hold a context-aware conversation — follow-up questions understand prior turns
- Clear the chat or reset the whole session at any time

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Orchestration | LangChain |
| PDF parsing | pypdf / PyPDFLoader |
| Embeddings (local, no API key) | HuggingFace `all-MiniLM-L6-v2` |
| Vector store | ChromaDB (default), Pinecone (optional) |
| LLM | Google Gemini 2.0 Flash (default), OpenRouter (optional) |
| Config | python-dotenv |

## Prerequisites

- Python 3.10 or higher
- A free Google AI Studio API key — get one at https://aistudio.google.com
- Internet connection (for LLM calls and the one-time embedding-model download ~90 MB)

## Installation

```bash
# 1. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

> The first run downloads the HuggingFace embedding model (~90 MB). This is a
> one-time download; subsequent runs use the local cache.

## Configuration (.env setup)

Copy the example file and fill in your key:

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

Edit `.env`:

```env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_actual_key_here

VECTOR_STORE=chroma
```

To use OpenRouter instead of Gemini, set `LLM_PROVIDER=openrouter` and add your
`OPENROUTER_API_KEY` and `OPENROUTER_MODEL`. To use Pinecone instead of ChromaDB,
set `VECTOR_STORE=pinecone` and add your Pinecone credentials.

> **Never commit `.env`** — it is git-ignored. Only `.env.example` is committed.

## Running the app

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

## How to use

1. In the sidebar, click **Browse files** and select one or more PDFs.
2. Click **⚡ Process PDFs** — wait for the success message.
3. Type a question in the chat box at the bottom and press Enter.
4. Read the answer; the source PDF name appears beneath each response.
5. Ask follow-up questions — the bot remembers the conversation.
6. Use **🗑 Clear Chat** to reset the conversation (keeps your indexed PDFs) or
   **🔄 Reset All** to start over completely.

## Project structure

```
multi-pdf-chatbot/
├── app.py              # Streamlit UI + session orchestration
├── pdf_processor.py    # PDF loading, text extraction, chunking
├── vector_store.py     # ChromaDB embeddings + retriever
├── rag_chain.py        # LangChain RAG chain + memory
├── config.py           # All constants and env vars
├── utils.py            # Helper utilities (source formatting, validation)
├── requirements.txt    # Python dependencies
├── .env                # API keys (not committed)
├── .env.example        # Template for environment variables
├── .gitignore
├── README.md
└── chroma_db/          # Persisted vector store (auto-created)
```

## Known limitations

- **Single-user / localhost** — not designed for concurrent multi-user access.
- **Text-based PDFs only** — scanned image-only PDFs are not supported (no OCR).
- **Chat history is session-scoped** — it is lost on a browser refresh (the
  indexed vector store on disk, however, persists across restarts).
- **Gemini free-tier rate limits** apply (e.g. requests/minute and tokens/day).
