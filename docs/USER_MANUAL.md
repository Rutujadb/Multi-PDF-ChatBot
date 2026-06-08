# User Manual — Multi-PDF ChatBot

Step-by-step guide for using the app: upload PDFs, ask questions, read sources, and troubleshoot common issues.

**Audience:** End users (no coding required).  
**Technical details:** See [DESIGN.md](./DESIGN.md).

---

## 1. What this app does

Multi-PDF ChatBot lets you:

- Upload **one or more PDF files** into a shared knowledge base
- Ask **natural-language questions** answered from your PDF content only
- See **which PDF and page** each answer came from (source citations)
- **Click a source** to open a highlighted PDF preview (React UI)
- Continue a **conversation** — follow-up questions remember prior turns

> **Important:** The app works with **text-based PDFs** (you can select/copy text in the file). Scanned or image-only PDFs are **not supported** yet.

---

## 2. Choose your interface

| Interface | URL | Best for |
|-----------|-----|----------|
| **React UI (recommended)** | https://multi-pdf-chat-bot.vercel.app/ | Modern dashboard, source preview side panel |
| **Streamlit classic** | https://multi-pdf-chatbot-rb.streamlit.app/ | Simple sidebar upload and chat |

Both interfaces answer from the same RAG technology. The React UI connects to a separate API backend and offers a richer source viewer.

**API backend (React only):** https://multi-pdf-chatbot-y6nu.onrender.com  
Health check: https://multi-pdf-chatbot-y6nu.onrender.com/api/health

---

## 3. Before you start

### What you need

- PDF files with **selectable text** (not scanned images)
- A modern web browser (Chrome, Firefox, Edge, Safari)
- Internet connection (the AI model runs remotely)

### First-time delay (React UI)

The React app uses a free-tier API host that **sleeps after ~15 minutes of inactivity**. Your first action after idle may take **1–3 minutes** while the server wakes up. If upload or chat seems stuck, wait and try again.

---

## 4. React dashboard — step by step

### 4.1 Open the dashboard

1. Go to https://multi-pdf-chat-bot.vercel.app/
2. Click **Get Started** or open the dashboard directly: https://multi-pdf-chat-bot.vercel.app/dashboard

![Landing page](screenshots/react-landing.png)

### 4.2 Upload PDFs

1. In the **sidebar**, drag and drop PDF files or use the file picker
2. You can queue **multiple files** before processing
3. Click **Process PDFs**
4. Wait for indexing to finish — the sidebar shows each file with page and chunk counts

![Dashboard upload](screenshots/react-upload.png)

**Tips**

- Only **new filenames** are indexed. If you upload `report.pdf` twice, the second upload is skipped.
- To re-index an updated file, **rename it** (e.g. `report-v2.pdf`) before uploading.
- Invalid or empty files are skipped with a message.

### 4.3 Ask questions

1. Type your question in the chat input at the bottom
2. Press Enter or click Send
3. Read the answer — it is generated only from your uploaded PDFs
4. Below the answer, **Sources retrieved** appear as colored chips (filename + page)

![Chat with sources](screenshots/react-chat.png)

**Example questions**

| Question | When to use |
|----------|-------------|
| "Summarise the uploaded documents." | After uploading 2+ PDFs |
| "What is each PDF about?" | Overview across all files |
| "What are the key points in the cover letter?" | Specific document |
| "What is on page 3 of report.pdf?" | Page-targeted lookup |
| "What topics are covered?" | Broad content discovery |

Follow-up questions work naturally: "Can you elaborate on the second point?" remembers the conversation.

### 4.4 View a source (highlighted preview)

1. Click any **source chip** under an answer
2. A **right-side panel** opens (~50% of the screen) showing the PDF page
3. The relevant passage is **highlighted in yellow**
4. Use **Download** to save the annotated page as a PDF
5. Click **Close** or the X to dismiss the panel

![Source preview panel](screenshots/react-source-preview.png)

### 4.5 Session controls

| Button | What it does |
|--------|--------------|
| **Clear chat** | Removes chat messages only — **indexed PDFs stay** |
| **Reset session** | Wipes chat, indexed PDFs, and vector data — start fresh |

Use **Clear chat** when you want a new topic but keep your documents. Use **Reset session** when you want to upload a completely new set of files.

---

## 5. Streamlit classic — step by step

### 5.1 Open the app

Go to https://multi-pdf-chatbot-rb.streamlit.app/

### 5.2 Upload and process

1. Use the **sidebar** file uploader to select one or more PDFs
2. Click **Process PDFs**
3. Wait for the success message and indexed file list

![Streamlit sidebar upload](screenshots/streamlit-upload.png)

### 5.3 Chat and citations

1. Type a question in the chat input
2. Read the answer and source citations below it
3. **Click a citation** to preview the highlighted page (shown as a PNG image)

![Streamlit chat with citations](screenshots/streamlit-chat.png)

### 5.4 Session controls

| Button | What it does |
|--------|--------------|
| **Clear Chat** | Clears messages; keeps indexed PDFs |
| **Reset All** | Clears everything and starts over |

---

## 6. React vs Streamlit — quick comparison

| Feature | React UI | Streamlit |
|---------|----------|-----------|
| Landing page | Yes | No |
| Source preview | Side panel + download | Inline PNG preview |
| Multi-PDF balanced answers | Yes | Yes |
| Session persistence after refresh | No (new session) | No |
| Cold start delay | Yes (API on Render) | Minimal |
| Recommended for demos | Yes | Alternate / backup |

---

## 7. Understanding answers and sources

### Grounded answers

The assistant answers **only from your uploaded PDFs**. If the documents do not contain relevant information, you may see:

> "I don't have enough information in the uploaded documents to answer this."

This is expected — the app will not invent facts from outside knowledge.

### Source chips

Each source shows:

- **File name** — which PDF the passage came from
- **Page number** — which page (1-based, human-readable)

Chip colors are for visual distinction only — they do not indicate processing status.

### Multi-PDF questions

For questions like "Summarise each document" or "What is each PDF about?", upload **all PDFs first**, then ask. The app retrieves context from every indexed file before answering.

---

## 8. Tips for best results

1. **Upload everything before broad questions** — especially for summaries across multiple files.
2. **Use specific filenames** when you have many PDFs: "What does `policy-2024.pdf` say about refunds?"
3. **One clear topic per question** often works better than very long compound questions.
4. **Wait for indexing to finish** before chatting — the Process button must complete successfully.
5. **Rename files to re-index** — same filename uploads are skipped by design.
6. **Be patient on first load** — the React API may need time to wake from sleep.

---

## 9. Troubleshooting

| Problem | Likely cause | What to do |
|---------|--------------|------------|
| Upload or chat hangs (React) | API cold start on Render | Wait 1–3 minutes; check [health endpoint](https://multi-pdf-chatbot-y6nu.onrender.com/api/health); retry |
| "No readable text found" | Scanned/image-only PDF | Use a text-based PDF or OCR tool first |
| Answer mentions only one PDF | Asked before all files indexed | Process all PDFs, then ask "What is each PDF about?" |
| Source preview empty or error | Server restarted; PDFs cleared from disk | Click **Reset session**, re-upload all PDFs |
| "I don't have enough information…" | Topic not in documents | Rephrase; confirm the right PDF is indexed |
| Same file not re-indexed | Filename dedup | Rename the file and upload again |
| Chat history gone after refresh | Session-scoped by design | Re-ask your question; PDFs may still be indexed if session restored |
| Slow first response | Embedding model + LLM loading | Normal on cold start; subsequent questions are faster |

---

## 10. FAQ

**Do I need an account?**  
No. The public demos work without signing up.

**Are my PDFs private?**  
On hosted demos, files are stored on the server for your session. Treat public instances as **not suitable for highly confidential documents**. Data may be cleared when the server restarts or you reset the session.

**Which UI should I use?**  
Use the **React UI** for the best experience (source panel, download). Use **Streamlit** as a simpler alternative.

**Does chat history persist?**  
No. Refreshing the browser starts a new session. Indexed PDFs may persist on disk briefly, but do not rely on this on free hosting.

**Can I use Word or Excel files?**  
Not currently — only `.pdf` files are supported.

**Why was my file skipped?**  
Either it was already indexed (same filename) or it failed validation (not a PDF, empty, or no extractable text).

---

## 11. Run locally (optional)

If you want to run the app on your own machine:

```bash
# Clone the repo, create venv, install deps
pip install -r requirements.txt
cp .env.example .env   # add your OPENROUTER_API_KEY

# React + API (recommended)
python run_dev.py
# → http://localhost:5173 (UI) and http://localhost:8000 (API)

# Streamlit only
streamlit run app.py
# → http://localhost:8501
```

Full setup details: [README.md](../README.md).

---

## 12. Architecture at a glance (for curious users)

If you want to see how data flows through the system:

| UI | Flowchart |
|----|-----------|
| React + FastAPI | ![React architecture](screenshots/react-flowchart.png) |
| Streamlit | ![Streamlit architecture](screenshots/streamlit-flowchart.png) |

For technical depth, see [DESIGN.md](./DESIGN.md).

---

## 13. Links and support

| Resource | URL |
|----------|-----|
| React live demo | https://multi-pdf-chat-bot.vercel.app/ |
| Streamlit live demo | https://multi-pdf-chatbot-rb.streamlit.app/ |
| GitHub repository | https://github.com/Rutujadb/Multi-PDF-ChatBot |
| Report an issue | https://github.com/Rutujadb/Multi-PDF-ChatBot/issues |

---

*Last updated for the React + FastAPI + Streamlit PoC deployment.*
