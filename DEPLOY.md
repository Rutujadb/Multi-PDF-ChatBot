# Deploy React UI + FastAPI (free tier)

Split deployment:

| Part | Host | Cost |
|------|------|------|
| **Backend** | [Render](https://render.com) Web Service | Free |
| **Frontend** | [Vercel](https://vercel.com) **or** [Cloudflare Pages](https://pages.cloudflare.com) | Free |

Deploy the **backend first**, then the frontend (you need the API URL for `VITE_API_BASE_URL`).

---

## Prerequisites

1. GitHub repo pushed and up to date (`Rutujadb/Multi-PDF-ChatBot`)
2. OpenRouter API key (or Gemini key if using `LLM_PROVIDER=gemini`)
3. Code includes production config:
   - `VITE_API_BASE_URL` in `frontend/src/api/client.js`
   - `FRONTEND_ALLOWED_ORIGINS` CORS in `config.py` / `api.py`

---

## Part 1 — Deploy FastAPI on Render

### 1. Create Render account

1. Go to [render.com](https://render.com) → **Get Started** → sign in with **GitHub**
2. Authorize Render to access your repository

### 2. Create Web Service

1. **Dashboard** → **New +** → **Web Service**
2. Connect **Multi-PDF-ChatBot** repository
3. Configure:

| Field | Value |
|-------|--------|
| **Name** | `multi-pdf-api` (or your choice) |
| **Region** | Closest to you |
| **Branch** | `main` |
| **Root Directory** | *(leave empty — repo root)* |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r requirements-api.txt` |
| **Start Command** | `python start.py` |
| **Instance Type** | **Free** |

> Optional: Render can read `render.yaml` from the repo if you use **Blueprint** deploy.

### 3. Environment variables (Render → Environment)

| Key | Value | Secret? |
|-----|--------|---------|
| `OPENROUTER_API_KEY` | your key | Yes |
| `LLM_PROVIDER` | `openrouter` | No |
| `OPENROUTER_MODEL` | e.g. `google/gemma-2-9b-it:free` | No |
| `VECTOR_STORE` | `chroma` | No |
| `STREAMLIT_APP_URL` | `https://multi-pdf-chatbot-rb.streamlit.app/` | No |
| `FRONTEND_ALLOWED_ORIGINS` | *(set after frontend deploy — step 5)* | No |

Click **Save Changes** → Render builds and deploys.

### 4. Verify API

Your API URL will look like:

```
https://multi-pdf-api.onrender.com
```

Test health:

```
https://multi-pdf-api.onrender.com/api/health
```

Expected: `{"status":"ok","app":"Multi-PDF ChatBot"}`

**First request after idle:** free tier sleeps ~15 min; first load can take **2–5 minutes** while dependencies and the embedding model load.

### 5. Add CORS after you know the frontend URL

Once Vercel/Pages gives you a URL, set on Render:

```
FRONTEND_ALLOWED_ORIGINS=https://your-app.vercel.app
```

Multiple origins (comma-separated, no trailing slash):

```
FRONTEND_ALLOWED_ORIGINS=https://your-app.vercel.app,https://your-app.pages.dev
```

Redeploy or wait for env reload.

### Render free-tier limits

- **512 MB RAM** — may OOM on first embed; retry or upgrade if build fails
- **Ephemeral disk** — `chroma_db/` and `uploaded_pdfs/` reset on restart/redeploy
- **Cold starts** — slow first request after sleep

---

## Part 2A — Deploy frontend on Vercel

### 1. Create Vercel account

1. [vercel.com](https://vercel.com) → **Sign Up** → **Continue with GitHub**
2. Import **Multi-PDF-ChatBot** repository

### 2. Configure project

| Field | Value |
|-------|--------|
| **Framework Preset** | Vite |
| **Root Directory** | `frontend` |
| **Build Command** | `npm run build` |
| **Output Directory** | `dist` |
| **Install Command** | `npm install` |

### 3. Environment variables (Vercel → Settings → Environment Variables)

| Key | Value | Environments |
|-----|--------|--------------|
| `VITE_API_BASE_URL` | `https://multi-pdf-api.onrender.com` | Production (and Preview if you want) |

Use your actual Render URL. **No trailing slash.**

### 4. Deploy

Click **Deploy**. Vercel builds and gives you a URL, e.g.:

```
https://multi-pdf-chatbot.vercel.app
```

`frontend/vercel.json` handles React Router (`/dashboard`).

### 5. Update Render CORS

On Render, set:

```
FRONTEND_ALLOWED_ORIGINS=https://multi-pdf-chatbot.vercel.app
```

### 6. Test

1. Open `https://your-app.vercel.app`
2. Go to **Dashboard**
3. Upload PDFs → **Process PDFs** (wait for cold start if API was sleeping)
4. Ask a question → check sources and preview panel

---

## Part 2B — Deploy frontend on Cloudflare Pages (alternative)

### 1. Cloudflare account

1. [dash.cloudflare.com](https://dash.cloudflare.com) → **Workers & Pages** → **Create** → **Pages** → **Connect to Git**

### 2. Build settings

| Field | Value |
|-------|--------|
| **Project name** | `multi-pdf-chatbot` |
| **Production branch** | `main` |
| **Root directory** | `frontend` |
| **Build command** | `npm run build` |
| **Build output** | `dist` |

### 3. Environment variable

**Settings → Environment variables → Production:**

| Key | Value |
|-----|--------|
| `VITE_API_BASE_URL` | `https://multi-pdf-api.onrender.com` |

### 4. Deploy & CORS

URL example: `https://multi-pdf-chatbot.pages.dev`

Add to Render `FRONTEND_ALLOWED_ORIGINS`:

```
FRONTEND_ALLOWED_ORIGINS=https://multi-pdf-chatbot.pages.dev
```

`frontend/public/_redirects` enables SPA routing on Pages.

---

## End-to-end checklist

- [ ] Render deploy succeeded; `/api/health` returns 200
- [ ] `OPENROUTER_API_KEY` set on Render
- [ ] Frontend deployed with `VITE_API_BASE_URL` = Render URL
- [ ] `FRONTEND_ALLOWED_ORIGINS` on Render includes frontend URL
- [ ] Upload + process + chat works on live site
- [ ] Source preview opens (needs PDF saved on server after upload)
- [ ] README live demo link updated

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| CORS error in browser | Add exact frontend URL to `FRONTEND_ALLOWED_ORIGINS` on Render (no trailing `/`) |
| API request goes to wrong host | Check `VITE_API_BASE_URL` on Vercel/Pages; redeploy frontend after changing |
| 502 / timeout on first request | Render waking up; wait 2–5 min and retry |
| Build fails on Render (memory) | Use `requirements-api.txt`; retry build; consider Hugging Face Docker Space or a small VM |
| Port scan timeout / no open ports | Set start command to `python start.py`; use `requirements-api.txt`; health check path `/api/health` |
| Indexed PDFs gone after restart | Expected on Render free — ephemeral disk |
| `/dashboard` 404 on refresh | Ensure `vercel.json` or `_redirects` is deployed |

---

## Local production test (before cloud deploy)

```powershell
# Terminal 1 — API
cd "d:\Yash Technology Office Work\Multi-pdf Chatbot"
$env:FRONTEND_ALLOWED_ORIGINS="http://localhost:4173"
uvicorn api:app --host 0.0.0.0 --port 8000

# Terminal 2 — frontend production build
cd frontend
$env:VITE_API_BASE_URL="http://localhost:8000"
npm run build
npm run preview
```

Open `http://localhost:4173/dashboard` and test upload + chat.
