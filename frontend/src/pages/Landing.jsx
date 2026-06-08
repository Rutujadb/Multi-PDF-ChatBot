import { Link } from 'react-router-dom'
import Logo from '../components/Logo.jsx'
import { GITHUB_REPO_URL, STREAMLIT_APP_URL } from '../config.js'

const features = [
  {
    bg: 'bg-brand-50 hover:bg-brand-100',
    stroke: '#3B82F6',
    title: 'Multi-PDF upload',
    text: 'Drop multiple files at once. Each chunk is tagged with source + page so citations stay accurate.',
    tag: 'FR-PDF-01 → 05',
    tagClass: 'text-brand-600',
  },
  {
    bg: 'bg-emerald2-50 hover:bg-emerald2-100',
    stroke: '#10B981',
    title: 'Semantic retrieval',
    text: 'Cosine-similarity search over ChromaDB returns the top chunks - across every file, not just one.',
    tag: 'FR-VS-01 → 03',
    tagClass: 'text-emerald2-600',
  },
  {
    bg: 'bg-amber2-50 hover:bg-amber2-100',
    stroke: '#F59E0B',
    title: 'Conversation memory',
    text: 'Follow-up questions stay context-aware. The chain condenses history + new query into a standalone search.',
    tag: 'FR-MEM-01 → 04',
    tagClass: 'text-amber2-600',
  },
  {
    bg: 'bg-muted hover:bg-line',
    stroke: '#111827',
    title: 'Source citations',
    text: 'Every answer shows which PDF (and page) it came from. The model is told to say "I don\'t know" if context is missing.',
    tag: 'FR-UI-03 · FR-RAG-04',
    tagClass: 'text-ink/60',
  },
  {
    bg: 'bg-brand-50 hover:bg-brand-100',
    stroke: '#3B82F6',
    title: 'Persistent store',
    text: 'ChromaDB writes to disk. Restart the app and your index is still there - no re-upload needed.',
    tag: 'FR-VS-02',
    tagClass: 'text-brand-600',
  },
  {
    bg: 'bg-emerald2-50 hover:bg-emerald2-100',
    stroke: '#10B981',
    title: 'Swappable backends',
    text: 'Flip an env var to use Pinecone instead of Chroma, or Gemini instead of OpenRouter. Same chain interface.',
    tag: 'FR-VS-04 · FR-RAG-03',
    tagClass: 'text-emerald2-600',
  },
]

const faqs = [
  {
    q: 'Do I need a paid API key?',
    a: 'Mostly no for embeddings and storage. The default stack uses OpenRouter (free-tier models available) for answers and HuggingFace embeddings that run locally with no key. ChromaDB is open-source and stores everything on your disk.',
    open: true,
  },
  {
    q: 'What happens when I refresh the browser?',
    a: 'Chat history is session-scoped and will reset. Your indexed PDFs persist - ChromaDB writes to ./chroma_db/ so they\'re loaded automatically on next start.',
  },
  {
    q: 'Will it work with scanned PDFs?',
    a: 'Not currently. The pipeline expects selectable text. Image-only PDFs are detected on upload and skipped with a clear warning - OCR is on the roadmap.',
  },
  {
    q: 'How do I swap to Pinecone?',
    a: 'Set VECTOR_STORE=pinecone in your .env, plus index name + API key. Core logic doesn\'t change - that\'s the maintainability requirement from the SRS.',
  },
]

export default function Landing() {
  return (
    <div className="bg-paper text-ink antialiased min-h-screen">
      <header className="border-b border-line">
        <div className="max-w-7xl mx-auto px-6 lg:px-10 h-20 flex items-center justify-between">
          <Logo />
          <nav className="hidden md:flex items-center gap-8 text-sm font-medium">
            <a href="#how" className="hover:text-brand-500 transition-colors">How it works</a>
            <a href="#features" className="hover:text-brand-500 transition-colors">Features</a>
            <a href="#stack" className="hover:text-brand-500 transition-colors">Stack</a>
            <a href="#faq" className="hover:text-brand-500 transition-colors">FAQ</a>
          </nav>
          <div className="flex items-center gap-3">
            <a
              href={GITHUB_REPO_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex h-11 px-4 items-center gap-2 rounded-md border-2 border-ink text-ink text-sm font-semibold btn-primary"
            >
              GitHub
            </a>
            <Link to="/dashboard" className="inline-flex h-11 px-5 items-center rounded-md bg-ink text-paper text-sm font-semibold btn-primary">
              Open app →
            </Link>
          </div>
        </div>
      </header>

      <section className="relative overflow-hidden bg-brand-500 text-paper">
        <div className="absolute inset-0 grid-bg pointer-events-none" />
        <div className="absolute -top-32 -right-32 w-[520px] h-[520px] rounded-full bg-paper/10" />
        <div className="absolute top-40 right-40 w-40 h-40 rounded-full bg-amber2-500" />
        <div className="absolute -bottom-24 -left-24 w-[420px] h-[420px] rotate-12 bg-ink/15 rounded-3xl" />
        <div className="absolute bottom-20 left-[36%] w-24 h-24 rotate-45 bg-emerald2-500" />

        <div className="relative max-w-7xl mx-auto px-6 lg:px-10 pt-20 pb-28 grid lg:grid-cols-12 gap-10 items-center">
          <div className="lg:col-span-7">
            <div className="inline-flex items-center gap-2 bg-ink text-paper px-3 h-8 rounded-md label">
              <span className="w-2 h-2 bg-emerald2-500 rounded-sm" />
              Document RAG · self-hosted
            </div>
            <h1 className="mt-6 h-display font-black text-[88px] sm:text-[112px] lg:text-[148px]">
              Chat<br />with<br />your{' '}
              <span className="bg-ink text-paper px-3 inline-block rounded-md">PDFs.</span>
            </h1>
            <p className="mt-8 max-w-xl text-lg font-medium text-paper/90 leading-relaxed">
              Drop in a stack of documents. Ask anything. Get grounded answers with citations - powered by a local vector store and OpenRouter.
            </p>
            <div className="mt-10 flex flex-wrap gap-3">
              <Link to="/dashboard" className="inline-flex h-16 px-8 items-center rounded-md bg-paper text-ink font-bold text-lg btn-primary">
                Open the dashboard
                <svg className="ml-3" width="20" height="20" viewBox="0 0 24 24" fill="none">
                  <path d="M5 12h14M13 6l6 6-6 6" stroke="currentColor" strokeWidth="2.5" strokeLinecap="square" strokeLinejoin="round" />
                </svg>
              </Link>
              <a href="#how" className="inline-flex h-16 px-8 items-center rounded-md border-4 border-paper text-paper font-bold text-lg btn-outline">
                See how it works
              </a>
            </div>
            <div className="mt-12 flex flex-wrap items-center gap-x-10 gap-y-4 text-sm font-medium text-paper/80">
              <div className="flex items-center gap-2"><span className="w-2 h-2 bg-paper rounded-sm" /> Local embeddings · no API key</div>
              <div className="flex items-center gap-2"><span className="w-2 h-2 bg-paper rounded-sm" /> Persistent ChromaDB</div>
              <div className="flex items-center gap-2"><span className="w-2 h-2 bg-paper rounded-sm" /> Source-cited answers</div>
            </div>
          </div>

          <div className="lg:col-span-5">
            <div className="relative">
              <div className="absolute -top-4 -left-4 w-24 h-24 bg-amber2-500 rounded-md" />
              <div className="absolute -bottom-6 -right-6 w-32 h-32 bg-emerald2-500 rounded-full" />
              <div className="relative bg-paper text-ink rounded-lg p-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="w-3 h-3 bg-ink rounded-sm" />
                    <span className="font-bold text-sm">Chat session</span>
                  </div>
                  <span className="label text-ink/60">live</span>
                </div>
                <div className="mt-5 space-y-4">
                  <div className="flex justify-end">
                    <div className="max-w-[80%] bg-ink text-paper rounded-md p-3 text-sm">
                      Summarise the Q3 risks across both filings.
                    </div>
                  </div>
                  <div className="flex gap-3">
                    <span className="w-8 h-8 bg-brand-500 rounded-md grid place-items-center text-paper font-bold text-sm shrink-0">A</span>
                    <div className="flex-1">
                      <div className="bg-muted rounded-md p-3 text-sm leading-relaxed">
                        Three risks repeat across the filings: <strong>supply concentration</strong>, FX exposure in EMEA, and a renewal cliff on a top-5 contract.
                      </div>
                      <div className="mt-2 flex flex-wrap gap-2">
                        <span className="inline-flex items-center gap-1 h-6 px-2 bg-brand-50 text-brand-600 text-xs font-semibold rounded-md">
                          <span className="w-1.5 h-1.5 bg-brand-500 rounded-sm" /> 10-Q_2024.pdf · p.14
                        </span>
                        <span className="inline-flex items-center gap-1 h-6 px-2 bg-emerald2-50 text-emerald2-600 text-xs font-semibold rounded-md">
                          <span className="w-1.5 h-1.5 bg-emerald2-500 rounded-sm" /> annual_report.pdf · p.41
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="border-y border-line bg-paper">
        <div className="max-w-7xl mx-auto px-6 lg:px-10 grid grid-cols-2 lg:grid-cols-4">
          {[
            ['384', 'embedding dims', 'text-brand-500'],
            ['≤ 15s', 'query-to-answer', 'text-emerald2-500'],
            ['10k', 'chunks supported', 'text-amber2-500'],
            ['$0', 'free-tier ready', 'text-ink'],
          ].map(([value, label, color], i) => (
            <div key={label} className={`py-10 ${i === 0 ? 'pr-6 border-r border-line' : i === 3 ? 'pl-6' : 'px-6 border-r border-line'}`}>
              <div className={`text-5xl font-extrabold h-tight ${color}`}>{value}</div>
              <div className="mt-2 label text-ink/60">{label}</div>
            </div>
          ))}
        </div>
      </section>

      <section id="how" className="relative bg-ink text-paper overflow-hidden">
        <div className="absolute -top-16 right-10 w-56 h-56 rotate-12 bg-brand-500/30 rounded-3xl pointer-events-none" />
        <div className="relative max-w-7xl mx-auto px-6 lg:px-10 py-24">
          <div className="flex flex-col lg:flex-row lg:items-end lg:justify-between gap-6">
            <div>
              <div className="label text-paper/60">How it works</div>
              <h2 className="mt-3 h-display font-black text-6xl lg:text-7xl">Two phases.<br />One pipeline.</h2>
            </div>
            <p className="max-w-md text-paper/80 text-lg">
              A retrieval-augmented generation flow. Index your PDFs locally; query them through a conversational chain that cites the source.
            </p>
          </div>
          <div className="mt-16 grid lg:grid-cols-2 gap-6">
            <div className="bg-paper text-ink rounded-lg p-8 lg:p-10 card-hover relative overflow-hidden">
              <div className="absolute -right-10 -top-10 w-32 h-32 rounded-full bg-brand-100" />
              <div className="relative flex items-center gap-4">
                <span className="w-12 h-12 grid place-items-center rounded-md bg-brand-500 text-paper font-extrabold text-xl">1</span>
                <div className="label text-brand-500">Indexing</div>
              </div>
              <h3 className="relative mt-6 font-extrabold text-3xl h-tight">Drop a PDF.<br />We chunk + embed.</h3>
              <ul className="relative mt-8 space-y-3 text-[15px]">
                {['PyPDFLoader reads each page in order.', 'RecursiveCharacterTextSplitter - 500 char chunks, 50 char overlap.', 'all-MiniLM-L6-v2 generates 384-dim vectors locally.', 'Written to ChromaDB with source + page metadata.'].map((item) => (
                  <li key={item} className="flex items-start gap-3">
                    <span className="mt-1.5 w-2 h-2 bg-brand-500 rounded-sm shrink-0" />
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>
            <div className="bg-paper text-ink rounded-lg p-8 lg:p-10 card-hover relative overflow-hidden">
              <div className="absolute -right-10 -top-10 w-32 h-32 rounded-full bg-emerald2-100" />
              <div className="relative flex items-center gap-4">
                <span className="w-12 h-12 grid place-items-center rounded-md bg-emerald2-500 text-paper font-extrabold text-xl">2</span>
                <div className="label text-emerald2-600">Query</div>
              </div>
              <h3 className="relative mt-6 font-extrabold text-3xl h-tight">Ask a question.<br />We retrieve + ground.</h3>
              <ul className="relative mt-8 space-y-3 text-[15px]">
                {['Chat history + question condensed to a standalone query.', 'Top chunks retrieved by cosine similarity.', 'OpenRouter generates the answer, grounded in context.', 'Sources shown beneath each response.'].map((item) => (
                  <li key={item} className="flex items-start gap-3">
                    <span className="mt-1.5 w-2 h-2 bg-emerald2-500 rounded-sm shrink-0" />
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section id="features" className="bg-paper">
        <div className="max-w-7xl mx-auto px-6 lg:px-10 py-24">
          <div className="grid lg:grid-cols-3 gap-6 items-end">
            <div className="lg:col-span-2">
              <div className="label text-ink/60">Capabilities</div>
              <h2 className="mt-3 h-display font-black text-6xl lg:text-7xl">Everything the<br />User asks for.</h2>
            </div>
            <p className="text-ink/70 text-lg">Six functional pillars - built from open-source pieces, no paid infra required.</p>
          </div>
          <div className="mt-14 grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
            {features.map((f) => (
              <article key={f.title} className={`group ${f.bg} rounded-lg p-8 card-hover`}>
                <div className="w-14 h-14 rounded-full bg-paper grid place-items-center icon-pop">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke={f.stroke} strokeWidth="2.5">
                    <path d="M12 4v12m0 0l-5-5m5 5l5-5M4 20h16" />
                  </svg>
                </div>
                <h3 className="mt-6 font-extrabold text-2xl h-tight">{f.title}</h3>
                <p className="mt-3 text-ink/70">{f.text}</p>
                <div className={`mt-5 label ${f.tagClass}`}>{f.tag}</div>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section id="stack" className="relative bg-emerald2-500 text-paper overflow-hidden">
        <div className="relative max-w-7xl mx-auto px-6 lg:px-10 py-24 grid lg:grid-cols-12 gap-10">
          <div className="lg:col-span-5">
            <div className="label text-paper/70">The stack</div>
            <h2 className="mt-3 h-display font-black text-6xl lg:text-7xl">Free<br />tier,<br />real<br />RAG.</h2>
            <p className="mt-6 text-paper/90 text-lg max-w-md">
              Production-grade patterns with zero billing surprises. React + Tailwind UI with a FastAPI backend - Streamlit classic UI still available.
            </p>
          </div>
          <div className="lg:col-span-7 grid sm:grid-cols-2 gap-4">
            {[
              ['UI', 'React + Tailwind', 'Landing + dashboard · Streamlit alt'],
              ['API', 'FastAPI', 'Wraps LangChain pipeline'],
              ['Embeddings', 'HuggingFace', 'all-MiniLM-L6-v2 · local'],
              ['Vector store', 'ChromaDB', 'Pinecone optional'],
            ].map(([label, title, mono]) => (
              <div key={title} className="bg-paper text-ink rounded-lg p-6">
                <div className="label text-emerald2-600">{label}</div>
                <div className="mt-2 font-extrabold text-2xl h-tight">{title}</div>
                <div className="mt-2 font-mono text-xs text-ink/60">{mono}</div>
              </div>
            ))}
            <div className="bg-paper text-ink rounded-lg p-6 sm:col-span-2">
              <div className="label text-brand-600">LLM</div>
              <div className="mt-2 font-extrabold text-3xl h-tight flex items-center gap-3">
                OpenRouter
                <span className="text-sm font-semibold text-ink/50">· free-tier models</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section id="faq" className="bg-paper">
        <div className="max-w-5xl mx-auto px-6 lg:px-10 py-24">
          <div className="label text-ink/60">FAQ</div>
          <h2 className="mt-3 h-display font-black text-6xl">Common questions.</h2>
          <div className="mt-12 border-y-2 border-ink">
            {faqs.map((item) => (
              <details key={item.q} className="group border-b-2 border-ink py-6" open={item.open}>
                <summary className="cursor-pointer list-none flex items-center justify-between">
                  <h3 className="font-extrabold text-2xl h-tight">{item.q}</h3>
                  <span className="w-10 h-10 rounded-md bg-muted grid place-items-center font-black text-xl group-open:bg-ink group-open:text-paper transition-colors">+</span>
                </summary>
                <p className="mt-4 text-ink/70 text-lg max-w-3xl">{item.a}</p>
              </details>
            ))}
          </div>
        </div>
      </section>

      <section className="relative bg-amber2-500 overflow-hidden">
        <div className="relative max-w-7xl mx-auto px-6 lg:px-10 py-24 grid lg:grid-cols-12 gap-10 items-center">
          <div className="lg:col-span-8">
            <div className="label text-ink/70">Ready when you are</div>
            <h2 className="mt-3 h-display font-black text-6xl lg:text-8xl text-ink">Open the<br />dashboard.</h2>
            <p className="mt-6 text-ink/80 text-lg max-w-xl">Upload a few PDFs, hit process, and start asking questions. No sign-up. No billing.</p>
          </div>
          <div className="lg:col-span-4 flex lg:justify-end">
            <Link to="/dashboard" className="inline-flex h-20 px-10 items-center rounded-md bg-ink text-paper font-bold text-xl btn-primary">
              Launch app
              <svg className="ml-4" width="24" height="24" viewBox="0 0 24 24" fill="none">
                <path d="M5 12h14M13 6l6 6-6 6" stroke="currentColor" strokeWidth="2.5" strokeLinecap="square" strokeLinejoin="round" />
              </svg>
            </Link>
          </div>
        </div>
      </section>

      <footer className="bg-ink text-paper">
        <div className="max-w-7xl mx-auto px-6 lg:px-10 py-16 grid md:grid-cols-12 gap-10">
          <div className="md:col-span-5">
            <Logo inverted />
            <p className="mt-5 text-paper/60 max-w-sm">Local-first RAG on your uploaded documents. React UI + FastAPI backend, with Streamlit as a classic alternative.</p>
          </div>
          <div className="md:col-span-2">
            <div className="label text-paper/50">Product</div>
            <ul className="mt-4 space-y-2 text-sm font-medium">
              <li><a className="hover:text-brand-500" href="#how">How it works</a></li>
              <li><a className="hover:text-brand-500" href="#features">Features</a></li>
              <li><Link className="hover:text-brand-500" to="/dashboard">Dashboard</Link></li>
            </ul>
          </div>
          <div className="md:col-span-3">
            <div className="label text-paper/50">Alternatives</div>
            <ul className="mt-4 space-y-2 text-sm font-medium">
              <li>
                <a className="hover:text-brand-500" href={STREAMLIT_APP_URL} target="_blank" rel="noopener noreferrer">
                  Streamlit classic UI
                </a>
              </li>
            </ul>
          </div>
        </div>
        <div className="border-t border-paper/10">
          <div className="max-w-7xl mx-auto px-6 lg:px-10 py-6 flex flex-wrap items-center justify-between gap-3 text-xs text-paper/50 font-medium">
            <div>© 2026 Multi-PDF ChatBot</div>
            <div className="font-mono">React · 5173 · API · 8000 · Streamlit · 8501</div>
          </div>
        </div>
      </footer>
    </div>
  )
}
