import { useCallback, useEffect, useRef, useState } from 'react'
import Logo from '../components/Logo.jsx'
import {
  clearChat,
  ensureSession,
  fetchStatus,
  resetSession,
  sendChat,
  updateModel,
  uploadPdfs,
} from '../api/client.js'
import { STREAMLIT_APP_URL } from '../config.js'
import SourceViewerPanel from '../components/SourceViewerPanel.jsx'

const COLOR_CLASSES = {
  brand: 'bg-brand-50 text-brand-600',
  emerald2: 'bg-emerald2-50 text-emerald2-600',
  amber2: 'bg-amber2-50 text-amber2-600',
}

const DOT_CLASSES = {
  brand: 'bg-brand-500',
  emerald2: 'bg-emerald2-500',
  amber2: 'bg-amber2-500',
}

function Toast({ message, kind, onDone }) {
  useEffect(() => {
    const timer = setTimeout(onDone, 2400)
    return () => clearTimeout(timer)
  }, [onDone])

  const palette = {
    info: 'bg-ink text-paper',
    ok: 'bg-emerald2-500 text-paper',
    warn: 'bg-amber2-500 text-ink',
  }

  return (
    <div className={`fade-up rounded-md px-4 h-12 inline-flex items-center font-semibold text-sm ${palette[kind] || palette.info}`}>
      {message}
    </div>
  )
}

function MessageBubble({ message, onSourceClick }) {
  if (message.role === 'user') {
    return (
      <div className="fade-up flex justify-end mb-6">
        <div className="max-w-[78%] bg-ink text-paper rounded-md px-5 py-4 text-[15px] font-medium leading-relaxed whitespace-pre-line">
          {message.text}
        </div>
      </div>
    )
  }

  if (message.role === 'thinking') {
    return (
      <div className="fade-up flex items-start gap-3 mb-6">
        <span className="w-9 h-9 bg-brand-500 rounded-md grid place-items-center text-paper font-extrabold text-sm shrink-0">A</span>
        <div className="bg-muted rounded-md px-5 py-4 inline-flex items-center gap-2">
          <span className="text-xs font-bold text-ink/60 label">retrieving</span>
          <span className="dot" /><span className="dot" /><span className="dot" />
        </div>
      </div>
    )
  }

  return (
    <div className="fade-up flex items-start gap-3 mb-8">
      <span className="w-9 h-9 bg-brand-500 rounded-md grid place-items-center text-paper font-extrabold text-sm shrink-0">A</span>
      <div className="flex-1 min-w-0 max-w-[88%]">
        <div className="bg-muted rounded-md px-5 py-4 text-[15px] leading-relaxed whitespace-pre-line">{message.text}</div>
        {message.sources?.length > 0 && (
          <div className="mt-3">
            <div className="label text-ink/50 mb-2">Sources retrieved</div>
            <div className="flex flex-wrap gap-2">
              {message.sources.map((source, sourceIndex) => (
                <button
                  type="button"
                  key={`${source.file}-${source.page}-${source.line || 0}-${sourceIndex}`}
                  onClick={() => onSourceClick?.(source)}
                  className={`inline-flex items-center gap-1.5 h-7 px-2.5 rounded-md text-xs font-semibold cursor-pointer transition-opacity hover:opacity-85 ${COLOR_CLASSES[source.color] || COLOR_CLASSES.brand}`}
                  title="Open highlighted source preview"
                >
                  <span className={`w-1.5 h-1.5 rounded-sm ${DOT_CLASSES[source.color] || DOT_CLASSES.brand}`} />
                  {source.label || source.file}{' '}
                  <span className="opacity-60">· p.{source.page}</span>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default function Dashboard() {
  const [indexedFiles, setIndexedFiles] = useState([])
  const [stats, setStats] = useState({ chunks: 0, pages: 0, dims: 384, top_k: 4 })
  const [config, setConfig] = useState(null)
  const [availableModels, setAvailableModels] = useState([])
  const [selectedModelLabel, setSelectedModelLabel] = useState('')
  const [messages, setMessages] = useState([])
  const [suggestedQuestions, setSuggestedQuestions] = useState([])
  const [pendingFiles, setPendingFiles] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(true)
  const [processing, setProcessing] = useState(false)
  const [sending, setSending] = useState(false)
  const [toasts, setToasts] = useState([])
  const [streamlitUrl, setStreamlitUrl] = useState(STREAMLIT_APP_URL)
  const [dropHover, setDropHover] = useState(false)
  const [activeSource, setActiveSource] = useState(null)
  const messagesRef = useRef(null)
  const fileInputRef = useRef(null)

  const pushToast = useCallback((message, kind = 'info') => {
    const id = Date.now() + Math.random()
    setToasts((prev) => [...prev, { id, message, kind }])
  }, [])

  const removeToast = useCallback((id) => {
    setToasts((prev) => prev.filter((t) => t.id !== id))
  }, [])

  const refreshStatus = useCallback(async () => {
    const data = await fetchStatus()
    setIndexedFiles(data.indexed_files || [])
    setStats((prev) => data.stats || prev)
    setConfig(data.config || null)
    setAvailableModels(data.available_models || [])
    const selected = data.selected_model
    if (selected?.provider && selected?.model) {
      const label = `${selected.provider}::${selected.model}`
      setSelectedModelLabel(label)
    }
    setMessages(data.messages || [])
    setSuggestedQuestions(data.suggested_questions || data.example_questions || [])
    if (data.streamlit_url) setStreamlitUrl(data.streamlit_url)
    return data
  }, [])

  useEffect(() => {
    ensureSession()
      .then(refreshStatus)
      .catch((err) => pushToast(err.message, 'warn'))
      .finally(() => setLoading(false))
  }, [pushToast, refreshStatus])

  useEffect(() => {
    if (messagesRef.current) {
      messagesRef.current.scrollTop = messagesRef.current.scrollHeight
    }
  }, [messages, sending])

  const queueFiles = (files) => {
    const pdfs = [...files].filter((f) => f.name.toLowerCase().endsWith('.pdf'))
    if (pdfs.length === 0) {
      pushToast('Only PDF files are supported', 'warn')
      return
    }
    setPendingFiles((prev) => {
      const next = [...prev]
      pdfs.forEach((file) => {
        const indexed = indexedFiles.find((f) => f.name === file.name)
        if (indexed) {
          pushToast(
            `you cannot add same file twice. Existing doc reference: ${indexed.name} (${indexed.pages} pages, ${indexed.chunks} chunks)`,
            'warn'
          )
          return
        }
        if (next.some((p) => p.name === file.name)) {
          pushToast(`you cannot add same file twice. Existing doc reference: ${file.name}`, 'warn')
          return
        }
        next.push(file)
      })
      return next
    })
  }

  const processPending = async () => {
    if (pendingFiles.length === 0 || processing) return
    setProcessing(true)
    const batch = [...pendingFiles]
    try {
      const result = await uploadPdfs(batch)
      const failed = new Set(result.failed || [])
      const invalid = new Set(result.invalid || [])
      const skipped = new Set(result.skipped || [])
      setPendingFiles((prev) =>
        prev.filter(
          (file) =>
            failed.has(file.name) ||
            invalid.has(file.name) ||
            (skipped.has(file.name) && !result.indexed_files?.some((f) => f.name === file.name))
        )
      )
      if (result.indexed_files) {
        setIndexedFiles(result.indexed_files)
      }
      if (result.suggested_questions?.length) {
        setSuggestedQuestions(result.suggested_questions)
      }
      await refreshStatus()
      const hasWarnings = (result.failed?.length || 0) + (result.invalid?.length || 0) > 0
      pushToast(result.message || 'PDFs indexed', hasWarnings ? 'warn' : 'ok')
    } catch (err) {
      const references = err?.payload?.existing_references
      if (Array.isArray(references) && references.length > 0) {
        const refText = references
          .map((item) => `${item.name} (${item.pages ?? 0} pages, ${item.chunks ?? 0} chunks)`)
          .join(', ')
        pushToast(`${err.message}. Existing doc reference: ${refText}`, 'warn')
      } else {
        pushToast(err.message, 'warn')
      }
    } finally {
      setProcessing(false)
    }
  }

  const handleSend = async (text) => {
    const prompt = (text || input).trim()
    if (!prompt || sending) return
    if (indexedFiles.length === 0) {
      pushToast('Upload + process PDFs first', 'warn')
      return
    }

    setSending(true)
    setInput('')
    setMessages((prev) => [...prev, { role: 'user', text: prompt }, { role: 'thinking' }])
    try {
      const result = await sendChat(prompt)
      setMessages(result.messages || [])
    } catch (err) {
      setMessages((prev) => prev.filter((m) => m.role !== 'thinking'))
      pushToast(err.message, 'warn')
    } finally {
      setSending(false)
    }
  }

  const handleClearChat = async () => {
    try {
      const result = await clearChat()
      setMessages(result.messages || [])
      pushToast('Chat cleared', 'info')
    } catch (err) {
      pushToast(err.message, 'warn')
    }
  }

  const handleReset = async () => {
    if (!window.confirm('Reset session? This clears chat AND removes indexed PDFs.')) return
    try {
      await resetSession()
      setPendingFiles([])
      await refreshStatus()
      pushToast('Session reset', 'warn')
    } catch (err) {
      pushToast(err.message, 'warn')
    }
  }

  const handleModelChange = async (event) => {
    const value = event.target.value
    setSelectedModelLabel(value)
    const [provider, model] = value.split('::')
    try {
      await updateModel(provider, model)
      await refreshStatus()
      pushToast('Model updated', 'ok')
    } catch (err) {
      pushToast(err.message, 'warn')
      await refreshStatus()
    }
  }

  if (loading) {
    return (
      <div className="h-screen grid place-items-center bg-muted text-ink">
        <div className="text-center">
          <span className="spin inline-block w-8 h-8 border-2 border-ink border-t-transparent rounded-full" />
          <p className="mt-4 font-semibold">Loading dashboard…</p>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-muted text-ink antialiased h-screen overflow-hidden">
      <div className={`h-full flex flex-col transition-[margin] duration-300 ${activeSource ? 'mr-[50vw]' : ''}`}>
        <header className="bg-paper border-b border-line shrink-0">
          <div className="h-16 px-5 lg:px-8 flex items-center justify-between">
            <div className="flex items-center gap-5">
              <Logo compact />
              <span className="hidden md:inline-flex items-center gap-2 h-7 px-2.5 bg-emerald2-50 text-emerald2-600 rounded-md label">
                <span className="w-1.5 h-1.5 bg-emerald2-500 rounded-sm" />
                Connected · {config?.provider || 'openrouter'} · {config?.llm || 'openrouter'}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <button type="button" onClick={handleClearChat} className="btn h-10 px-4 rounded-md bg-muted hover:bg-line text-sm font-semibold">
                Clear chat
              </button>
              <button type="button" onClick={handleReset} className="btn h-10 px-4 rounded-md bg-muted hover:bg-line text-sm font-semibold">
                Reset session
              </button>
              <div className="w-px h-8 bg-line mx-1 hidden sm:block" />
              <a
                href={streamlitUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="btn hidden sm:inline-flex h-10 px-4 rounded-md bg-brand-50 hover:bg-brand-100 text-brand-600 text-sm font-semibold items-center gap-2"
                title="Open the classic Streamlit UI"
              >
                Streamlit UI
              </a>
            </div>
          </div>
        </header>

        <div className="flex-1 grid grid-cols-12 gap-0 min-h-0">
          <aside className="col-span-12 md:col-span-4 lg:col-span-3 bg-paper border-r border-line flex flex-col min-h-0">
            <div className="p-6 border-b border-line">
              <div className="label text-ink/60">Knowledge base</div>
              <h2 className="mt-1 font-extrabold text-2xl h-tight">Documents</h2>
            </div>

            <div className="p-6 space-y-6 overflow-y-auto scroll flex-1">
              <div>
                <div className="label text-ink/60 mb-3">Upload</div>
                <label
                  htmlFor="file-input"
                  className={`dropzone block rounded-lg p-6 text-center cursor-pointer card-hover bg-brand-50/50 ${dropHover ? 'dropzone-hover' : ''}`}
                  onDragEnter={(e) => { e.preventDefault(); setDropHover(true) }}
                  onDragOver={(e) => e.preventDefault()}
                  onDragLeave={() => setDropHover(false)}
                  onDrop={(e) => {
                    e.preventDefault()
                    setDropHover(false)
                    queueFiles(e.dataTransfer.files)
                  }}
                >
                  <input
                    ref={fileInputRef}
                    id="file-input"
                    type="file"
                    accept=".pdf"
                    multiple
                    className="hidden"
                    onChange={(e) => {
                      queueFiles(e.target.files)
                      e.target.value = ''
                    }}
                  />
                  <div className="w-12 h-12 mx-auto rounded-full bg-paper grid place-items-center">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" strokeWidth="2.5">
                      <path d="M12 4v12m0 0l-5-5m5 5l5-5M4 20h16" />
                    </svg>
                  </div>
                  <div className="mt-4 font-bold text-base">Drop PDFs here</div>
                  <div className="mt-1 text-xs font-medium text-ink/60">or click to browse</div>
                </label>
              </div>

              {availableModels.length > 0 && (
                <div>
                  <div className="label text-ink/60 mb-3">Models</div>
                  <select
                    value={selectedModelLabel}
                    onChange={handleModelChange}
                    className="w-full h-12 rounded-md bg-muted border border-line px-3 text-sm font-medium outline-none focus:border-brand-500"
                  >
                    {availableModels.map((option) => {
                      const value = `${option.provider}::${option.model}`
                      return (
                        <option key={value} value={value}>
                          {option.label}
                        </option>
                      )
                    })}
                  </select>
                </div>
              )}

              {pendingFiles.length > 0 && (
                <div>
                  <div className="label text-ink/60 mb-3">Queued</div>
                  <div className="space-y-2">
                    {pendingFiles.map((file, index) => (
                      <div key={file.name} className="fade-up flex items-center gap-3 bg-muted rounded-md p-3">
                        <div className="flex-1 min-w-0">
                          <div className="text-sm font-bold truncate">{file.name}</div>
                          <div className="text-[11px] font-mono text-ink/50">{(file.size / 1024).toFixed(0)} KB · pending</div>
                        </div>
                        <button
                          type="button"
                          className="btn h-8 w-8 rounded-md hover:bg-line grid place-items-center"
                          onClick={() => setPendingFiles((prev) => prev.filter((_, i) => i !== index))}
                        >
                          ×
                        </button>
                      </div>
                    ))}
                  </div>
                  <button
                    type="button"
                    disabled={processing}
                    onClick={processPending}
                    className="btn mt-4 w-full h-12 rounded-md bg-brand-500 hover:bg-brand-600 text-paper font-bold inline-flex items-center justify-center gap-2 disabled:opacity-60"
                  >
                    {processing ? (
                      <>
                        <span className="spin inline-block w-4 h-4 border-2 border-paper border-t-transparent rounded-full" />
                        Processing…
                      </>
                    ) : (
                      'Process PDFs'
                    )}
                  </button>
                </div>
              )}

              <div>
                <div className="flex items-center justify-between mb-3">
                  <div className="label text-ink/60">Indexed</div>
                  <span className="label text-brand-500">{indexedFiles.length} file{indexedFiles.length === 1 ? '' : 's'}</span>
                </div>
                <div className="space-y-2">
                  {indexedFiles.length === 0 ? (
                    <div className="text-sm text-ink/50 italic">No documents indexed yet.</div>
                  ) : (
                    indexedFiles.map((file, index) => {
                      const colors = ['brand', 'emerald2', 'amber2']
                      const color = colors[index % 3]
                      return (
                        <div key={file.name} className={`fade-up rounded-md p-3 card-hover ${COLOR_CLASSES[color]}`}>
                          <div className="text-sm font-bold truncate">{file.name}</div>
                          <div className="text-[11px] font-mono opacity-70">{file.pages} pages · {file.chunks} chunks</div>
                        </div>
                      )
                    })
                  )}
                </div>
              </div>

              <div className="bg-muted rounded-lg p-5">
                <div className="label text-ink/60">Vector store</div>
                <div className="mt-3 grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <div className="font-extrabold text-2xl h-tight text-brand-500">{stats.chunks}</div>
                    <div className="label text-ink/50 mt-0.5">chunks</div>
                  </div>
                  <div>
                    <div className="font-extrabold text-2xl h-tight text-emerald2-500">{stats.dims}</div>
                    <div className="label text-ink/50 mt-0.5">dims</div>
                  </div>
                  <div>
                    <div className="font-extrabold text-2xl h-tight text-amber2-500">{stats.top_k}</div>
                    <div className="label text-ink/50 mt-0.5">top-k</div>
                  </div>
                  <div>
                    <div className="font-extrabold text-2xl h-tight text-ink">{stats.pages}</div>
                    <div className="label text-ink/50 mt-0.5">pages</div>
                  </div>
                </div>
              </div>
            </div>

            {config && (
              <div className="p-5 border-t border-line bg-paper">
                <div className="label text-ink/60 mb-3">Configuration</div>
                <div className="space-y-2">
                  {[
                    ['LLM', config.llm],
                    ['Store', config.store],
                    ['Embeddings', config.embeddings],
                  ].map(([label, value]) => (
                    <div key={label} className="flex items-center justify-between bg-muted rounded-md h-9 px-3">
                      <span className="text-xs font-semibold text-ink/60">{label}</span>
                      <span className="text-xs font-mono">{value}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </aside>

          <main className="col-span-12 md:col-span-8 lg:col-span-9 flex flex-col min-h-0 bg-paper">
            <div className="px-8 py-5 border-b border-line flex items-center justify-between">
              <div>
                <div className="label text-ink/60">Session</div>
                <h1 className="mt-1 font-extrabold text-2xl h-tight">Chat with your PDFs</h1>
              </div>
              <span className="hidden lg:inline-flex h-8 px-3 items-center bg-muted rounded-md label text-ink/60">
                <span className="w-1.5 h-1.5 bg-emerald2-500 rounded-sm mr-2" />
                memory · buffer
              </span>
            </div>

            <div ref={messagesRef} className="flex-1 overflow-y-auto scroll px-8 py-8">
              {messages.length === 0 ? (
                <div className="h-full grid place-items-center">
                  <div className="max-w-2xl text-center">
                    <div className="relative inline-flex">
                      <span className="absolute -top-4 -left-6 w-14 h-14 rounded-full bg-amber2-500" />
                      <span className="relative inline-flex w-20 h-20 bg-brand-500 text-paper rounded-md items-center justify-center font-black text-3xl">?</span>
                    </div>
                    <h2 className="mt-10 font-black text-5xl h-tight">
                      {indexedFiles.length > 0 ? 'Ask your first question.' : 'Upload PDFs to begin.'}
                    </h2>
                    <p className="mt-4 text-ink/60 text-lg">
                      {indexedFiles.length > 0
                        ? 'Type a question below or pick a suggested prompt. Answers will cite the source PDF + page.'
                        : 'Drop one or more PDFs in the sidebar and hit Process PDFs. The chat unlocks once your index is built.'}
                    </p>
                    {indexedFiles.length > 0 && suggestedQuestions.length > 0 && (
                      <div className="mt-8 flex flex-wrap gap-2 justify-center">
                        {suggestedQuestions.map((prompt) => (
                          <button
                            key={prompt}
                            type="button"
                            className="btn h-12 px-5 rounded-md bg-muted hover:bg-brand-50 hover:text-brand-600 text-sm font-semibold"
                            onClick={() => handleSend(prompt)}
                          >
                            {prompt}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                messages.map((message, index) => (
                  <MessageBubble
                    key={`${message.role}-${index}`}
                    message={message}
                    onSourceClick={setActiveSource}
                  />
                ))
              )}
            </div>

            {indexedFiles.length > 0 && suggestedQuestions.length > 0 && (
              <div className="px-8 pb-3">
                <div className="label text-ink/50 mb-3">Suggested questions</div>
                <div className="flex flex-wrap gap-2">
                  {suggestedQuestions.map((prompt) => (
                    <button
                      key={prompt}
                      type="button"
                      className="btn h-10 px-4 rounded-md bg-muted hover:bg-brand-50 hover:text-brand-600 text-sm font-semibold"
                      onClick={() => setInput(prompt)}
                    >
                      {prompt}
                    </button>
                  ))}
                </div>
              </div>
            )}

            <div className="px-8 pb-8 pt-3 border-t border-line bg-paper">
              <form
                className="relative"
                onSubmit={(e) => {
                  e.preventDefault()
                  handleSend()
                }}
              >
                <input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  type="text"
                  autoComplete="off"
                  placeholder="Ask anything about your indexed PDFs…"
                  disabled={sending}
                  className="w-full h-16 bg-muted focus:bg-paper rounded-md px-5 pr-36 text-base font-medium outline-none border-2 border-transparent focus:border-brand-500 transition-colors"
                />
                <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-2">
                  <button
                    type="submit"
                    disabled={sending}
                    className="btn h-12 px-5 rounded-md bg-ink hover:bg-brand-500 text-paper font-bold text-sm inline-flex items-center gap-2 disabled:opacity-60"
                  >
                    Send
                  </button>
                </div>
              </form>
              <div className="mt-3 flex items-center justify-between text-xs font-medium text-ink/50">
                <span>Enter to send</span>
                <div className="font-mono">
                  temperature {config?.temperature ?? 0.3} · top-k {stats.top_k} · max {config?.max_tokens ?? 1024}
                </div>
              </div>
            </div>
          </main>
        </div>
      </div>

      {activeSource ? (
        <SourceViewerPanel source={activeSource} onClose={() => setActiveSource(null)} />
      ) : null}

      <div className="fixed bottom-6 right-6 space-y-2 z-50">
        {toasts.map((toast) => (
          <Toast
            key={toast.id}
            message={toast.message}
            kind={toast.kind}
            onDone={() => removeToast(toast.id)}
          />
        ))}
      </div>
    </div>
  )
}
