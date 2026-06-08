import { useEffect, useState } from 'react'
import { downloadSourcePdf, fetchSourcePreviewUrl } from '../api/client.js'

export default function SourceViewerPanel({ source, onClose }) {
  const [previewUrl, setPreviewUrl] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [downloading, setDownloading] = useState(false)

  useEffect(() => {
    let active = true
    let objectUrl = ''

    setLoading(true)
    setError('')
    setPreviewUrl('')

    fetchSourcePreviewUrl(source)
      .then((url) => {
        if (!active) {
          URL.revokeObjectURL(url)
          return
        }
        objectUrl = url
        setPreviewUrl(url)
      })
      .catch((err) => {
        if (active) setError(err.message || 'Could not load source preview.')
      })
      .finally(() => {
        if (active) setLoading(false)
      })

    return () => {
      active = false
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [source])

  const handleDownload = async () => {
    setDownloading(true)
    try {
      await downloadSourcePdf(source)
    } catch (err) {
      setError(err.message || 'Download failed.')
    } finally {
      setDownloading(false)
    }
  }

  const label = source.label || `${source.file} - p.${source.page}`

  return (
    <aside className="fixed top-0 right-0 z-40 flex h-full w-1/2 min-w-[320px] max-w-[50vw] flex-col border-l border-line bg-paper shadow-[-16px_0_48px_rgba(17,24,39,0.12)]">
      <div className="flex items-center justify-between gap-4 border-b border-line bg-muted px-5 py-4">
        <div className="min-w-0 flex-1">
          <div className="truncate text-sm font-bold">{source.file}</div>
          <div className="mt-1 truncate text-xs font-medium text-ink/60">{label}</div>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <button
            type="button"
            onClick={handleDownload}
            disabled={downloading || loading || Boolean(error)}
            className="btn h-10 rounded-md bg-brand-50 px-4 text-sm font-semibold text-brand-600 hover:bg-brand-100 disabled:opacity-50"
          >
            {downloading ? 'Downloading…' : 'Download PDF'}
          </button>
          <button
            type="button"
            onClick={onClose}
            className="btn h-10 rounded-md bg-ink px-4 text-sm font-semibold text-paper hover:bg-brand-500"
          >
            Close
          </button>
        </div>
      </div>

      {source.excerpt ? (
        <div className="border-b border-line bg-amber2-50 px-5 py-3 text-xs font-medium text-ink/70">
          Highlighted passage: {source.excerpt.slice(0, 280)}
        </div>
      ) : null}

      <div className="relative min-h-0 flex-1 overflow-auto bg-[#f3f4f6]">
        {loading ? (
          <div className="grid h-full place-items-center text-sm font-semibold text-ink/60">
            Loading source preview…
          </div>
        ) : null}
        {error ? (
          <div className="grid h-full place-items-center px-6 text-center text-sm font-medium text-ink/70">
            {error}
          </div>
        ) : null}
        {!loading && !error && previewUrl ? (
          <img src={previewUrl} alt={`Source preview for ${source.file}`} className="block w-full bg-white" />
        ) : null}
      </div>
    </aside>
  )
}
