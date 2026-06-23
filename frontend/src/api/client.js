const SESSION_KEY = 'mpdf_session_id'
const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')

function apiUrl(path, sessionId) {
  const url = `${API_BASE}${path}`
  return sessionId ? `${url}?session_id=${encodeURIComponent(sessionId)}` : url
}

export function getSessionId() {
  return localStorage.getItem(SESSION_KEY)
}

export function setSessionId(id) {
  localStorage.setItem(SESSION_KEY, id)
}

async function request(path, options = {}) {
  const sessionId = getSessionId()
  const response = await fetch(apiUrl(path, sessionId), options)
  const data = await response.json().catch(() => ({}))
  if (!response.ok) {
    const detail = Array.isArray(data.detail)
      ? data.detail.map((item) => item.msg || item).join(', ')
      : data.detail
    throw new Error(detail || data.message || 'Request failed')
  }
  return data
}

let sessionBootstrap = null

export async function ensureSession() {
  const existing = getSessionId()
  if (existing) return existing
  if (!sessionBootstrap) {
    sessionBootstrap = request('/api/session', { method: 'POST' })
      .then((data) => {
        setSessionId(data.session_id)
        return data.session_id
      })
      .finally(() => {
        sessionBootstrap = null
      })
  }
  return sessionBootstrap
}

export function fetchStatus() {
  return request('/api/status')
}

export function updateModel(provider, model) {
  return request('/api/model', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ provider, model }),
  })
}

export async function uploadPdfs(files) {
  const formData = new FormData()
  files.forEach((file) => formData.append('files', file))
  return request('/api/upload', { method: 'POST', body: formData })
}

export function sendChat(message) {
  return request('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
  })
}

export function clearChat() {
  return request('/api/clear-chat', { method: 'POST' })
}

export function resetSession() {
  return request('/api/reset', { method: 'POST' })
}

async function sourceRequest(path, source) {
  const sessionId = getSessionId()
  const response = await fetch(apiUrl(path, sessionId), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(source),
  })
  if (!response.ok) {
    const data = await response.json().catch(() => ({}))
    const detail = Array.isArray(data.detail)
      ? data.detail.map((item) => item.msg || item).join(', ')
      : data.detail
    throw new Error(detail || data.message || 'Request failed')
  }
  return response
}

export async function fetchSourcePreviewUrl(source) {
  const response = await sourceRequest('/api/source/preview', source)
  const blob = await response.blob()
  return URL.createObjectURL(blob)
}

export async function downloadSourcePdf(source) {
  const response = await sourceRequest('/api/source/download', source)
  const blob = await response.blob()
  const filename = `${source.file?.replace(/\.pdf$/i, '') || 'source'}-p${source.page}-highlighted.pdf`
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}
