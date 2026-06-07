const SESSION_KEY = 'mpdf_session_id'

export function getSessionId() {
  return localStorage.getItem(SESSION_KEY)
}

export function setSessionId(id) {
  localStorage.setItem(SESSION_KEY, id)
}

async function request(path, options = {}) {
  const sessionId = getSessionId()
  const url = sessionId ? `${path}?session_id=${encodeURIComponent(sessionId)}` : path
  const response = await fetch(url, options)
  const data = await response.json().catch(() => ({}))
  if (!response.ok) {
    const detail = Array.isArray(data.detail)
      ? data.detail.map((item) => item.msg || item).join(', ')
      : data.detail
    throw new Error(detail || data.message || 'Request failed')
  }
  return data
}

export async function ensureSession() {
  let sessionId = getSessionId()
  if (!sessionId) {
    const data = await request('/api/session', { method: 'POST' })
    sessionId = data.session_id
    setSessionId(sessionId)
  }
  return sessionId
}

export function fetchStatus() {
  return request('/api/status')
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
