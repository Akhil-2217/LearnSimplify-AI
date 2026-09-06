/**
 * api.js — thin wrapper around fetch() for all backend calls.
 * The Vite proxy rewrites /api/* → http://127.0.0.1:8000/api/*,
 * so no backend URL is ever hardcoded in frontend source code.
 * API keys are NEVER sent from the frontend — they live only in backend/.env.
 */

const BASE = '/api'

/**
 * Map HTTP status codes to friendly user-facing messages for common errors.
 * Always falls back to the server-provided detail if no mapping exists.
 */
function friendlyError(status, serverDetail) {
  // Prefer the server-provided detail when it is clear and safe.
  if (serverDetail && serverDetail !== `HTTP ${status}`) {
    return serverDetail
  }
  switch (status) {
    case 400: return 'Invalid request. Please check your input and try again.'
    case 404: return 'No document found. Please upload a document first.'
    case 413: return 'File is too large. Please upload a file under 10 MB.'
    case 415: return 'Unsupported file type. Please upload a PDF, DOCX, or TXT file.'
    case 422: return 'The request could not be processed. Please check your input.'
    case 500: return 'An unexpected server error occurred. Please try again.'
    case 502: return 'The AI model returned an unexpected response. Please try again.'
    case 503: return 'The AI service is temporarily unavailable. Please try again later.'
    default:  return `An error occurred (HTTP ${status}). Please try again.`
  }
}

async function request(method, path, body = null) {
  const options = {
    method,
    headers: {},
  }

  if (body && !(body instanceof FormData)) {
    options.headers['Content-Type'] = 'application/json'
    options.body = JSON.stringify(body)
  } else if (body instanceof FormData) {
    // Let the browser set the correct multipart/form-data boundary.
    options.body = body
  }

  let res
  try {
    res = await fetch(`${BASE}${path}`, options)
  } catch {
    // Network-level failure (backend not running, DNS error, etc.)
    throw new Error(
      'Could not reach the backend. Make sure the FastAPI server is running on port 8000.',
    )
  }

  if (!res.ok) {
    let serverDetail = `HTTP ${res.status}`
    try {
      const err = await res.json()
      serverDetail = err.detail || err.message || serverDetail
    } catch {
      // ignore JSON parse errors — use the status code fallback
    }
    throw new Error(friendlyError(res.status, serverDetail))
  }

  return res.json()
}

export const api = {
  health: () => request('GET', '/health'),

  /**
   * Upload a document file (PDF / DOCX / TXT).
   * Sends multipart/form-data; the field name must match the FastAPI parameter: "file".
   */
  upload: (file) => {
    const form = new FormData()
    form.append('file', file)
    return request('POST', '/upload', form)
  },

  /**
   * Ask a question about the uploaded document.
   * Phase 4/5: uses RAG retrieval + IBM Granite to generate a grounded answer.
   *
   * @param {string} question        - The student's question
   * @param {string|null} docId      - Optional: restrict to a specific document
   * @param {number} topK            - Number of chunks to retrieve (default 5)
   * @param {string} learningLevel   - One of: beginner | intermediate | advanced | expert
   * @returns {Promise<{success, answer, sources, message}>}
   */
  ask: (question, docId = null, topK = 5, learningLevel = 'beginner') =>
    request('POST', '/ask', {
      question,
      document_id: docId,
      top_k: topK,
      learning_level: learningLevel,
    }),

  /**
   * Summarize the uploaded document.
   * Phase 5: uses RAG retrieval + IBM Granite.
   *
   * @param {string|null} docId  - Optional: restrict to a specific document
   * @param {string} length      - One of: short | medium | detailed
   * @returns {Promise<{success, summary, length, message}>}
   */
  summarize: (docId = null, length = 'medium') =>
    request('POST', '/summarize', {
      document_id: docId,
      length,
    }),

  /**
   * Generate a 5-question multiple-choice quiz from the uploaded document.
   * Phase 5: uses RAG retrieval + IBM Granite.
   *
   * @param {string|null} docId  - Optional: restrict to a specific document
   * @returns {Promise<{success, questions, message}>}
   */
  quiz: (docId = null) =>
    request('POST', '/quiz', {
      document_id: docId,
    }),

  // ── Phase 6 — Conversation ─────────────────────────────────────────────

  /**
   * Ask a follow-up question with conversation context.
   * Conversation history is stored server-side per doc_id.
   *
   * @param {string} question        - The student's follow-up question
   * @param {string|null} docId      - Document ID (should match uploaded doc)
   * @param {number} topK            - Number of chunks to retrieve
   * @param {string} learningLevel   - beginner | intermediate | advanced | expert
   * @returns {Promise<{success, answer, sources, message}>}
   */
  conversationAsk: (question, docId = null, topK = 5, learningLevel = 'beginner') =>
    request('POST', '/conversation/ask', {
      question,
      doc_id: docId,
      top_k: topK,
      learning_level: learningLevel,
    }),

  /**
   * Fetch the full conversation history for a document.
   *
   * @param {string} docId  - The document ID
   * @returns {Promise<{success, doc_id, turns, message}>}
   */
  getConversationHistory: (docId) =>
    request('GET', `/conversation/${encodeURIComponent(docId)}`),

  /**
   * Clear the conversation history for a document.
   *
   * @param {string} docId  - The document ID
   * @returns {Promise<{success, doc_id, message}>}
   */
  clearConversation: (docId) =>
    request('DELETE', `/conversation/${encodeURIComponent(docId)}`),
}
