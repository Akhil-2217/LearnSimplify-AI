/**
 * api.js — thin wrapper around fetch() for all backend calls.
 *
 * In production, requests are sent to the deployed Render backend.
 * API keys are NEVER sent from the frontend — they live only in backend/.env.
 */

const BASE = 'https://learnsimplify-ai.onrender.com/api'

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
    case 400:
      return 'Invalid request. Please check your input and try again.'

    case 404:
      return 'No document found. Please upload a document first.'

    case 413:
      return 'File is too large. Please upload a file under 10 MB.'

    case 415:
      return 'Unsupported file type. Please upload a PDF, DOCX, or TXT file.'

    case 422:
      return 'The request could not be processed. Please check your input.'

    case 500:
      return 'An unexpected server error occurred. Please try again.'

    case 502:
      return 'The AI model returned an unexpected response. Please try again.'

    case 503:
      return 'The AI service is temporarily unavailable. Please try again later.'

    default:
      return `An error occurred (HTTP ${status}). Please try again.`
  }
}

/**
 * Send a request to the backend API.
 */
async function request(method, path, body = null) {
  const options = {
    method,
    headers: {},
  }

  // Store the response outside the try block so it can be
  // accessed after fetch() completes.
  let res

  // JSON request body
  if (body && !(body instanceof FormData)) {
    options.headers['Content-Type'] = 'application/json'
    options.body = JSON.stringify(body)
  }

  // File upload using multipart/form-data
  else if (body instanceof FormData) {
    // Do NOT manually set Content-Type.
    // The browser automatically adds the multipart boundary.
    options.body = body
  }

  try {
    res = await fetch(`${BASE}${path}`, options)
  } catch (err) {
    console.error('API request failed:', err)
    console.error('Request URL:', `${BASE}${path}`)

    throw new Error(
      `Could not reach the backend: ${err?.message || 'Network error'}`
    )
  }

  // Handle HTTP errors
  if (!res.ok) {
    let serverDetail = `HTTP ${res.status}`

    try {
      const err = await res.json()
      serverDetail = err.detail || err.message || serverDetail
    } catch {
      // Ignore JSON parse errors.
      // Use the status code fallback instead.
    }

    throw new Error(friendlyError(res.status, serverDetail))
  }

  // Return successful JSON response
  return res.json()
}

export const api = {
  /**
   * Check whether the backend is running.
   */
  health: () => request('GET', '/health'),

  /**
   * Upload a document file (PDF / DOCX / TXT).
   *
   * Sends multipart/form-data.
   * The field name must match the FastAPI parameter: "file".
   */
  upload: (file) => {
    const form = new FormData()
    form.append('file', file)

    return request('POST', '/upload', form)
  },

  /**
   * Ask a question about the uploaded document.
   *
   * Uses RAG retrieval + IBM Granite to generate a grounded answer.
   *
   * @param {string} question
   * @param {string|null} docId
   * @param {number} topK
   * @param {string} learningLevel
   */
  ask: (
    question,
    docId = null,
    topK = 5,
    learningLevel = 'beginner'
  ) =>
    request('POST', '/ask', {
      question,
      document_id: docId,
      top_k: topK,
      learning_level: learningLevel,
    }),

  /**
   * Summarize the uploaded document.
   *
   * @param {string|null} docId
   * @param {string} length
   */
  summarize: (docId = null, length = 'medium') =>
    request('POST', '/summarize', {
      document_id: docId,
      length,
    }),

  /**
   * Generate a 5-question multiple-choice quiz
   * from the uploaded document.
   *
   * @param {string|null} docId
   */
  quiz: (docId = null) =>
    request('POST', '/quiz', {
      document_id: docId,
    }),

  // ─────────────────────────────────────────────────────────────
  // Phase 6 — Conversation
  // ─────────────────────────────────────────────────────────────

  /**
   * Ask a follow-up question with conversation context.
   *
   * Conversation history is stored server-side per doc_id.
   *
   * @param {string} question
   * @param {string|null} docId
   * @param {number} topK
   * @param {string} learningLevel
   */
  conversationAsk: (
    question,
    docId = null,
    topK = 5,
    learningLevel = 'beginner'
  ) =>
    request('POST', '/conversation/ask', {
      question,
      doc_id: docId,
      top_k: topK,
      learning_level: learningLevel,
    }),

  /**
   * Fetch the full conversation history for a document.
   *
   * @param {string} docId
   */
  getConversationHistory: (docId) =>
    request(
      'GET',
      `/conversation/${encodeURIComponent(docId)}`
    ),

  /**
   * Clear the conversation history for a document.
   *
   * @param {string} docId
   */
  clearConversation: (docId) =>
    request(
      'DELETE',
      `/conversation/${encodeURIComponent(docId)}`
    ),
}