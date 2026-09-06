/**
 * Conversation.jsx — Phase 6 conversation history + follow-up Q&A panel.
 *
 * Props:
 *   docId (string) — the document_id returned by the upload endpoint.
 *
 * Features:
 *  - Displays the full conversation history for the uploaded document.
 *  - Allows follow-up questions that use conversation context.
 *  - Clear Conversation button resets the history server-side.
 *  - Preserves the existing LearnSimplify AI design language.
 */

import { useState, useEffect, useRef } from 'react'
import { api } from '../services/api'
import './Conversation.css'

const LEARNING_LEVELS = [
  { value: 'beginner',     label: 'Beginner',     desc: 'Simple language, everyday analogies' },
  { value: 'intermediate', label: 'Intermediate',  desc: 'Clear language, key terms defined' },
  { value: 'advanced',     label: 'Advanced',      desc: 'Technical language, nuanced detail' },
  { value: 'expert',       label: 'Expert',        desc: 'Full technical depth, edge cases' },
]

export default function Conversation({ docId }) {
  const [history, setHistory]           = useState([])   // ConversationTurn[]
  const [question, setQuestion]         = useState('')
  const [learningLevel, setLearningLevel] = useState('beginner')
  const [status, setStatus]             = useState(null) // null | {type, title, detail}
  const historyEndRef                   = useRef(null)

  // Load existing history when docId changes (e.g. after a new upload)
  useEffect(() => {
    if (!docId) return
    api.getConversationHistory(docId)
      .then((data) => setHistory(data.turns || []))
      .catch(() => setHistory([]))
  }, [docId])

  // Auto-scroll to the bottom of the conversation list after new turns
  useEffect(() => {
    historyEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [history])

  const isLoading = status?.type === 'loading'
  const selectedLevel = LEARNING_LEVELS.find((l) => l.value === learningLevel)

  async function handleAsk() {
    const trimmed = question.trim()
    if (!trimmed) {
      setStatus({ type: 'error', title: 'Question is empty', detail: 'Please type a question before clicking Ask.' })
      return
    }

    setStatus({ type: 'loading', title: 'Thinking…', detail: 'Retrieving relevant content and generating an answer.' })

    try {
      const data = await api.conversationAsk(trimmed, docId, 5, learningLevel)

      // Append the new turn to local state immediately (no extra GET needed)
      const newTurn = {
        turn_id: Date.now().toString(),
        question: trimmed,
        answer: data.answer,
        learning_level: learningLevel,
        sources: data.sources || [],
        timestamp: new Date().toISOString(),
      }
      setHistory((prev) => [...prev, newTurn])
      setQuestion('')
      setStatus(null)
    } catch (err) {
      setStatus({
        type: 'error',
        title: 'Could not get an answer',
        detail: err.message || 'An unexpected error occurred. Please try again.',
      })
    }
  }

  async function handleClear() {
    try {
      await api.clearConversation(docId)
      setHistory([])
      setStatus(null)
    } catch (err) {
      setStatus({
        type: 'error',
        title: 'Could not clear conversation',
        detail: err.message || 'An unexpected error occurred.',
      })
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleAsk()
    }
  }

  return (
    <div className="conv-card">
      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <div className="conv-header-row">
        <h2>💬 Conversation</h2>
        {history.length > 0 && (
          <button
            className="conv-clear-btn"
            onClick={handleClear}
            disabled={isLoading}
            type="button"
            title="Clear all conversation history for this document"
          >
            🗑 Clear conversation
          </button>
        )}
      </div>
      <p className="conv-hint">
        Ask questions about the uploaded course material. Follow-up questions use
        the conversation context so you can dig deeper naturally.
      </p>

      {/* ── Conversation history ────────────────────────────────────────────── */}
      {history.length > 0 && (
        <div className="conv-history" aria-label="Conversation history">
          {history.map((turn, idx) => (
            <div key={turn.turn_id || idx} className="conv-turn">
              {/* Question bubble */}
              <div className="conv-bubble conv-bubble--user">
                <span className="conv-bubble-role">You</span>
                <p className="conv-bubble-text">{turn.question}</p>
              </div>

              {/* Answer bubble */}
              <div className="conv-bubble conv-bubble--ai">
                <span className="conv-bubble-role">
                  AI
                  <span className="conv-level-badge">
                    {turn.learning_level}
                  </span>
                </span>
                <p className="conv-bubble-text">{turn.answer}</p>
                {turn.sources && turn.sources.length > 0 && (
                  <details className="conv-sources">
                    <summary className="conv-sources-summary">
                      {turn.sources.length} source{turn.sources.length !== 1 ? 's' : ''}
                    </summary>
                    <ul className="conv-sources-list">
                      {turn.sources.map((src, si) => (
                        <li key={si} className="conv-source-item">
                          <span className="conv-source-file">📄 {src.filename}</span>
                          <span className="conv-source-meta">
                            Chunk {src.chunk_index} · score {src.score.toFixed(3)}
                          </span>
                        </li>
                      ))}
                    </ul>
                  </details>
                )}
              </div>
            </div>
          ))}
          <div ref={historyEndRef} />
        </div>
      )}

      {/* ── Empty state ─────────────────────────────────────────────────────── */}
      {history.length === 0 && !status && (
        <div className="conv-empty">
          Ask your first question below to start a conversation about this document.
        </div>
      )}

      {/* ── Status / error ──────────────────────────────────────────────────── */}
      {status && status.type !== 'loading' && (
        <div className={`conv-status ${status.type}`} role="alert">
          <em className="conv-status-icon" aria-hidden="true">⚠️</em>
          <div className="conv-status-body">
            <strong>{status.title}</strong>
            <span>{status.detail}</span>
          </div>
        </div>
      )}

      {isLoading && (
        <div className="conv-status loading" role="status">
          <span className="conv-spinner" aria-hidden="true" />
          <div className="conv-status-body">
            <strong>{status.title}</strong>
            <span>{status.detail}</span>
          </div>
        </div>
      )}

      {/* ── Learning level selector ─────────────────────────────────────────── */}
      <div className="conv-level-row">
        <label className="conv-level-label">Level</label>
        <div className="conv-level-buttons" role="group" aria-label="Learning level">
          {LEARNING_LEVELS.map((lvl) => (
            <button
              key={lvl.value}
              className={`conv-level-btn ${learningLevel === lvl.value ? 'active' : ''}`}
              onClick={() => setLearningLevel(lvl.value)}
              disabled={isLoading}
              title={lvl.desc}
              aria-pressed={learningLevel === lvl.value}
              type="button"
            >
              {lvl.label}
            </button>
          ))}
        </div>
        {selectedLevel && (
          <span className="conv-level-desc">{selectedLevel.desc}</span>
        )}
      </div>

      {/* ── Input row ───────────────────────────────────────────────────────── */}
      <div className="conv-input-row">
        <textarea
          className="conv-textarea"
          rows={3}
          placeholder={
            history.length > 0
              ? 'Ask a follow-up question…'
              : 'e.g. Explain the OSI model.'
          }
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isLoading}
          aria-label="Your question"
        />
        <button
          className="conv-ask-btn"
          onClick={handleAsk}
          disabled={isLoading || !question.trim()}
          aria-label="Ask question"
        >
          {isLoading ? (
            <>
              <span className="conv-spinner" aria-hidden="true" />
              Thinking…
            </>
          ) : history.length > 0 ? (
            'Follow up'
          ) : (
            'Ask'
          )}
        </button>
      </div>
    </div>
  )
}
