import { useState } from 'react'
import { api } from '../services/api'
import './QuestionAnswer.css'

/**
 * QuestionAnswer — Phase 4/5 Q&A panel.
 *
 * Props:
 *   docId (string|null) — the document_id returned by the upload endpoint.
 *                         Pass null to search across all uploaded documents.
 */

const LEARNING_LEVELS = [
  { value: 'beginner',     label: 'Beginner',     desc: 'Simple language, everyday analogies' },
  { value: 'intermediate', label: 'Intermediate',  desc: 'Clear language, key terms defined' },
  { value: 'advanced',     label: 'Advanced',      desc: 'Technical language, nuanced detail' },
  { value: 'expert',       label: 'Expert',        desc: 'Full technical depth, edge cases' },
]

export default function QuestionAnswer({ docId }) {
  const [question, setQuestion] = useState('')
  const [learningLevel, setLearningLevel] = useState('beginner')
  const [status, setStatus] = useState(null) // null | {type, title, detail}
  const [answer, setAnswer] = useState(null)   // string | null
  const [sources, setSources] = useState([])   // SourceChunk[]

  async function handleAsk() {
    const trimmed = question.trim()
    if (!trimmed) {
      setStatus({ type: 'error', title: 'Question is empty', detail: 'Please type a question before clicking Ask.' })
      return
    }

    setStatus({ type: 'loading', title: 'Thinking…', detail: 'Retrieving relevant content and generating an answer.' })
    setAnswer(null)
    setSources([])

    try {
      const data = await api.ask(trimmed, docId, 5, learningLevel)
      setAnswer(data.answer)
      setSources(data.sources || [])
      setStatus(null)
    } catch (err) {
      setAnswer(null)
      setSources([])
      setStatus({
        type: 'error',
        title: 'Could not get an answer',
        detail: err.message || 'An unexpected error occurred. Please try again.',
      })
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleAsk()
    }
  }

  const isLoading = status?.type === 'loading'
  const selectedLevel = LEARNING_LEVELS.find(l => l.value === learningLevel)

  return (
    <div className="qa-card">
      <h2>💬 Ask a Question</h2>
      <p className="qa-hint">
        Ask anything about the uploaded course material. IBM Granite will answer
        using only the content from your document.
      </p>

      {/* Learning Level selector */}
      <div className="qa-level-row">
        <label className="qa-level-label" htmlFor="learning-level">
          Learning level
        </label>
        <div className="qa-level-buttons" role="group" aria-label="Learning level">
          {LEARNING_LEVELS.map((lvl) => (
            <button
              key={lvl.value}
              id={lvl.value === learningLevel ? 'learning-level' : undefined}
              className={`qa-level-btn ${learningLevel === lvl.value ? 'active' : ''}`}
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
          <span className="qa-level-desc">{selectedLevel.desc}</span>
        )}
      </div>

      {/* Question input */}
      <div className="qa-input-row">
        <textarea
          className="qa-textarea"
          rows={3}
          placeholder="e.g. Explain the OSI model."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isLoading}
          aria-label="Your question"
        />
        <button
          className="qa-ask-btn"
          onClick={handleAsk}
          disabled={isLoading || !question.trim()}
          aria-label="Ask question"
        >
          {isLoading ? (
            <>
              <span className="qa-spinner" aria-hidden="true" />
              Thinking…
            </>
          ) : (
            'Ask'
          )}
        </button>
      </div>

      {/* Status / error message */}
      {status && status.type !== 'loading' && (
        <div className={`qa-status ${status.type}`} role="alert">
          <em className="qa-status-icon" aria-hidden="true">⚠️</em>
          <div className="qa-status-body">
            <strong>{status.title}</strong>
            <span>{status.detail}</span>
          </div>
        </div>
      )}

      {/* Loading indicator */}
      {isLoading && (
        <div className="qa-status loading" role="status">
          <span className="qa-spinner" aria-hidden="true" />
          <div className="qa-status-body">
            <strong>{status.title}</strong>
            <span>{status.detail}</span>
          </div>
        </div>
      )}

      {/* AI Answer */}
      {answer !== null && (
        <div className="qa-answer-section">
          <h3 className="qa-section-label">
            AI Answer
            <span className="qa-level-badge">{selectedLevel?.label}</span>
          </h3>
          <div className="qa-answer-box" role="region" aria-label="AI answer">
            {answer}
          </div>
        </div>
      )}

      {/* Sources */}
      {sources.length > 0 && (
        <div className="qa-sources-section">
          <h3 className="qa-section-label">Sources used</h3>
          <ul className="qa-sources-list" aria-label="Source chunks">
            {sources.map((src, i) => (
              <li key={i} className="qa-source-item">
                <span className="qa-source-file" title={src.filename}>
                  📄 {src.filename}
                </span>
                <span className="qa-source-meta">
                  Chunk {src.chunk_index} &nbsp;·&nbsp; score {src.score.toFixed(3)}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
