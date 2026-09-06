import { useState } from 'react'
import { api } from '../services/api'
import './Summarize.css'

/**
 * Summarize — Phase 5 document summarisation panel.
 *
 * Props:
 *   docId (string|null) — the document_id returned by the upload endpoint.
 */

const LENGTHS = [
  { value: 'short',    label: 'Short',    desc: '3–5 sentence overview' },
  { value: 'medium',   label: 'Medium',   desc: '2–3 paragraph summary' },
  { value: 'detailed', label: 'Detailed', desc: 'Full topic-by-topic breakdown' },
]

export default function Summarize({ docId }) {
  const [length, setLength] = useState('medium')
  const [status, setStatus] = useState(null)   // null | {type, title, detail}
  const [summary, setSummary] = useState(null) // string | null

  async function handleSummarize() {
    setStatus({ type: 'loading', title: 'Summarising…', detail: 'Reading the document and generating a summary.' })
    setSummary(null)

    try {
      const data = await api.summarize(docId, length)
      setSummary(data.summary)
      setStatus(null)
    } catch (err) {
      setSummary(null)
      setStatus({
        type: 'error',
        title: 'Could not generate summary',
        detail: err.message || 'An unexpected error occurred. Please try again.',
      })
    }
  }

  const isLoading = status?.type === 'loading'
  const selectedLength = LENGTHS.find(l => l.value === length)

  return (
    <div className="sum-card">
      <h2>📋 Summarize Document</h2>
      <p className="sum-hint">
        Generate a summary of the uploaded document using IBM Granite.
        Choose how detailed the summary should be.
      </p>

      {/* Length selector */}
      <div className="sum-length-row">
        <span className="sum-length-label">Summary length</span>
        <div className="sum-length-buttons" role="group" aria-label="Summary length">
          {LENGTHS.map((l) => (
            <button
              key={l.value}
              className={`sum-length-btn ${length === l.value ? 'active' : ''}`}
              onClick={() => setLength(l.value)}
              disabled={isLoading}
              title={l.desc}
              aria-pressed={length === l.value}
              type="button"
            >
              {l.label}
            </button>
          ))}
        </div>
        {selectedLength && (
          <span className="sum-length-desc">{selectedLength.desc}</span>
        )}
      </div>

      {/* Summarize button */}
      <button
        className="sum-btn"
        onClick={handleSummarize}
        disabled={isLoading}
        aria-label="Generate summary"
      >
        {isLoading ? (
          <>
            <span className="sum-spinner" aria-hidden="true" />
            Summarising…
          </>
        ) : (
          '📋 Generate Summary'
        )}
      </button>

      {/* Status / error */}
      {status && status.type !== 'loading' && (
        <div className={`sum-status ${status.type}`} role="alert">
          <em className="sum-status-icon" aria-hidden="true">⚠️</em>
          <div className="sum-status-body">
            <strong>{status.title}</strong>
            <span>{status.detail}</span>
          </div>
        </div>
      )}

      {/* Loading */}
      {isLoading && (
        <div className="sum-status loading" role="status">
          <span className="sum-spinner" aria-hidden="true" />
          <div className="sum-status-body">
            <strong>{status.title}</strong>
            <span>{status.detail}</span>
          </div>
        </div>
      )}

      {/* Summary output */}
      {summary !== null && (
        <div className="sum-result-section">
          <h3 className="sum-section-label">
            Summary
            <span className="sum-length-badge">{selectedLength?.label}</span>
          </h3>
          <div className="sum-result-box" role="region" aria-label="Document summary">
            {summary}
          </div>
        </div>
      )}
    </div>
  )
}
