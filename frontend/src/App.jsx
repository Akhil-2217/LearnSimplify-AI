import { useState, useEffect } from 'react'
import { api } from './services/api'
import Conversation from './components/Conversation'
import DocumentUpload from './components/DocumentUpload'
import QuestionAnswer from './components/QuestionAnswer'
import Summarize from './components/Summarize'
import Quiz from './components/Quiz'
import './App.css'

/**
 * App — Phase 7 shell.
 *
 * Layout:
 *   Header → Backend status → Step 1 (Upload) → Step 2 (Interact panels) → Footer
 *
 * The "interact" section is only shown after a document is uploaded.
 * Each feature panel (Q&A, Conversation, Summarize, Quiz) is labelled
 * clearly so users can navigate the single-page layout easily.
 */
export default function App() {
  const [backendStatus, setBackendStatus] = useState('checking')
  const [backendMessage, setBackendMessage] = useState('Connecting to backend…')
  const [uploadedDoc, setUploadedDoc] = useState(null)

  useEffect(() => {
    api
      .health()
      .then((data) => {
        setBackendStatus('ok')
        setBackendMessage(data.message)
      })
      .catch(() => {
        setBackendStatus('error')
        setBackendMessage(
          'Backend is unreachable. Make sure the FastAPI server is running on port 8000.',
        )
      })
  }, [])

  function handleUploadSuccess(data) {
    setUploadedDoc(data)
  }

  return (
    <div className="app-wrapper">

      {/* ── Header ─────────────────────────────────────────── */}
      <header className="app-header">
        <div className="header-brand">
          <span className="header-icon" aria-hidden="true">🎓</span>
          <div className="header-text">
            <h1>LearnSimplify AI</h1>
            <p>AI-Powered Course Content Simplification Agent</p>
          </div>
        </div>
        <div className="header-badge">IBM watsonx.ai · Granite</div>
      </header>

      {/* ── Main ───────────────────────────────────────────── */}
      <main className="app-main">

        {/* Backend connectivity indicator */}
        <div className={`backend-indicator ${backendStatus}`}>
          <span className={`status-dot ${backendStatus}`} aria-hidden="true" />
          <span className="backend-label">
            {backendStatus === 'checking' && 'Connecting to backend…'}
            {backendStatus === 'ok'       && backendMessage}
            {backendStatus === 'error'    && backendMessage}
          </span>
        </div>

        {/* ── Step 1: Upload ────────────────────────────────── */}
        <section className="app-section" aria-labelledby="section-upload-title">
          <div className="section-header">
            <span className="section-step">Step 1</span>
            <h2 id="section-upload-title" className="section-title">Upload Course Material</h2>
            <p className="section-desc">
              Upload your PDF, DOCX, or TXT course file. The document is parsed,
              chunked, and indexed so the AI can answer questions about it.
            </p>
          </div>

          <DocumentUpload onSuccess={handleUploadSuccess} />

          {/* Document info card — shown after successful upload */}
          {uploadedDoc && (
            <div className="doc-ready-card">
              <div className="doc-ready-header">
                <span className="doc-ready-icon" aria-hidden="true">✅</span>
                <strong>Document ready — indexed and ready for Q&amp;A</strong>
              </div>
              <ul className="doc-meta">
                <li>
                  <span>Filename</span>
                  <span className="doc-meta-value">{uploadedDoc.filename}</span>
                </li>
                <li>
                  <span>Type</span>
                  <span className="doc-meta-value">{uploadedDoc.file_type.toUpperCase()}</span>
                </li>
                <li>
                  <span>Characters extracted</span>
                  <span className="doc-meta-value">{uploadedDoc.text_length.toLocaleString()}</span>
                </li>
                <li>
                  <span>Chunks indexed</span>
                  <span className="doc-meta-value">{uploadedDoc.chunk_count}</span>
                </li>
              </ul>
            </div>
          )}
        </section>

        {/* ── Step 2: Interact (only shown after upload) ────── */}
        {uploadedDoc && (
          <section className="app-section app-section--interact" aria-labelledby="section-interact-title">
            <div className="section-header">
              <span className="section-step">Step 2</span>
              <h2 id="section-interact-title" className="section-title">Explore Your Document</h2>
              <p className="section-desc">
                Ask questions, read a summary, test yourself with a quiz, or have a
                multi-turn conversation — all grounded in your uploaded material.
              </p>
            </div>

            {/* Feature nav chips — quick-scroll links */}
            <nav className="feature-nav" aria-label="Feature navigation">
              <a href="#panel-qa"           className="feature-chip">💬 Q&amp;A</a>
              <a href="#panel-conversation" className="feature-chip">🗨 Conversation</a>
              <a href="#panel-summarize"    className="feature-chip">📋 Summary</a>
              <a href="#panel-quiz"         className="feature-chip">🧠 Quiz</a>
            </nav>

            {/* Q&A panel */}
            <div id="panel-qa" className="panel-anchor">
              <QuestionAnswer docId={uploadedDoc.doc_id} />
            </div>

            {/* Conversation panel */}
            <div id="panel-conversation" className="panel-anchor">
              <Conversation docId={uploadedDoc.doc_id} />
            </div>

            {/* Summarize panel */}
            <div id="panel-summarize" className="panel-anchor">
              <Summarize docId={uploadedDoc.doc_id} />
            </div>

            {/* Quiz panel */}
            <div id="panel-quiz" className="panel-anchor">
              <Quiz docId={uploadedDoc.doc_id} />
            </div>
          </section>
        )}

        {/* ── Upload prompt when no doc yet ─────────────────── */}
        {!uploadedDoc && backendStatus === 'ok' && (
          <div className="no-doc-hint">
            <span aria-hidden="true">⬆️</span>
            Upload a course document above to unlock all AI features.
          </div>
        )}

      </main>

      {/* ── Footer ─────────────────────────────────────────── */}
      <footer className="app-footer">
        <span>LearnSimplify AI</span>
        <span className="footer-sep" aria-hidden="true">·</span>
        <span>AICTE-2026 Problem Statement No. 19</span>
        <span className="footer-sep" aria-hidden="true">·</span>
        <span>Powered by IBM watsonx.ai &amp; Granite</span>
      </footer>
    </div>
  )
}
