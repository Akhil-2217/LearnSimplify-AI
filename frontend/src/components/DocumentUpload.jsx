import { useRef, useState } from 'react'
import { api } from '../services/api'
import './DocumentUpload.css'

const MAX_SIZE_MB = 10
const MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024
const ALLOWED_EXTS = ['.pdf', '.docx', '.txt']
const ALLOWED_MIME = [
  'application/pdf',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  'text/plain',
]

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1_048_576) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1_048_576).toFixed(1)} MB`
}

function getExtension(filename) {
  return filename.slice(filename.lastIndexOf('.')).toLowerCase()
}

/**
 * DocumentUpload — Phase 2 upload panel.
 *
 * Props:
 *   onSuccess(data) — called when the backend returns a successful response.
 */
export default function DocumentUpload({ onSuccess }) {
  const [dragOver, setDragOver] = useState(false)
  const [selectedFile, setSelectedFile] = useState(null)
  const [status, setStatus] = useState(null) // null | {type, title, detail}
  const inputRef = useRef(null)

  // ── file selection ──────────────────────────────────────────────────────────
  function validateAndSet(file) {
    if (!file) return

    const ext = getExtension(file.name)
    if (!ALLOWED_EXTS.includes(ext)) {
      setStatus({
        type: 'error',
        title: 'Unsupported file type',
        detail: `Only PDF, DOCX, and TXT files are allowed. You selected "${ext}".`,
      })
      setSelectedFile(null)
      return
    }

    if (file.size > MAX_SIZE_BYTES) {
      setStatus({
        type: 'error',
        title: 'File too large',
        detail: `Maximum allowed size is ${MAX_SIZE_MB} MB. Your file is ${formatBytes(file.size)}.`,
      })
      setSelectedFile(null)
      return
    }

    setSelectedFile(file)
    setStatus(null)
  }

  function handleInputChange(e) {
    validateAndSet(e.target.files?.[0] ?? null)
    // Reset input so the same file can be re-selected after clearing
    e.target.value = ''
  }

  function handleDrop(e) {
    e.preventDefault()
    setDragOver(false)
    validateAndSet(e.dataTransfer.files?.[0] ?? null)
  }

  function handleClear() {
    setSelectedFile(null)
    setStatus(null)
  }

  // ── upload ─────────────────────────────────────────────────────────────────
  async function handleUpload() {
    if (!selectedFile) return

    setStatus({ type: 'loading', title: 'Uploading document…', detail: 'Please wait.' })

    try {
      const data = await api.upload(selectedFile)
      setStatus({
        type: 'success',
        title: '✓ Document uploaded successfully',
        detail: `${data.filename} · ${data.file_type.toUpperCase()} · ${data.text_length.toLocaleString()} characters extracted`,
      })
      if (onSuccess) onSuccess(data)
    } catch (err) {
      setStatus({
        type: 'error',
        title: 'Upload failed',
        detail: err.message || 'An unexpected error occurred. Please try again.',
      })
    }
  }

  // ── render ─────────────────────────────────────────────────────────────────
  return (
    <div className="upload-card">
      <h2>📄 Upload Course Material</h2>

      {/* Drop zone */}
      <div
        className={`drop-zone${dragOver ? ' drag-over' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
        aria-label="Click or drag a file here to upload"
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.docx,.txt"
          onChange={handleInputChange}
          tabIndex={-1}
        />
        <span className="drop-icon" aria-hidden="true">☁️</span>
        <p className="drop-label">
          <strong>Click to browse</strong> or drag &amp; drop a file here
        </p>
        <p className="drop-hint">PDF, DOCX, TXT &nbsp;·&nbsp; Max {MAX_SIZE_MB} MB</p>
      </div>

      {/* Selected file row */}
      {selectedFile && (
        <div className="selected-file">
          <span aria-hidden="true">📎</span>
          <span className="file-name">{selectedFile.name}</span>
          <span className="file-size">{formatBytes(selectedFile.size)}</span>
          <button className="clear-btn" onClick={handleClear} title="Remove file" aria-label="Remove selected file">
            ✕
          </button>
        </div>
      )}

      {/* Upload button */}
      <button
        className="upload-btn"
        onClick={handleUpload}
        disabled={!selectedFile || status?.type === 'loading'}
      >
        {status?.type === 'loading' ? 'Processing…' : 'Upload & Process'}
      </button>

      {/* Status message */}
      {status && (
        <div className={`upload-status ${status.type}`} role={status.type === 'error' ? 'alert' : 'status'}>
          {status.type === 'loading' ? (
            <span className="spinner" aria-hidden="true" />
          ) : (
            <em className="status-icon" aria-hidden="true">
              {status.type === 'success' ? '✅' : '⚠️'}
            </em>
          )}
          <div className="status-body">
            <strong>{status.title}</strong>
            <span>{status.detail}</span>
          </div>
        </div>
      )}
    </div>
  )
}
