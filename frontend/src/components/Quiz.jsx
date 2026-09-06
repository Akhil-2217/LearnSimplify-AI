import { useState } from 'react'
import { api } from '../services/api'
import './Quiz.css'

/**
 * Quiz — Phase 5 quiz generation panel.
 *
 * Props:
 *   docId (string|null) — the document_id returned by the upload endpoint.
 */
export default function Quiz({ docId }) {
  const [status, setStatus] = useState(null)     // null | {type, title, detail}
  const [questions, setQuestions] = useState([]) // QuizQuestion[]
  const [selected, setSelected] = useState({})   // { [questionNumber]: optionLabel }
  const [revealed, setRevealed] = useState({})   // { [questionNumber]: boolean }

  async function handleGenerate() {
    setStatus({ type: 'loading', title: 'Generating quiz…', detail: 'Creating 5 questions from the uploaded document.' })
    setQuestions([])
    setSelected({})
    setRevealed({})

    try {
      const data = await api.quiz(docId)
      setQuestions(data.questions || [])
      setStatus(null)
    } catch (err) {
      setQuestions([])
      setStatus({
        type: 'error',
        title: 'Could not generate quiz',
        detail: err.message || 'An unexpected error occurred. Please try again.',
      })
    }
  }

  function handleSelect(questionNumber, optionLabel) {
    if (revealed[questionNumber]) return  // locked once revealed
    setSelected(prev => ({ ...prev, [questionNumber]: optionLabel }))
  }

  function handleReveal(questionNumber) {
    if (!selected[questionNumber]) return
    setRevealed(prev => ({ ...prev, [questionNumber]: true }))
  }

  function handleRevealAll() {
    const allRevealed = {}
    questions.forEach(q => { allRevealed[q.number] = true })
    setRevealed(allRevealed)
  }

  const isLoading = status?.type === 'loading'
  const answeredCount = Object.keys(selected).length
  const allAnswered = questions.length > 0 && answeredCount === questions.length

  return (
    <div className="quiz-card">
      <h2>🧠 Generate Quiz</h2>
      <p className="quiz-hint">
        Generate 5 multiple-choice questions from the uploaded document.
        Test your understanding of the course material.
      </p>

      {/* Generate button */}
      <button
        className="quiz-btn"
        onClick={handleGenerate}
        disabled={isLoading}
        aria-label="Generate quiz"
      >
        {isLoading ? (
          <>
            <span className="quiz-spinner" aria-hidden="true" />
            Generating…
          </>
        ) : (
          '🧠 Generate Quiz'
        )}
      </button>

      {/* Status / error */}
      {status && status.type !== 'loading' && (
        <div className={`quiz-status ${status.type}`} role="alert">
          <em className="quiz-status-icon" aria-hidden="true">⚠️</em>
          <div className="quiz-status-body">
            <strong>{status.title}</strong>
            <span>{status.detail}</span>
          </div>
        </div>
      )}

      {/* Loading */}
      {isLoading && (
        <div className="quiz-status loading" role="status">
          <span className="quiz-spinner" aria-hidden="true" />
          <div className="quiz-status-body">
            <strong>{status.title}</strong>
            <span>{status.detail}</span>
          </div>
        </div>
      )}

      {/* Quiz questions */}
      {questions.length > 0 && (
        <div className="quiz-questions" aria-label="Quiz questions">
          {/* Progress bar */}
          <div className="quiz-progress-row">
            <span className="quiz-progress-label">
              {answeredCount}/{questions.length} answered
            </span>
            <div className="quiz-progress-bar" aria-hidden="true">
              <div
                className="quiz-progress-fill"
                style={{ width: `${(answeredCount / questions.length) * 100}%` }}
              />
            </div>
            {allAnswered && (
              <button
                className="quiz-reveal-all-btn"
                onClick={handleRevealAll}
                type="button"
              >
                Reveal all
              </button>
            )}
          </div>

          {questions.map((q) => {
            const sel = selected[q.number]
            const isRevealed = revealed[q.number]
            const isCorrect = sel === q.correct_answer

            return (
              <div
                key={q.number}
                className={`quiz-question-block ${isRevealed ? (isCorrect ? 'correct' : 'incorrect') : ''}`}
              >
                <p className="quiz-question-text">
                  <span className="quiz-q-number">Q{q.number}.</span> {q.question}
                </p>

                <ul className="quiz-options" role="radiogroup" aria-label={`Options for question ${q.number}`}>
                  {q.options.map((opt) => {
                    let optClass = 'quiz-option'
                    if (sel === opt.label) optClass += ' selected'
                    if (isRevealed) {
                      if (opt.label === q.correct_answer) optClass += ' correct-option'
                      else if (sel === opt.label) optClass += ' wrong-option'
                    }
                    return (
                      <li key={opt.label}>
                        <button
                          className={optClass}
                          onClick={() => handleSelect(q.number, opt.label)}
                          disabled={isRevealed}
                          aria-pressed={sel === opt.label}
                          type="button"
                        >
                          <span className="quiz-option-label">{opt.label}</span>
                          <span className="quiz-option-text">{opt.text}</span>
                        </button>
                      </li>
                    )
                  })}
                </ul>

                {/* Check answer button */}
                {!isRevealed && sel && (
                  <button
                    className="quiz-check-btn"
                    onClick={() => handleReveal(q.number)}
                    type="button"
                  >
                    Check answer
                  </button>
                )}

                {/* Explanation */}
                {isRevealed && (
                  <div className={`quiz-explanation ${isCorrect ? 'correct' : 'incorrect'}`}>
                    <strong>{isCorrect ? '✓ Correct!' : `✗ Incorrect. Correct answer: ${q.correct_answer}`}</strong>
                    <p>{q.explanation}</p>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
