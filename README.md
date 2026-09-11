# LearnSimplify AI

## 🚀 Live Demo

👉 **[Open LearnSimplify AI](https://akhil-2217.github.io/LearnSimplify-AI/)**

**GitHub Repository:** https://github.com/Akhil-2217/LearnSimplify-AI
> **AICTE-2026 Problem Statement No. 19**  
> AI-Powered Course Content Simplification Agent

---

## Overview

**LearnSimplify AI** is an AI-powered educational assistant that helps students understand dense academic material at their own level. Students upload a course document (PDF, DOCX, or TXT), and the system uses **Retrieval-Augmented Generation (RAG)** combined with **IBM Granite** (via IBM watsonx.ai) to answer questions, generate summaries, create quizzes, and support multi-turn conversations — all grounded in the uploaded material.

This project was built as a complete solution to **AICTE-2026 Problem Statement No. 19**: an intelligent system that simplifies complex course content and presents it in a way suited to each learner's level.

---

## Features

| Feature | Description |
|---|---|
| **Document Upload** | Upload PDF, DOCX, or TXT course files (up to 10 MB). Text is extracted, cleaned, chunked, and indexed automatically. |
| **RAG Pipeline** | Semantic chunking, sentence-transformer embeddings, FAISS vector search — answers are always grounded in the uploaded material. |
| **Q&A (IBM Granite)** | Ask any question about the uploaded document. IBM Granite generates a grounded answer at your chosen learning level. |
| **Learning Levels** | Four levels: **Beginner** (everyday language), **Intermediate** (clear language + key terms), **Advanced** (technical depth), **Expert** (full depth + edge cases). |
| **Summarization** | Generate **Short** (3–5 sentences), **Medium** (2–3 paragraphs), or **Detailed** (topic-by-topic) summaries. |
| **Quiz Generation** | Auto-generate 5 MCQs from the document. Select answers, check them, and read explanations. |
| **Conversation** | Multi-turn follow-up questions with conversation history stored server-side. Context is injected into each prompt so the AI "remembers" the conversation. |

---

## Architecture

```
Student Browser
      │
      ▼
React Frontend (Vite, port 5173)
      │   /api/*  proxied — no backend URL or API key in frontend code
      ▼
FastAPI Backend (Uvicorn, port 8000)
      │
      ├── GET  /api/health                 Liveness probe
      ├── POST /api/upload                 Document parsing → RAG indexing
      ├── POST /api/retrieve               Raw chunk retrieval (no LLM)
      ├── POST /api/ask                    Q&A: RAG + IBM Granite
      ├── POST /api/summarize              Summary: RAG + IBM Granite
      ├── POST /api/quiz                   Quiz: RAG + IBM Granite
      ├── POST /api/conversation/ask       Follow-up Q&A with history
      ├── GET  /api/conversation/{doc_id}  Fetch conversation history
      └── DELETE /api/conversation/{doc_id} Clear conversation history
```

### RAG → IBM Granite Workflow

```
Upload Document
      ↓
Text Extraction    (PyMuPDF / python-docx / built-in open())
      ↓
Text Cleaning      (normalise unicode, collapse whitespace)
      ↓
Chunking           (~1000 chars, 150-char overlap, sentence-boundary aware)
      ↓
Embeddings         (sentence-transformers all-MiniLM-L6-v2, 384-dim, L2-normalised)
      ↓
FAISS Index        (IndexFlatIP — cosine similarity, stored in memory)
      ↓
──────────────────────────────────────────────────────
User asks a question (POST /api/ask)
      ↓
Question Embedding (same model)
      ↓
Similarity Search → Top-5 chunks
      ↓
Grounded Prompt    (system instructions + level + course context + question)
      ↓
IBM Granite        (ibm/granite-4-h-small via watsonx.ai)
      ↓
Answer + Source Citations → Student
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, Vite, JavaScript (JSX), CSS |
| Backend | Python 3.11+, FastAPI, Uvicorn |
| AI Model | IBM Granite (`ibm/granite-4-h-small`) |
| AI Platform | IBM watsonx.ai |
| Document Parsing | PyMuPDF (PDF), python-docx (DOCX), stdlib (TXT) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim) |
| Vector Search | FAISS (`faiss-cpu`, inner-product cosine similarity) |
| HTTP Client | `requests` (IBM IAM + watsonx.ai API calls) |
| Testing | `pytest`, `starlette.testclient` |

---

## Project Structure

```
course-content-simplifier/
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── DocumentUpload.jsx   Upload panel (drag-and-drop + file picker)
│   │   │   ├── DocumentUpload.css
│   │   │   ├── QuestionAnswer.jsx   Q&A panel (single question + learning level)
│   │   │   ├── QuestionAnswer.css
│   │   │   ├── Conversation.jsx     Multi-turn conversation panel
│   │   │   ├── Conversation.css
│   │   │   ├── Summarize.jsx        Summarization panel (short/medium/detailed)
│   │   │   ├── Summarize.css
│   │   │   ├── Quiz.jsx             Interactive MCQ quiz panel
│   │   │   └── Quiz.css
│   │   ├── services/
│   │   │   └── api.js               Fetch wrapper — all API calls go here
│   │   ├── App.jsx                  Root component — layout shell
│   │   ├── App.css                  Layout styles
│   │   ├── main.jsx                 React entry point
│   │   └── index.css                Global CSS variables and reset
│   ├── index.html
│   ├── vite.config.js               Proxy /api/* → backend:8000
│   └── package.json
│
├── backend/
│   ├── app/
│   │   ├── main.py                  FastAPI app + CORS + router registration
│   │   ├── routes/
│   │   │   ├── health.py            GET  /api/health
│   │   │   ├── upload.py            POST /api/upload
│   │   │   ├── retrieve.py          POST /api/retrieve
│   │   │   ├── ask.py               POST /api/ask
│   │   │   ├── summarize.py         POST /api/summarize
│   │   │   ├── quiz.py              POST /api/quiz
│   │   │   └── conversation.py      POST/GET/DELETE /api/conversation/*
│   │   ├── services/
│   │   │   ├── watsonx_service.py   IBM IAM auth + Granite text generation
│   │   │   ├── conversation_store.py Server-side conversation history (in-memory)
│   │   │   └── document_processor.py Text extraction + RAG indexing
│   │   ├── rag/
│   │   │   ├── chunker.py           Sentence-boundary text chunking
│   │   │   ├── embedder.py          all-MiniLM-L6-v2 embeddings
│   │   │   ├── vector_store.py      FAISS in-memory index
│   │   │   └── retriever.py         Public RAG retrieval API
│   │   └── models/
│   │       └── upload_models.py     Pydantic request/response models
│   ├── tests/
│   │   ├── test_health.py           Phase 1 tests
│   │   ├── test_upload.py           Phase 2 tests
│   │   ├── test_ask.py              Phase 3/4 tests (IBM mocked)
│   │   ├── test_phase5.py           Phase 5 tests (summarize, quiz, learning levels)
│   │   └── test_phase6.py           Phase 6 tests (conversation history)
│   ├── requirements.txt
│   ├── .env.example                 Template — copy to .env, fill API key
│   └── .env                        ← NEVER commit (git-ignored)
│
├── .gitignore
└── README.md
```

---

## Environment Variables

Set these in `backend/.env` (copy from `backend/.env.example`):

| Variable | Required | Description |
|---|---|---|
| `WATSONX_API_KEY` | ✅ | Your IBM Cloud API key — **keep secret, never commit** |
| `WATSONX_PROJECT_ID` | ✅ | IBM watsonx project ID (pre-filled in `.env.example`) |
| `WATSONX_URL` | ✅ | watsonx.ai base URL (pre-filled) |
| `WATSONX_MODEL_ID` | ✅ | Granite model ID (pre-filled) |

> **Security note:** `backend/.env` is listed in `.gitignore`. The API key is never sent to the frontend, never logged, and never included in error responses.

---

## Setup Instructions

### Prerequisites

- Python 3.11 or newer
- Node.js 18 or newer
- `pip`

### 1 — Clone the repository

```bash
git clone <repo-url>
cd course-content-simplifier
```

### 2 — Backend setup

```bash
cd backend

# Create and activate a virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS / Linux:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3 — Configure the API key

```bash
# Windows:
copy .env.example .env
# macOS / Linux:
# cp .env.example .env
```

Open `backend/.env` and fill in your IBM Cloud API key:

```
WATSONX_API_KEY=your_actual_api_key_here
```

The other three variables (`WATSONX_PROJECT_ID`, `WATSONX_URL`, `WATSONX_MODEL_ID`) are pre-filled in `.env.example` and do not need to change.

### 4 — Frontend setup

```bash
cd frontend
npm install
```

---

## Running the Project

### Start the backend

```bash
cd backend

# Windows:
.venv\Scripts\activate
# macOS / Linux:
# source .venv/bin/activate

uvicorn app.main:app --reload --port 8000
```

- API base:  http://127.0.0.1:8000
- Swagger UI: http://127.0.0.1:8000/docs

### Start the frontend

```bash
cd frontend
npm run dev
```

- App: http://localhost:5173

---

## Testing

Run the full test suite (all tests mock the IBM API — no real API key needed):

```bash
cd backend
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # macOS / Linux

pytest tests/ -v
```

Expected output: **129 passed** (1 deprecation warning from `starlette.testclient` — not a failure).

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Liveness probe |
| POST | `/api/upload` | Upload and index a document (PDF/DOCX/TXT) |
| POST | `/api/retrieve` | Raw RAG chunk retrieval (no LLM) |
| POST | `/api/ask` | Ask a question — RAG + IBM Granite answer |
| POST | `/api/summarize` | Summarize the document — RAG + IBM Granite |
| POST | `/api/quiz` | Generate 5 MCQs — RAG + IBM Granite |
| POST | `/api/conversation/ask` | Follow-up question with conversation context |
| GET | `/api/conversation/{doc_id}` | Retrieve conversation history |
| DELETE | `/api/conversation/{doc_id}` | Clear conversation history |

### POST /api/ask

**Request:**
```json
{
  "question": "Explain the OSI model.",
  "document_id": "optional-doc-id",
  "top_k": 5,
  "learning_level": "beginner"
}
```

**Response:**
```json
{
  "success": true,
  "answer": "The OSI model is a conceptual framework with seven layers…",
  "sources": [
    { "filename": "lecture_notes.pdf", "chunk_index": 3, "score": 0.87 }
  ],
  "message": "Answer generated from 1 source chunk(s)."
}
```

**Error responses:**

| HTTP | Condition |
|---|---|
| 400 | Empty question |
| 404 | No documents indexed / unknown document_id |
| 422 | Retrieval validation error |
| 502 | IBM API error / empty model response |
| 503 | API key missing / authentication failed / network error |

### POST /api/summarize

**Request:**
```json
{
  "document_id": "optional-doc-id",
  "length": "medium"
}
```
`length` is one of: `short` | `medium` | `detailed`

### POST /api/quiz

**Request:**
```json
{
  "document_id": "optional-doc-id"
}
```

### POST /api/conversation/ask

**Request:**
```json
{
  "question": "What about the data link layer specifically?",
  "doc_id": "doc-id-from-upload",
  "top_k": 5,
  "learning_level": "intermediate"
}
```

---

## Demo Workflow

1. Open http://localhost:5173.
2. **Upload** your course document (PDF, DOCX, or TXT) using the drag-and-drop zone or file picker.
3. Wait for the **"Document ready"** confirmation card showing filename, type, characters extracted, and chunks indexed.
4. Click the **Q&A** chip and type a question (e.g. *"Explain the OSI model"*). Select a learning level and click **Ask**.
5. Read the AI-generated answer grounded in your document. Expand **Sources** to see which chunks were used.
6. Click the **Conversation** chip for multi-turn follow-ups (e.g. *"Tell me more about layer 3"*).
7. Click the **Summary** chip and select a length (Short / Medium / Detailed). Click **Generate Summary**.
8. Click the **Quiz** chip and click **Generate Quiz**. Answer each MCQ, click **Check answer**, and read the explanation.

---

## IBM Granite Integration

| Setting | Value |
|---|---|
| Model | `ibm/granite-4-h-small` |
| Platform | IBM watsonx.ai |
| Base URL | `https://us-south.ml.cloud.ibm.com` |
| Generation endpoint | `/ml/v1/text/generation?version=2023-05-29` |
| Auth | IBM IAM bearer token (exchanged from `WATSONX_API_KEY`) |
| Token caching | Tokens are cached in-memory and refreshed 5 minutes before expiry |

The service [`backend/app/services/watsonx_service.py`](backend/app/services/watsonx_service.py) is the **single point of contact** with IBM APIs. The API key never leaves the backend and is never logged or returned by any endpoint.

---

## Security

- `WATSONX_API_KEY` lives **only** in `backend/.env`.
- `.env` is git-ignored and **never** committed.
- The frontend **never** receives, stores, or transmits the API key.
- The API key is **never** logged or returned by any endpoint.
- IBM credentials are **never** included in error responses to the client.
- Internal stack traces are suppressed; users see only friendly messages.

---

## Phase Summary

| Phase | What was built |
|---|---|
| **Phase 1** | Project scaffold, FastAPI backend, health endpoint, React + Vite frontend shell |
| **Phase 2** | Document upload endpoint, text extraction (PDF/DOCX/TXT), file validation |
| **Phase 3** | RAG pipeline: chunking, embeddings (all-MiniLM-L6-v2), FAISS vector store, `/api/retrieve` |
| **Phase 4** | IBM Granite integration via watsonx.ai, `/api/ask` Q&A endpoint with grounded prompts |
| **Phase 5** | Learning levels (Beginner/Intermediate/Advanced/Expert), `/api/summarize`, `/api/quiz` |
| **Phase 6** | Multi-turn conversation with history (`/api/conversation/*`), server-side ConversationStore |
| **Phase 7** | UI/UX polish (step flow, feature nav, responsive design), improved error messages, code cleanup, full README |

---

*LearnSimplify AI — AICTE-2026 Problem Statement No. 19*
