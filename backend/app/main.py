"""
LearnSimplify AI — FastAPI backend entry point.
Phase 6: Conversation history and follow-up question support.
"""

from dotenv import load_dotenv

# Load variables from backend/.env before importing services
# that read environment variables.
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.ask import router as ask_router
from app.routes.conversation import router as conversation_router
from app.routes.health import router as health_router
from app.routes.quiz import router as quiz_router
from app.routes.retrieve import router as retrieve_router
from app.routes.summarize import router as summarize_router
from app.routes.upload import router as upload_router


app = FastAPI(
    title="LearnSimplify AI",
    description="AI Course Content Simplification Agent — Backend API",
    version="1.0.0",
)


# Allow the Vite frontend to communicate with the backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# API Routers
# ---------------------------------------------------------------------------

app.include_router(
    health_router,
    prefix="/api",
)

app.include_router(
    upload_router,
    prefix="/api",
)

app.include_router(
    retrieve_router,
    prefix="/api",
)

app.include_router(
    ask_router,
    prefix="/api",
)

app.include_router(
    summarize_router,
    prefix="/api",
)

app.include_router(
    quiz_router,
    prefix="/api",
)

app.include_router(
    conversation_router,
    prefix="/api",
)