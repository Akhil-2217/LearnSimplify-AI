"""
quiz.py — POST /api/quiz route.

Phase 5: generates 5 multiple-choice questions based on the uploaded document
using the existing RAG pipeline and IBM Granite.
"""

from __future__ import annotations

import json
import re

from fastapi import APIRouter, HTTPException

from app.models.upload_models import (
    QuizOption,
    QuizQuestion,
    QuizRequest,
    QuizResponse,
)

from app.rag.retriever import retrieve
from app.rag.vector_store import get_doc_ids

from app.services import watsonx_service

from app.services.watsonx_service import (
    WatsonxAuthError,
    WatsonxConfigError,
    WatsonxEmptyResponse,
    WatsonxError,
    WatsonxNetworkError,
)

router = APIRouter()

_QUIZ_TOP_K = 10
_MAX_CONTEXT_CHARS = 4000


# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────

def build_quiz_prompt(context_chunks: list) -> str:
    """Build a quiz-generation prompt from document chunks."""

    context_parts: list[str] = []
    total_chars = 0

    for chunk in context_chunks:
        snippet = chunk.text.strip()

        if not snippet:
            continue

        remaining = _MAX_CONTEXT_CHARS - total_chars

        if remaining <= 0:
            break

        if len(snippet) > remaining:
            if remaining > 100:
                context_parts.append(snippet[:remaining])
            break

        context_parts.append(snippet)
        total_chars += len(snippet)

    context_text = "\n\n---\n\n".join(context_parts)

    return f"""
You are an educational quiz generator.

Create exactly 5 multiple-choice questions using ONLY the course material below.

STRICT OUTPUT RULES:

1. Return ONLY a valid JSON array.
2. Do NOT use markdown.
3. Do NOT use ```json.
4. Do NOT write anything before or after the JSON.
5. The array MUST contain exactly 5 objects.
6. Every question MUST contain exactly 4 options.
7. Option labels MUST be exactly A, B, C, D.
8. correct_answer MUST be exactly A, B, C, or D.
9. Keep questions and options short.
10. Keep explanations short.
11. Do not repeat questions.
12. Do not invent information that is not present in the course material.

Use EXACTLY this structure:

[
  {{
    "number": 1,
    "question": "Question text",
    "options": [
      {{"label": "A", "text": "Option A"}},
      {{"label": "B", "text": "Option B"}},
      {{"label": "C", "text": "Option C"}},
      {{"label": "D", "text": "Option D"}}
    ],
    "correct_answer": "A",
    "explanation": "Short explanation."
  }},
  {{
    "number": 2,
    "question": "Question text",
    "options": [
      {{"label": "A", "text": "Option A"}},
      {{"label": "B", "text": "Option B"}},
      {{"label": "C", "text": "Option C"}},
      {{"label": "D", "text": "Option D"}}
    ],
    "correct_answer": "B",
    "explanation": "Short explanation."
  }},
  {{
    "number": 3,
    "question": "Question text",
    "options": [
      {{"label": "A", "text": "Option A"}},
      {{"label": "B", "text": "Option B"}},
      {{"label": "C", "text": "Option C"}},
      {{"label": "D", "text": "Option D"}}
    ],
    "correct_answer": "C",
    "explanation": "Short explanation."
  }},
  {{
    "number": 4,
    "question": "Question text",
    "options": [
      {{"label": "A", "text": "Option A"}},
      {{"label": "B", "text": "Option B"}},
      {{"label": "C", "text": "Option C"}},
      {{"label": "D", "text": "Option D"}}
    ],
    "correct_answer": "D",
    "explanation": "Short explanation."
  }},
  {{
    "number": 5,
    "question": "Question text",
    "options": [
      {{"label": "A", "text": "Option A"}},
      {{"label": "B", "text": "Option B"}},
      {{"label": "C", "text": "Option C"}},
      {{"label": "D", "text": "Option D"}}
    ],
    "correct_answer": "A",
    "explanation": "Short explanation."
  }}
]

COURSE MATERIAL:

{context_text}

END OF COURSE MATERIAL.

Return ONLY the JSON array.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# JSON extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> str:
    """
    Extract a JSON array from Granite output.

    Handles:
    - plain JSON
    - ```json ... ```
    - extra text surrounding the JSON
    """

    if not text:
        return ""

    text = text.strip()

    # Remove markdown code fences.
    text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```\s*", "", text)

    # Find the JSON array.
    start = text.find("[")
    end = text.rfind("]")

    if start == -1 or end == -1 or end <= start:
        return text

    return text[start:end + 1].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Parse and validate quiz
# ─────────────────────────────────────────────────────────────────────────────

def parse_quiz_response(raw: str) -> list[QuizQuestion]:
    """
    Parse Granite output into QuizQuestion objects.

    Raises ValueError if the response is invalid.
    """

    json_str = _extract_json(raw)

    if not json_str:
        raise ValueError("Model returned an empty quiz.")

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Model output is not valid JSON: {exc}"
        ) from exc

    if not isinstance(data, list):
        raise ValueError("Expected a JSON array.")

    if len(data) < 5:
        raise ValueError(
            f"Expected 5 questions but received {len(data)}."
        )

    questions: list[QuizQuestion] = []

    for i, item in enumerate(data[:5]):

        if not isinstance(item, dict):
            raise ValueError(
                f"Question {i + 1} is not a JSON object."
            )

        question_text = str(
            item.get("question", "")
        ).strip()

        if not question_text:
            raise ValueError(
                f"Question {i + 1} has no question text."
            )

        raw_options = item.get("options", [])

        if not isinstance(raw_options, list):
            raise ValueError(
                f"Question {i + 1} options are invalid."
            )

        if len(raw_options) != 4:
            raise ValueError(
                f"Question {i + 1} must have exactly 4 options."
            )

        options: list[QuizOption] = []

        for option in raw_options:

            if not isinstance(option, dict):
                raise ValueError(
                    f"Invalid option in question {i + 1}."
                )

            label = str(
                option.get("label", "")
            ).strip().upper()

            option_text = str(
                option.get("text", "")
            ).strip()

            if label not in {"A", "B", "C", "D"}:
                raise ValueError(
                    f"Invalid option label in question {i + 1}."
                )

            if not option_text:
                raise ValueError(
                    f"Empty option in question {i + 1}."
                )

            options.append(
                QuizOption(
                    label=label,
                    text=option_text,
                )
            )

        labels = [option.label for option in options]

        if sorted(labels) != ["A", "B", "C", "D"]:
            raise ValueError(
                f"Question {i + 1} must contain A, B, C and D."
            )

        correct_answer = str(
            item.get("correct_answer", "")
        ).strip().upper()

        if correct_answer not in {"A", "B", "C", "D"}:
            raise ValueError(
                f"Invalid correct answer in question {i + 1}."
            )

        explanation = str(
            item.get("explanation", "")
        ).strip()

        questions.append(
            QuizQuestion(
                number=i + 1,
                question=question_text,
                options=options,
                correct_answer=correct_answer,
                explanation=explanation,
            )
        )

    return questions


# ─────────────────────────────────────────────────────────────────────────────
# API route
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/quiz",
    response_model=QuizResponse,
    tags=["Learning"],
)
async def generate_quiz(
    request: QuizRequest,
) -> QuizResponse:

    # 1. Check whether documents exist.
    doc_ids = get_doc_ids()

    if not doc_ids:
        raise HTTPException(
            status_code=404,
            detail=(
                "No documents have been indexed yet. "
                "Please upload a document before generating a quiz."
            ),
        )

    # 2. Validate requested document.
    if (
        request.document_id
        and request.document_id not in doc_ids
    ):
        raise HTTPException(
            status_code=404,
            detail=(
                f"Document ID '{request.document_id}' not found. "
                "Please upload the document first."
            ),
        )

    # 3. Retrieve document content.
    try:

        chunks = retrieve(
            question=(
                "important concepts definitions facts "
                "topics and key ideas in this document"
            ),
            doc_id=request.document_id,
            top_k=_QUIZ_TOP_K,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="Retrieval failed due to an internal error.",
        ) from exc

    if not chunks:
        raise HTTPException(
            status_code=422,
            detail=(
                "Could not retrieve any content from the document "
                "to generate a quiz."
            ),
        )

    # 4. Build Granite prompt.
    prompt = build_quiz_prompt(chunks)

    # 5. Call IBM Granite.
    try:

        raw_output = watsonx_service.generate_text(prompt)

    except WatsonxConfigError as exc:

        raise HTTPException(
            status_code=503,
            detail=(
                "The AI service is not configured. "
                "Please contact the administrator."
            ),
        ) from exc

    except WatsonxAuthError as exc:

        raise HTTPException(
            status_code=503,
            detail=(
                "Authentication with IBM watsonx.ai failed. "
                "Please contact the administrator."
            ),
        ) from exc

    except WatsonxNetworkError as exc:

        raise HTTPException(
            status_code=503,
            detail=(
                "Could not reach the IBM watsonx.ai service. "
                "Please try again later."
            ),
        ) from exc

    except WatsonxEmptyResponse as exc:

        raise HTTPException(
            status_code=502,
            detail=(
                "The AI model returned an empty response. "
                "Please try again."
            ),
        ) from exc

    except WatsonxError as exc:

        raise HTTPException(
            status_code=502,
            detail=(
                "The AI service encountered an error. "
                "Please try again."
            ),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "An unexpected error occurred while "
                "generating the quiz."
            ),
        ) from exc

    # 6. Parse Granite response.
    try:

        questions = parse_quiz_response(raw_output)

    except ValueError as exc:

        # Print the raw response in the backend terminal.
        # This helps debugging without exposing it to the frontend.
        print("\n===== INVALID GRANITE QUIZ RESPONSE =====")
        print(raw_output)
        print("===== END GRANITE RESPONSE =====\n")

        raise HTTPException(
            status_code=502,
            detail=(
                "The AI model did not return a valid quiz. "
                "Please try again."
            ),
        ) from exc

    # 7. Return quiz.
    return QuizResponse(
        success=True,
        questions=questions,
        message=(
            f"Generated {len(questions)} quiz questions "
            "from the uploaded document."
        ),
    )