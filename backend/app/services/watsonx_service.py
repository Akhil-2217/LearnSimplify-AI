"""
watsonx_service.py — IBM Granite integration.

Loads credentials from environment variables, obtains an IBM IAM token,
and calls the watsonx.ai text-generation endpoint.
"""

from __future__ import annotations

import os
import time
from typing import Any

import requests


# ─────────────────────────────────────────────────────────────────────────────
# IBM IAM
# ─────────────────────────────────────────────────────────────────────────────

_IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"

# Current endpoint used by the project.
_GENERATION_PATH = "/ml/v1/text/generation?version=2023-05-29"


# ─────────────────────────────────────────────────────────────────────────────
# Generation parameters
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_PARAMS: dict[str, Any] = {
    "decoding_method": "greedy",

    # Increased from 600 because quiz generation needs enough
    # tokens for 5 questions, 20 options and explanations.
    "max_new_tokens": 1200,

    "min_new_tokens": 20,
    "repetition_penalty": 1.1,
}


# ─────────────────────────────────────────────────────────────────────────────
# Token cache
# ─────────────────────────────────────────────────────────────────────────────

_cached_token: str | None = None
_token_expiry: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

def _get_config() -> dict[str, str]:
    """
    Read watsonx configuration from environment variables.
    """

    api_key = os.environ.get("WATSONX_API_KEY", "").strip()

    if not api_key:
        raise WatsonxConfigError(
            "WATSONX_API_KEY is not set."
        )

    project_id = os.environ.get(
        "WATSONX_PROJECT_ID",
        "81290637-28b4-4a3d-bbc1-5c1040e15482",
    ).strip()

    url = os.environ.get(
        "WATSONX_URL",
        "https://us-south.ml.cloud.ibm.com",
    ).strip().rstrip("/")

    model_id = os.environ.get(
        "WATSONX_MODEL_ID",
        "ibm/granite-4-h-small",
    ).strip()

    if not project_id:
        raise WatsonxConfigError(
            "WATSONX_PROJECT_ID is not configured."
        )

    if not url:
        raise WatsonxConfigError(
            "WATSONX_URL is not configured."
        )

    if not model_id:
        raise WatsonxConfigError(
            "WATSONX_MODEL_ID is not configured."
        )

    return {
        "api_key": api_key,
        "project_id": project_id,
        "url": url,
        "model_id": model_id,
    }


# ─────────────────────────────────────────────────────────────────────────────
# IAM token
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_iam_token(api_key: str) -> tuple[str, float]:
    """
    Exchange IBM Cloud API key for IAM bearer token.
    """

    try:
        response = requests.post(
            _IAM_TOKEN_URL,
            data={
                "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                "apikey": api_key,
            },
            headers={
                "Accept": "application/json",
            },
            timeout=30,
        )

    except requests.exceptions.Timeout as exc:
        raise WatsonxNetworkError(
            "IBM IAM token request timed out."
        ) from exc

    except requests.exceptions.ConnectionError as exc:
        raise WatsonxNetworkError(
            "Could not connect to IBM IAM service."
        ) from exc

    if response.status_code != 200:
        raise WatsonxAuthError(
            f"IBM IAM authentication failed "
            f"(HTTP {response.status_code}). "
            "Check that WATSONX_API_KEY is correct."
        )

    try:
        body = response.json()
        token = body["access_token"]

    except (ValueError, KeyError) as exc:
        raise WatsonxAuthError(
            "IBM IAM returned an invalid authentication response."
        ) from exc

    expires_in = int(body.get("expires_in", 3600))

    # Refresh 5 minutes before expiration.
    expiry = time.time() + expires_in - 300

    return token, expiry


def _get_token(api_key: str) -> str:
    """
    Return cached IAM token or request a new one.
    """

    global _cached_token, _token_expiry

    if _cached_token and time.time() < _token_expiry:
        return _cached_token

    token, expiry = _fetch_iam_token(api_key)

    _cached_token = token
    _token_expiry = expiry

    return token


def _invalidate_token() -> None:
    """
    Clear cached IAM token.
    """

    global _cached_token, _token_expiry

    _cached_token = None
    _token_expiry = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Granite generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_text(prompt: str) -> str:
    """
    Send a prompt to IBM Granite and return generated text.
    """

    config = _get_config()

    token = _get_token(config["api_key"])

    url = config["url"] + _GENERATION_PATH

    payload = {
        "model_id": config["model_id"],
        "input": prompt,
        "parameters": _DEFAULT_PARAMS,
        "project_id": config["project_id"],
    }

    try:
        response = requests.post(
            url,
            json=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=90,
        )

    except requests.exceptions.Timeout as exc:
        raise WatsonxNetworkError(
            "The request to IBM Granite timed out. Please try again."
        ) from exc

    except requests.exceptions.ConnectionError as exc:
        raise WatsonxNetworkError(
            "Could not connect to IBM watsonx.ai."
        ) from exc

    # Token expired or became invalid.
    if response.status_code == 401:
        _invalidate_token()

        raise WatsonxAuthError(
            "IBM watsonx.ai authentication failed. "
            "Please check the API key."
        )

    if response.status_code != 200:
        _raise_api_error(response)

    try:
        body = response.json()

    except ValueError as exc:
        raise WatsonxAPIError(
            "IBM watsonx.ai returned an invalid JSON response."
        ) from exc

    generated = _extract_text(body)

    if not generated or not generated.strip():
        raise WatsonxEmptyResponse(
            "IBM Granite returned an empty response."
        )

    return generated.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Response extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_text(body: dict) -> str:
    """
    Extract generated text from IBM response.
    """

    results = body.get("results", [])

    if isinstance(results, list) and results:

        first_result = results[0]

        if isinstance(first_result, dict):
            generated = first_result.get(
                "generated_text",
                "",
            )

            if isinstance(generated, str):
                return generated

    # Fallback for older response formats.
    generated = body.get("generated_text", "")

    if isinstance(generated, str):
        return generated

    return ""


# ─────────────────────────────────────────────────────────────────────────────
# API error handling
# ─────────────────────────────────────────────────────────────────────────────

def _raise_api_error(response: requests.Response) -> None:
    """
    Convert IBM API errors into WatsonxAPIError.
    """

    try:
        body = response.json()

        errors = body.get("errors", [])

        if isinstance(errors, list) and errors:
            first_error = errors[0]

            if isinstance(first_error, dict):
                detail = first_error.get(
                    "message",
                    str(body),
                )
            else:
                detail = str(first_error)

        else:
            detail = body.get(
                "error",
                body.get(
                    "message",
                    str(body),
                ),
            )

    except Exception:
        detail = response.text or f"HTTP {response.status_code}"

    raise WatsonxAPIError(
        f"IBM watsonx.ai API error "
        f"(HTTP {response.status_code}): {detail}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Custom exceptions
# ─────────────────────────────────────────────────────────────────────────────

class WatsonxError(Exception):
    """Base class for watsonx errors."""


class WatsonxConfigError(WatsonxError):
    """Watsonx configuration is missing or invalid."""


class WatsonxAuthError(WatsonxError):
    """IBM authentication failed."""


class WatsonxAPIError(WatsonxError):
    """IBM API returned an error."""


class WatsonxNetworkError(WatsonxError):
    """Network-level failure."""


class WatsonxEmptyResponse(WatsonxError):
    """Granite returned no generated text."""