"""Shared Google Gemini configuration for every LLM task in the project.

All summarisation, extraction and RAG chains obtain their chat model from
:func:`get_llm` and invoke chains through :func:`invoke_with_retry`, so model
selection, credentials, retries and error messages live in exactly one place.

Gemini 3 notes (verified against the installed ``langchain-google-genai`` and
the Gemini API docs):

* ``temperature``, ``top_p``, ``top_k`` and ``candidate_count`` are deprecated
  for Gemini 3.6+ models, so this module never sets them.
* ``thinking_budget`` is replaced by ``thinking_level``. ``gemini-3.8-flash``
  supports ``low``, ``medium`` (default) and ``high`` — ``minimal`` is rejected.
* The SDK's own retry loop ignores the server's suggested retry delay, so SDK
  retries are disabled (``max_retries=1``) and retries are handled here.
"""

from __future__ import annotations

import logging
import os
import random
import re
import threading
import time
from functools import lru_cache
from typing import Any, Callable

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from core.config import load_environment

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.8-flash"
SUPPORTED_THINKING_LEVELS = ("low", "medium", "high")
DEFAULT_TIMEOUT_SECONDS = 180.0
DEFAULT_MAX_OUTPUT_TOKENS = 16_384  # well below the model's 65,536 limit
# Free-tier projects can have very small daily request quotas (e.g. 20/day), and failed
# attempts count toward them, so retries are deliberately few. Override: GEMINI_MAX_ATTEMPTS.
DEFAULT_MAX_ATTEMPTS = 3
MAX_RETRY_DELAY_SECONDS = 60.0
BASE_RETRY_DELAY_SECONDS = 2.0

# Finish reasons that mean the model refused / was blocked; retrying won't help.
_BLOCKED_FINISH_REASONS = {
    "SAFETY",
    "RECITATION",
    "BLOCKLIST",
    "PROHIBITED_CONTENT",
    "SPII",
    "IMAGE_SAFETY",
}


# ── Exceptions ──────────────────────────────────────────────────────────────────


class LLMError(RuntimeError):
    """Base class for user-facing Gemini errors (messages never contain secrets)."""


class LLMConfigurationError(LLMError):
    """Missing/invalid configuration. Permanent — never retried."""


class LLMServiceError(LLMError):
    """The Gemini API call failed (after retries where appropriate)."""


class LLMEmptyResponseError(LLMError):
    """Gemini returned no usable text. Retried a bounded number of times."""


class LLMBlockedResponseError(LLMError):
    """Gemini blocked the response (e.g. safety). Permanent — never retried."""


# ── Configuration ───────────────────────────────────────────────────────────────


def get_model_name() -> str:
    """Model identifier from ``GEMINI_MODEL`` (default ``gemini-3.8-flash``)."""
    load_environment()
    return (os.getenv("GEMINI_MODEL") or DEFAULT_MODEL).strip() or DEFAULT_MODEL


def _get_api_key() -> str:
    load_environment()
    key = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
    if not key:
        raise LLMConfigurationError(
            "Gemini API key not found. Add GOOGLE_API_KEY=<your key> to the .env file "
            "in the project folder (see .env.example), then restart the app."
        )
    return key


def is_llm_configured() -> bool:
    """True if a Gemini API key is present (the value itself is never exposed)."""
    try:
        _get_api_key()
        return True
    except LLMConfigurationError:
        return False


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


def resolve_thinking_level(level: str | None = None) -> str | None:
    """Validate a thinking level; falls back to ``GEMINI_THINKING_LEVEL`` / model default."""
    load_environment()
    value = (level or os.getenv("GEMINI_THINKING_LEVEL") or "").strip().lower()
    if not value:
        return None  # use the model's default (medium for gemini-3.8-flash)
    if value not in SUPPORTED_THINKING_LEVELS:
        raise LLMConfigurationError(
            f"Unsupported Gemini thinking level '{value}'. "
            f"Use one of: {', '.join(SUPPORTED_THINKING_LEVELS)}."
        )
    return value


@lru_cache(maxsize=16)
def _build_llm(
    model: str,
    thinking_level: str | None,
    timeout: float,
    max_output_tokens: int,
    api_key: str,
):
    # Imported lazily so that importing core modules never requires the SDK
    # to initialise (and so tests can patch get_llm without network access).
    from langchain_google_genai import ChatGoogleGenerativeAI

    kwargs: dict[str, Any] = {
        "model": model,
        "google_api_key": api_key,
        "max_retries": 1,  # SDK retries disabled; see invoke_with_retry
        "timeout": timeout,
        "max_output_tokens": max_output_tokens,
    }
    if thinking_level:
        kwargs["thinking_level"] = thinking_level
    return ChatGoogleGenerativeAI(**kwargs)


def get_llm(thinking_level: str | None = None):
    """Return the shared, configured Gemini chat model.

    Instances are cached per configuration, so Streamlit reruns don't recreate them.
    """
    api_key = _get_api_key()
    try:
        return _build_llm(
            get_model_name(),
            resolve_thinking_level(thinking_level),
            _float_env("GEMINI_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS),
            _int_env("GEMINI_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS),
            api_key,
        )
    except LLMError:
        raise
    except Exception as exc:  # pydantic validation errors etc.
        raise LLMConfigurationError(
            f"Could not initialise Gemini model '{get_model_name()}': {_redact(str(exc))}"
        ) from exc


# ── Response handling (LCEL building blocks) ────────────────────────────────────


def _check_response(message: Any) -> Any:
    """Raise on blocked/empty Gemini responses before they reach StrOutputParser."""
    if not isinstance(message, AIMessage):
        return message
    metadata = message.response_metadata or {}
    finish_reason = str(metadata.get("finish_reason") or "").upper()
    finish_reason = finish_reason.rsplit(".", 1)[-1]  # "FinishReason.SAFETY" → "SAFETY"
    text = message.text.strip() if isinstance(message.text, str) else ""
    if finish_reason in _BLOCKED_FINISH_REASONS:
        raise LLMBlockedResponseError(
            f"Gemini blocked the response (finish reason: {finish_reason})."
        )
    if not text:
        if finish_reason == "MAX_TOKENS":
            raise LLMEmptyResponseError(
                "Gemini hit the output-token limit before producing an answer. "
                "Increase GEMINI_MAX_OUTPUT_TOKENS or lower GEMINI_THINKING_LEVEL."
            )
        raise LLMEmptyResponseError("Gemini returned an empty response.")
    if finish_reason == "MAX_TOKENS":
        logger.warning("Gemini response was truncated at the output-token limit.")
    return message


def _require_text(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        raise LLMEmptyResponseError("Gemini returned an empty response.")
    return cleaned


def build_text_chain(
    prompt: ChatPromptTemplate, *, thinking_level: str | None = None
) -> Runnable:
    """``prompt | Gemini | response check | StrOutputParser | non-empty check``.

    Always returns a plain ``str`` even when Gemini returns content blocks
    (``StrOutputParser`` uses ``message.text``, which drops thought blocks).
    """
    return (
        prompt
        | get_llm(thinking_level)
        | RunnableLambda(_check_response)
        | StrOutputParser()
        | RunnableLambda(_require_text)
    )


# ── Retry / error classification ────────────────────────────────────────────────

_PERMANENT, _RETRYABLE, _RATE_LIMITED, _QUOTA_EXHAUSTED = (
    "permanent",
    "retryable",
    "rate_limited",
    "quota_exhausted",
)

_RETRY_DELAY_PATTERNS = (
    re.compile(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", re.I),
    re.compile(r"retry_delay\s*\{\s*seconds:\s*(\d+)", re.I),
    re.compile(r"retry in (\d+(?:\.\d+)?)\s*s", re.I),
)

_request_lock = threading.Lock()
_last_request_at = 0.0


def _redact(message: str) -> str:
    """Remove any API key value that might appear in an error message."""
    for var in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        secret = (os.getenv(var) or "").strip()
        if secret and len(secret) > 6:
            message = message.replace(secret, "***")
    return message


def _status_code(exc: BaseException) -> int | None:
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    # google-genai formats errors as e.g. "429 RESOURCE_EXHAUSTED. {...}"
    match = re.search(r"\b([45]\d\d)\s+[A-Z][A-Z_]{3,}", str(exc))
    if match:
        return int(match.group(1))
    return None


def classify_error(exc: BaseException) -> str:
    """Classify an exception as permanent, retryable, rate-limited or quota-exhausted."""
    from langchain_core import exceptions as lc_exc

    if isinstance(exc, LLMEmptyResponseError):
        return _RETRYABLE
    if isinstance(exc, LLMError):
        return _PERMANENT

    message = str(exc)
    status = _status_code(exc)
    is_rate_limit = isinstance(exc, getattr(lc_exc, "ModelRateLimitError", ())) or (
        status == 429 or "RESOURCE_EXHAUSTED" in message
    )
    if is_rate_limit:
        # Daily quotas (e.g. free-tier requests-per-day) won't recover by retrying.
        if re.search(r"per\s*day|PerDay|daily", message, re.I):
            return _QUOTA_EXHAUSTED
        return _RATE_LIMITED

    permanent_types = tuple(
        t
        for t in (
            getattr(lc_exc, "ModelAuthenticationError", None),
            getattr(lc_exc, "ModelPermissionDeniedError", None),
            getattr(lc_exc, "ModelNotFoundError", None),
            getattr(lc_exc, "ModelInvalidRequestError", None),
            getattr(lc_exc, "ContextOverflowError", None),
        )
        if t is not None
    )
    if permanent_types and isinstance(exc, permanent_types):
        return _PERMANENT

    if getattr(exc, "is_retryable", False):
        return _RETRYABLE
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return _RETRYABLE
    module = type(exc).__module__ or ""
    if module.startswith(("httpx", "httpcore", "requests", "urllib3")):
        return _RETRYABLE
    if status is not None:
        if status in (408, 500, 502, 503, 504):
            return _RETRYABLE
        if 400 <= status < 500:
            return _PERMANENT
    return _PERMANENT


def _server_retry_delay(exc: BaseException) -> float | None:
    text = str(exc)
    for pattern in _RETRY_DELAY_PATTERNS:
        match = pattern.search(text)
        if match:
            return float(match.group(1))
    return None


def _human_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{max(minutes, 1)} min"


def _backoff_delay(attempt: int, exc: BaseException) -> float:
    # Capacity errors (503 "high demand") need more patience than ordinary blips.
    base = BASE_RETRY_DELAY_SECONDS * (2 if _status_code(exc) == 503 else 1)
    exponential = base * (2 ** (attempt - 1)) + random.uniform(0, 1)
    hinted = _server_retry_delay(exc)
    delay = max(exponential, hinted or 0.0)
    return min(delay, MAX_RETRY_DELAY_SECONDS)


def friendly_error_message(exc: BaseException, task: str) -> str:
    """Translate an exception into an actionable, secret-free message."""
    from langchain_core import exceptions as lc_exc

    model = get_model_name()
    kind = classify_error(exc)
    raw = _redact(str(exc)).strip()
    short = raw if len(raw) <= 300 else raw[:300] + "…"

    if isinstance(exc, LLMError):
        return f"{task} failed: {raw}"
    if kind == _QUOTA_EXHAUSTED:
        limit = re.search(r"limit:\s*(\d+)", raw)
        limit_text = f" ({limit.group(1)} requests/day)" if limit else ""
        wait = _server_retry_delay(exc)
        reset_text = f" Quota resets in about {_human_duration(wait)}." if wait else ""
        return (
            f"{task} failed: the Gemini daily request quota for '{model}'{limit_text} is "
            f"exhausted (HTTP 429).{reset_text} Free-tier daily limits reset at midnight Pacific "
            "time; wait for the reset or enable billing on your Google AI Studio project."
        )
    if kind == _RATE_LIMITED:
        return (
            f"{task} failed: Gemini rate limit reached for '{model}' (HTTP 429) and retries "
            "were exhausted. Wait a minute and try again. Check your limits at "
            "https://aistudio.google.com/rate-limit."
        )
    if isinstance(exc, getattr(lc_exc, "ModelAuthenticationError", ())) or _status_code(exc) == 401:
        return f"{task} failed: Gemini rejected the API key (HTTP 401). Check GOOGLE_API_KEY in .env."
    if isinstance(exc, getattr(lc_exc, "ModelPermissionDeniedError", ())) or _status_code(exc) == 403:
        return (
            f"{task} failed: the API key does not have permission to use '{model}' (HTTP 403). "
            "Check the key's project and API restrictions."
        )
    if isinstance(exc, getattr(lc_exc, "ModelNotFoundError", ())) or _status_code(exc) == 404:
        return (
            f"{task} failed: model '{model}' is not available to your API project (HTTP 404). "
            "Check GEMINI_MODEL in .env."
        )
    if isinstance(exc, getattr(lc_exc, "ContextOverflowError", ())):
        return f"{task} failed: the input is larger than the model's context window."
    if _status_code(exc) == 503 or "UNAVAILABLE" in raw:
        return (
            f"{task} failed: Gemini model '{model}' is temporarily overloaded (HTTP 503 — high "
            "demand) and retries were exhausted. This is on Google's side; please try again in "
            "a few minutes."
        )
    if kind == _RETRYABLE:
        return (
            f"{task} failed: temporary Gemini/network error persisted after retries "
            f"({type(exc).__name__}: {short})."
        )
    return f"{task} failed: {type(exc).__name__}: {short}"


def _throttle() -> None:
    """Optional minimum spacing between Gemini requests (``GEMINI_MIN_REQUEST_INTERVAL``)."""
    global _last_request_at
    interval = _float_env("GEMINI_MIN_REQUEST_INTERVAL", 0.0)
    if interval <= 0:
        return
    with _request_lock:
        wait = _last_request_at + interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()


def invoke_with_retry(
    runnable: Runnable,
    payload: Any,
    *,
    task: str = "Gemini request",
    max_attempts: int | None = None,
    sleep: Callable[[float], None] | None = None,
) -> Any:
    """Invoke an LCEL runnable with bounded exponential backoff.

    * Retries transient failures (429 rate limits, 5xx, timeouts, connection
      errors, empty responses), honouring the server's ``retryDelay`` hint.
    * Fails fast on permanent errors (bad key, missing model, invalid request,
      exhausted daily quota, blocked content).
    * Always raises :class:`LLMError` subclasses with secret-free messages.
    """
    attempts = max(1, max_attempts or _int_env("GEMINI_MAX_ATTEMPTS", DEFAULT_MAX_ATTEMPTS))
    for attempt in range(1, attempts + 1):
        try:
            _throttle()
            return runnable.invoke(payload)
        except Exception as exc:
            kind = classify_error(exc)
            retryable = kind in (_RETRYABLE, _RATE_LIMITED)
            if not retryable or attempt >= attempts:
                if isinstance(exc, LLMConfigurationError):
                    raise
                if isinstance(exc, LLMError) and not retryable:
                    raise
                raise LLMServiceError(friendly_error_message(exc, task)) from exc
            delay = _backoff_delay(attempt, exc)
            logger.warning(
                "%s: attempt %d/%d failed (%s). Retrying in %.1fs.",
                task,
                attempt,
                attempts,
                type(exc).__name__,
                delay,
            )
            (sleep or time.sleep)(delay)
    raise LLMServiceError(f"{task} failed.")  # pragma: no cover - loop always returns/raises
