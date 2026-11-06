"""Shared LLM configuration for every LLM task in the project.

ACTIVE PROVIDER: OpenRouter — default model ``openai/gpt-4o-mini`` (set OPENROUTER_MODEL).

Default model: OpenAI GPT-4o mini via OpenRouter (paid, needs OpenRouter credits):
128K-token context, up to 16K output tokens, about $0.15 per million input tokens
and $0.60 per million output tokens — a typical meeting costs well under $0.01.
Served by OpenAI and Azure, so it is not subject to the shared free-model capacity
limits that made ``google/gemma-4-31b-it:free`` unreliable in testing.

All title, summarisation, extraction and RAG chains obtain their chat model from
:func:`get_llm` and invoke chains through :func:`invoke_with_retry`, so model
selection, credentials, retries and error messages live in exactly one place.
Only transcript text is ever sent to OpenRouter (never audio, never embeddings —
embeddings stay local with Hugging Face ``all-MiniLM-L6-v2``).

Integration: ``ChatOpenAI`` (``langchain-openai``) pointed at OpenRouter's
OpenAI-compatible endpoint. The dedicated ``langchain-openrouter`` package was not
used because its ``openrouter`` SDK requires pydantic < 2.13, which conflicts with
``kubernetes`` (a ChromaDB dependency) requiring pydantic >= 2.13.5.

Free models (``:free`` suffix, e.g. ``google/gemma-4-31b-it:free``) are still
supported if selected manually: OpenRouter allows 20 requests/minute and 50/day
(1000/day after buying >= 10 credits), so for them requests are spaced at least
``MIN_REQUEST_INTERVAL_SECONDS`` apart. For every model, requests are sequential,
retried at most once, and the app never switches to another model by itself.

Hosted providers used earlier (Google Gemini 3.8 Flash, Mistral AI) are preserved,
commented out, in the "INACTIVE HOSTED-MODEL INTEGRATIONS" section at the bottom.
"""

from __future__ import annotations

import logging
import os
import random
import re
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Iterator

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from core.config import load_environment

logger = logging.getLogger(__name__)

# ── Active provider defaults (OpenRouter) ───────────────────────────────────────

ACTIVE_PROVIDERS = ("openrouter",)
LEGACY_PROVIDERS = ("gemini", "mistral")  # preserved below, inactive

DEFAULT_PROVIDER = "openrouter"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"
# Optional free alternatives (select manually via OPENROUTER_MODEL; never used automatically).
# In testing they were often "temporarily rate-limited upstream" by Google AI Studio.
FREE_ALTERNATIVE_MODELS = ("google/gemma-4-31b-it:free", "google/gemma-4-26b-a4b-it:free")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT_SECONDS = 180.0
DEFAULT_MAX_OUTPUT_TOKENS = 4096
DEFAULT_MAX_ATTEMPTS = 2  # one retry at most: failed attempts may count toward the daily quota
MAX_RETRY_DELAY_SECONDS = 60.0  # a longer wait (e.g. daily quota) fails with a clear message instead
BASE_RETRY_DELAY_SECONDS = 4.0
MIN_REQUEST_INTERVAL_SECONDS = 3.1  # 20 requests/minute limit of ":free" models

# Input sizing. GPT-4o mini accepts 128K tokens; inputs are kept large enough that a
# one-hour meeting fits in one summary request (few requests, low cost) yet bounded
# for reliable output quality.
MAX_INPUT_TOKENS = 24_000
CHARS_PER_TOKEN = 3.5
PROMPT_OVERHEAD_TOKENS = 500  # system prompt + format instructions

# Finish reasons that mean the provider blocked the response; retrying won't help.
_BLOCKED_FINISH_REASONS = {"CONTENT_FILTER", "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT"}


# ── Exceptions ──────────────────────────────────────────────────────────────────


class LLMError(RuntimeError):
    """Base class for user-facing LLM errors (messages never contain secrets)."""


class LLMConfigurationError(LLMError):
    """Missing API key, invalid key, unavailable model, exhausted daily quota. Never retried."""


class LLMServiceError(LLMError):
    """The LLM call failed (after retries where appropriate)."""


class LLMEmptyResponseError(LLMError):
    """The model returned no usable text. Retried a bounded number of times."""


class LLMBlockedResponseError(LLMError):
    """The provider blocked the response. Never retried."""


class LLMContextOverflowError(LLMError):
    """The input is larger than the model's context window. Never retried."""


# ── Configuration ───────────────────────────────────────────────────────────────


def get_provider() -> str:
    load_environment()
    return (os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()


def get_model_name() -> str:
    """Identifier of the active model (shown in the UI and stored with each meeting)."""
    load_environment()
    return (os.getenv("OPENROUTER_MODEL") or DEFAULT_OPENROUTER_MODEL).strip()


def get_provider_label() -> str:
    return {"openrouter": "OpenRouter"}.get(get_provider(), f"{get_provider()} (inactive)")


def _api_key() -> str:
    load_environment()
    key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not key:
        raise LLMConfigurationError(
            "OpenRouter API key not found. Add OPENROUTER_API_KEY=<your key> to the .env file "
            "in the project folder (see .env.example), then restart the app."
        )
    return key


def _base_url() -> str:
    return (os.getenv("OPENROUTER_BASE_URL") or OPENROUTER_BASE_URL).strip().rstrip("/")


def input_char_budget(reserve_tokens: int = 0) -> int:
    """Characters of source text that fit in one request alongside prompt and output."""
    tokens = MAX_INPUT_TOKENS - PROMPT_OVERHEAD_TOKENS - reserve_tokens
    return max(1_000, int(tokens * CHARS_PER_TOKEN))


def estimate_tokens(text: str) -> int:
    return int(len(text or "") / CHARS_PER_TOKEN) + 1


# ── Connection / quota status (no model request is spent) ───────────────────────


@dataclass
class OpenRouterStatus:
    key_configured: bool
    model: str
    reachable: bool = False
    key_valid: bool = False
    model_available: bool = False
    daily_used: int | None = None
    daily_limit: int | None = None
    daily_remaining: int | None = None
    error: str = ""

    @property
    def is_free_model(self) -> bool:
        return self.model.endswith(":free")

    @property
    def ready(self) -> bool:
        # The 50/day free-request limit applies only to ":free" models.
        exhausted = self.is_free_model and self.daily_remaining is not None and self.daily_remaining <= 0
        return self.key_valid and self.model_available and not exhausted


def check_openrouter(timeout: float = 10.0) -> OpenRouterStatus:
    """Verify the key, the model's availability and today's free-request usage.

    Uses OpenRouter's ``/key`` and ``/models/.../endpoints`` endpoints, which do not
    count as model requests.
    """
    import httpx

    model = get_model_name()
    try:
        key = _api_key()
    except LLMConfigurationError as exc:
        return OpenRouterStatus(key_configured=False, model=model, error=str(exc))

    status = OpenRouterStatus(key_configured=True, model=model)
    base = _base_url()
    try:
        response = httpx.get(f"{base}/key", headers={"Authorization": f"Bearer {key}"}, timeout=timeout)
        status.reachable = True
        if response.status_code in (401, 403):
            status.error = "OpenRouter rejected the API key (HTTP %d). Check OPENROUTER_API_KEY in .env." % response.status_code
            return status
        response.raise_for_status()
        status.key_valid = True
        daily = (response.json().get("data") or {}).get("free_model_daily_requests") or {}
        status.daily_used, status.daily_limit, status.daily_remaining = (
            daily.get("used"), daily.get("limit"), daily.get("remaining")
        )

        endpoints = httpx.get(f"{base}/models/{model}/endpoints", timeout=timeout)
        if endpoints.status_code == 404:
            status.error = (
                f"Model '{model}' is not available on OpenRouter. Check OPENROUTER_MODEL in .env "
                "(no other model is used automatically)."
            )
            return status
        endpoints.raise_for_status()
        status.model_available = bool((endpoints.json().get("data") or {}).get("endpoints"))
        if not status.model_available:
            status.error = f"Model '{model}' currently has no available provider on OpenRouter. Try again later."
        elif status.is_free_model and status.daily_remaining is not None and status.daily_remaining <= 0:
            status.error = (
                f"The free OpenRouter quota for today is used up ({status.daily_used}/{status.daily_limit} "
                "requests). It resets at 00:00 UTC (05:30 IST)."
            )
    except Exception as exc:
        status.error = f"Could not reach OpenRouter ({type(exc).__name__}). Check your internet connection."
    return status


def ensure_llm_ready() -> None:
    """Fail fast with an actionable message before starting LLM work."""
    provider = get_provider()
    if provider in LEGACY_PROVIDERS:
        raise LLMConfigurationError(
            f"LLM_PROVIDER={provider} refers to a preserved but INACTIVE integration. Remove LLM_PROVIDER "
            "from .env (OpenRouter is the default) or re-enable it in core/llm.py."
        )
    if provider not in ACTIVE_PROVIDERS:
        raise LLMConfigurationError(f"Unknown LLM_PROVIDER '{provider}'. Use: openrouter.")
    status = check_openrouter()
    if not status.ready:
        raise LLMConfigurationError(status.error or "OpenRouter is not ready.")


def is_llm_configured() -> bool:
    try:
        ensure_llm_ready()
        return True
    except LLMConfigurationError:
        return False


# ── Model construction ──────────────────────────────────────────────────────────


@lru_cache(maxsize=4)
def _build_openrouter_llm(model: str, base_url: str, api_key: str):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        max_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
        timeout=DEFAULT_TIMEOUT_SECONDS,
        max_retries=0,  # retries are handled (and limited) by invoke_with_retry
        default_headers={"X-Title": "AI Meeting Assistant"},
    )


def get_llm(thinking_level: str | None = None):
    """Return the shared chat model for the ACTIVE provider (cached; no network call).

    ``thinking_level`` is accepted for compatibility with existing callers and the
    preserved Gemini integration; it is not used by OpenRouter.
    """
    provider = get_provider()
    if provider == "openrouter":
        return _build_openrouter_llm(get_model_name(), _base_url(), _api_key())
    # ── Inactive hosted providers (see bottom of file to re-enable) ──
    # if provider == "gemini":
    #     return _build_gemini_llm(thinking_level)
    # if provider == "mistral":
    #     return _build_mistral_llm()
    if provider in LEGACY_PROVIDERS:
        raise LLMConfigurationError(
            f"The {provider} integration is preserved but inactive. Remove LLM_PROVIDER from .env "
            "or re-enable it in core/llm.py."
        )
    raise LLMConfigurationError(f"Unknown LLM_PROVIDER '{provider}'. Use: openrouter.")


# ── Response handling (LCEL building blocks) ────────────────────────────────────


def _check_response(message: Any) -> Any:
    """Raise on blocked or empty responses before StrOutputParser."""
    if not isinstance(message, AIMessage):
        return message
    metadata = message.response_metadata or {}
    finish_reason = str(metadata.get("finish_reason") or "").upper().rsplit(".", 1)[-1]
    text = message.text.strip() if isinstance(message.text, str) else ""
    if finish_reason in _BLOCKED_FINISH_REASONS:
        raise LLMBlockedResponseError(f"The model blocked the response (finish reason: {finish_reason}).")
    if not text:
        if finish_reason in ("LENGTH", "MAX_TOKENS"):
            raise LLMEmptyResponseError("The model hit its output-token limit before producing an answer.")
        raise LLMEmptyResponseError("The model returned an empty response.")
    if finish_reason in ("LENGTH", "MAX_TOKENS"):
        logger.warning("Model response was cut off at the output-token limit.")
    return message


def _require_text(text: str) -> str:
    cleaned = (text or "").strip()
    # Defensive: strip a leaked <think>…</think> block from reasoning models.
    cleaned = re.sub(r"^<think>.*?</think>\s*", "", cleaned, flags=re.S).strip()
    if not cleaned:
        raise LLMEmptyResponseError("The model returned an empty response.")
    return cleaned


def build_stream_chain(prompt: ChatPromptTemplate) -> Runnable:
    """``prompt | chat model | StrOutputParser`` for token streaming (LCEL)."""
    return prompt | get_llm() | StrOutputParser()


def build_text_chain(prompt: ChatPromptTemplate, *, thinking_level: str | None = None) -> Runnable:
    """``prompt | chat model | response check | StrOutputParser | non-empty check`` (LCEL)."""
    return (
        prompt
        | get_llm(thinking_level)
        | RunnableLambda(_check_response)
        | StrOutputParser()
        | RunnableLambda(_require_text)
    )


# ── Retry / error classification ────────────────────────────────────────────────

_PERMANENT, _RETRYABLE, _RATE_LIMITED, _QUOTA_EXHAUSTED = "permanent", "retryable", "rate_limited", "quota_exhausted"

_inference_lock = threading.RLock()  # one LLM request at a time in this process
_request_lock = threading.Lock()
_last_request_at = 0.0


def _redact(message: str) -> str:
    """Remove any API key value that might appear in an error message."""
    for var in ("OPENROUTER_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "MISTRAL_API_KEY", "SARVAM_API_KEY"):
        secret = (os.getenv(var) or "").strip()
        if secret and len(secret) > 6:
            message = message.replace(secret, "***")
    return message


def _status_code(exc: BaseException) -> int | None:
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and 100 <= value < 600:
            return value
    match = re.search(r"Error code: (\d{3})", str(exc))
    return int(match.group(1)) if match else None


def _headers(exc: BaseException) -> dict:
    response = getattr(exc, "response", None)
    try:
        return {k.lower(): v for k, v in dict(response.headers).items()} if response is not None else {}
    except Exception:
        return {}


def _server_retry_delay(exc: BaseException) -> float | None:
    """Seconds to wait, from Retry-After or X-RateLimit-Reset (ms since epoch)."""
    headers = _headers(exc)
    retry_after = headers.get("retry-after", "")
    try:
        if retry_after:
            return max(0.0, float(retry_after))
    except ValueError:
        pass
    reset = headers.get("x-ratelimit-reset", "")
    if reset.isdigit():
        return max(0.0, int(reset) / 1000 - time.time())
    return None


def _provider_detail(exc: BaseException) -> str:
    """OpenRouter puts the upstream provider's explanation in error.metadata.raw."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", body)
        metadata = (error or {}).get("metadata") or {}
        raw = metadata.get("raw")
        if raw:
            return _redact(str(raw))[:300]
    return ""


def _is_daily_quota(exc: BaseException) -> bool:
    text = (str(exc) + " " + _provider_detail(exc)).lower()
    if "upstream" in text:
        return False  # the provider's capacity, not the account's daily quota
    if re.search(r"per[-_ ]?day|daily", text):
        return True
    delay = _server_retry_delay(exc)
    return delay is not None and delay > MAX_RETRY_DELAY_SECONDS


def classify_error(exc: BaseException) -> str:
    """Classify an exception as permanent, retryable, rate-limited or quota-exhausted."""
    if isinstance(exc, LLMEmptyResponseError):
        return _RETRYABLE
    if isinstance(exc, LLMError):
        return _PERMANENT
    status = _status_code(exc)
    name = type(exc).__name__
    if status == 429 or name == "RateLimitError":
        return _QUOTA_EXHAUSTED if _is_daily_quota(exc) else _RATE_LIMITED
    if status in (400, 401, 402, 403, 404, 413, 422) or name in (
        "AuthenticationError", "PermissionDeniedError", "NotFoundError", "BadRequestError", "UnprocessableEntityError",
    ):
        return _PERMANENT
    if status in (408, 500, 502, 503, 504) or name in ("APIConnectionError", "APITimeoutError", "InternalServerError"):
        return _RETRYABLE
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return _RETRYABLE
    if (type(exc).__module__ or "").startswith(("httpx", "httpcore")):
        return _RETRYABLE
    return _PERMANENT


def _backoff_delay(attempt: int, exc: BaseException) -> float:
    exponential = BASE_RETRY_DELAY_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 1)
    return min(max(exponential, _server_retry_delay(exc) or 0.0), MAX_RETRY_DELAY_SECONDS)


def friendly_error_message(exc: BaseException, task: str) -> str:
    """Translate an exception into an actionable, secret-free message."""
    model = get_model_name()
    kind = classify_error(exc)
    raw = _redact(str(exc)).strip()
    short = raw if len(raw) <= 300 else raw[:300] + "…"
    status = _status_code(exc)
    name = type(exc).__name__

    if isinstance(exc, LLMError):
        return f"{task} failed: {raw}"
    if kind == _QUOTA_EXHAUSTED:
        return (
            f"{task} failed: the free OpenRouter request limit for today has been reached (HTTP 429). "
            "The daily limit resets at 00:00 UTC (05:30 IST). Your transcript and meeting data are kept; "
            "use Retry analysis after the reset."
        )
    if kind == _RATE_LIMITED and "upstream" in _provider_detail(exc).lower():
        return (
            f"{task} failed: the free '{model}' is temporarily rate-limited by its provider (Google AI Studio) "
            "— this is shared free capacity, not your daily quota. Try again in a few minutes; your "
            "transcript and meeting data are kept."
        )
    if kind == _RATE_LIMITED:
        return (
            f"{task} failed: '{model}' is rate-limited right now (HTTP 429). Wait a minute and retry."
        )
    if status == 401 or name == "AuthenticationError":
        return f"{task} failed: OpenRouter rejected the API key (HTTP 401). Check OPENROUTER_API_KEY in .env."
    if status == 402:
        return (
            f"{task} failed: OpenRouter reports insufficient credits for '{model}' (HTTP 402). "
            "Add credits at https://openrouter.ai/settings/credits; your transcript and meeting data are kept."
        )
    if status == 403 or name == "PermissionDeniedError":
        return f"{task} failed: the request was refused by OpenRouter (HTTP 403). {short}"
    if status == 404 or name == "NotFoundError":
        return (
            f"{task} failed: model '{model}' is not available on OpenRouter (HTTP 404). "
            "Check OPENROUTER_MODEL in .env; no other model is used automatically."
        )
    if status in (400, 413) or "context length" in raw.lower() or "too long" in raw.lower():
        return f"{task} failed: the request was rejected as invalid or too large (HTTP {status}). {short}"
    if name in ("APIConnectionError", "APITimeoutError") or isinstance(exc, (TimeoutError, ConnectionError)):
        return f"{task} failed: could not reach OpenRouter (network error or timeout). Check your connection and retry."
    if kind == _RETRYABLE:
        return f"{task} failed: OpenRouter or the model provider is temporarily unavailable (HTTP {status}). Try again shortly."
    return f"{task} failed: {name}: {short}"


def _throttle() -> None:
    """Keep ":free" model requests at least MIN_REQUEST_INTERVAL_SECONDS apart (20/minute)."""
    global _last_request_at
    if MIN_REQUEST_INTERVAL_SECONDS <= 0 or not get_model_name().endswith(":free"):
        return
    with _request_lock:
        wait = _last_request_at + MIN_REQUEST_INTERVAL_SECONDS - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()


def invoke_with_retry(
    runnable: Runnable,
    payload: Any,
    *,
    task: str = "LLM request",
    max_attempts: int | None = None,
    sleep: Callable[[float], None] | None = None,
) -> Any:
    """Invoke an LCEL runnable — one request at a time — with at most one retry.

    Retries only transient failures (per-minute 429s, 5xx, timeouts, empty responses),
    honouring Retry-After / X-RateLimit-Reset up to MAX_RETRY_DELAY_SECONDS. Fails fast on
    the daily quota, bad keys and unavailable models. Never switches to another model.
    Always raises :class:`LLMError` subclasses with secret-free, actionable messages.
    """
    attempts = max(1, max_attempts or DEFAULT_MAX_ATTEMPTS)
    for attempt in range(1, attempts + 1):
        try:
            _throttle()
            with _inference_lock:
                return runnable.invoke(payload)
        except Exception as exc:
            kind = classify_error(exc)
            retryable = kind in (_RETRYABLE, _RATE_LIMITED)
            if not retryable or attempt >= attempts:
                if isinstance(exc, LLMError) and not retryable:
                    raise
                raise LLMServiceError(friendly_error_message(exc, task)) from exc
            delay = _backoff_delay(attempt, exc)
            logger.warning(
                "%s: attempt %d/%d failed (%s). Retrying in %.1fs.", task, attempt, attempts, type(exc).__name__, delay
            )
            (sleep or time.sleep)(delay)
    raise LLMServiceError(f"{task} failed.")  # pragma: no cover


def stream_with_retry(
    runnable: Runnable,
    payload: Any,
    *,
    task: str = "LLM request",
    max_attempts: int | None = None,
    sleep: Callable[[float], None] | None = None,
) -> Iterator[str]:
    """Stream text chunks from an LCEL runnable (for word-by-word chat answers).

    Same rules as :func:`invoke_with_retry`, except that a retry is only possible
    before the first chunk arrives — once text is shown, a failure is reported.
    """
    attempts = max(1, max_attempts or DEFAULT_MAX_ATTEMPTS)
    for attempt in range(1, attempts + 1):
        produced = False
        try:
            _throttle()
            with _inference_lock:
                for chunk in runnable.stream(payload):
                    text = chunk if isinstance(chunk, str) else getattr(chunk, "content", "") or ""
                    if text:
                        produced = True
                        yield text
            if not produced:
                raise LLMEmptyResponseError("The model returned an empty response.")
            return
        except Exception as exc:
            kind = classify_error(exc)
            retryable = kind in (_RETRYABLE, _RATE_LIMITED) and not produced
            if not retryable or attempt >= attempts:
                if isinstance(exc, LLMError) and not retryable:
                    raise
                raise LLMServiceError(friendly_error_message(exc, task)) from exc
            delay = _backoff_delay(attempt, exc)
            logger.warning("%s: stream attempt %d/%d failed (%s). Retrying in %.1fs.",
                           task, attempt, attempts, type(exc).__name__, delay)
            (sleep or time.sleep)(delay)


# ════════════════════════════════════════════════════════════════════════════════
# INACTIVE HOSTED-MODEL INTEGRATIONS (preserved for future use — NOT executed)
# ════════════════════════════════════════════════════════════════════════════════
#
# The application previously used other hosted LLM APIs. Their implementations are
# kept here, commented out, so they are never imported or initialised by the
# running app, and they are NEVER used as automatic fallbacks. To re-enable one:
#   1. Install its package:  pip install langchain-google-genai   (or langchain-mistralai)
#   2. Uncomment the matching function below AND its branch in get_llm().
#   3. Add it to ACTIVE_PROVIDERS, set LLM_PROVIDER=gemini (or mistral) in .env,
#      and put its key in .env (GOOGLE_API_KEY / MISTRAL_API_KEY).
#   4. Update ensure_llm_ready()/check_openrouter() to check that provider instead.
#
# ── Google Gemini 3.8 Flash ──
#
# GEMINI_DEFAULT_MODEL = "gemini-3.8-flash"
# GEMINI_THINKING_LEVELS = ("low", "medium", "high")   # "minimal" is rejected by 3.8 Flash
#
# def _gemini_api_key() -> str:
#     key = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
#     if not key:
#         raise LLMConfigurationError("Gemini API key not found. Add GOOGLE_API_KEY to .env.")
#     return key
#
# @lru_cache(maxsize=16)
# def _cached_gemini_llm(model: str, thinking_level: str | None, timeout: float,
#                        max_output_tokens: int, api_key: str):
#     from langchain_google_genai import ChatGoogleGenerativeAI
#     kwargs = {
#         "model": model,
#         "google_api_key": api_key,
#         "max_retries": 1,            # SDK retries disabled; invoke_with_retry handles them
#         "timeout": timeout,
#         "max_output_tokens": max_output_tokens,
#         # No temperature/top_p/top_k: deprecated for Gemini 3.6+ models.
#     }
#     if thinking_level:
#         kwargs["thinking_level"] = thinking_level
#     return ChatGoogleGenerativeAI(**kwargs)
#
# def _build_gemini_llm(thinking_level: str | None = None):
#     level = (thinking_level or os.getenv("GEMINI_THINKING_LEVEL") or "").strip().lower() or None
#     if level and level not in GEMINI_THINKING_LEVELS:
#         raise LLMConfigurationError(f"Unsupported Gemini thinking level '{level}'.")
#     return _cached_gemini_llm(
#         (os.getenv("GEMINI_MODEL") or GEMINI_DEFAULT_MODEL).strip(),
#         level,
#         180.0,
#         16_384,
#         _gemini_api_key(),
#     )
#
# ── Mistral AI (original implementation; was duplicated in summarizer.py,
#    extractor.py and rag_engine.py as get_llm()) ──
#
# def _build_mistral_llm():
#     from langchain_mistralai import ChatMistralAI
#     # summarizer.py / rag_engine.py used temperature=0.3, extractor.py used 0.2
#     return ChatMistralAI(
#         model="mistral-small-latest",
#         mistral_api_key=os.getenv("MISTRAL_API_KEY"),
#         temperature=0.3,
#     )
