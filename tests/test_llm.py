"""Gemini configuration, response handling and retry behaviour (offline)."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai.chat_models import (
    GoogleAPIError,
    GoogleAuthenticationError,
    GoogleModelNotFoundError,
    GoogleRateLimitError,
)

import core.llm as llm_module
from core.llm import (
    LLMBlockedResponseError,
    LLMConfigurationError,
    LLMEmptyResponseError,
    LLMServiceError,
    build_text_chain,
    classify_error,
    get_llm,
    invoke_with_retry,
)
from langchain_core.runnables import RunnableLambda


def _server_error(code: int = 503) -> GoogleAPIError:
    return GoogleAPIError(
        code, {"error": {"code": code, "message": "high demand", "status": "UNAVAILABLE"}}, None
    )


class TestInitialisation:
    def test_builds_gemini_without_sampling_params(self, fake_api_key):
        model = get_llm()
        assert type(model).__name__ == "ChatGoogleGenerativeAI"
        assert model.model == "gemini-3.8-flash"
        assert model.temperature is None  # Gemini 3: no custom sampling
        assert model.top_p is None and model.top_k is None
        assert model.n == 1
        assert model.max_retries == 1  # retries handled by invoke_with_retry
        assert model.thinking_budget is None

    def test_model_is_cached_between_calls(self, fake_api_key):
        assert get_llm() is get_llm()

    def test_thinking_level_is_passed(self, fake_api_key):
        assert get_llm("low").reasoning_effort == "low"

    def test_minimal_thinking_level_rejected(self, fake_api_key):
        with pytest.raises(LLMConfigurationError, match="minimal"):
            get_llm("minimal")

    def test_model_name_from_env(self, fake_api_key, monkeypatch):
        monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
        assert get_llm().model == "gemini-3.6-flash"

    def test_missing_api_key_gives_actionable_error(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(LLMConfigurationError, match="GOOGLE_API_KEY"):
            get_llm()
        assert llm_module.is_llm_configured() is False

    def test_gemini_api_key_fallback(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "fallback-key-123456")
        assert llm_module.is_llm_configured() is True
        assert get_llm().google_api_key.get_secret_value() == "fallback-key-123456"


class TestResponseHandling:
    def _chain(self, fake_llm, reply):
        fake_llm(reply)
        return build_text_chain(ChatPromptTemplate.from_messages([("human", "{q}")]))

    def test_plain_string_output(self, fake_llm):
        assert self._chain(fake_llm, "  Hello  ").invoke({"q": "x"}) == "Hello"

    def test_content_blocks_drop_thoughts(self, fake_llm):
        reply = AIMessage(
            content=[
                {"type": "thinking", "thinking": "internal reasoning"},
                {"type": "text", "text": "Final answer"},
            ]
        )
        assert self._chain(fake_llm, reply).invoke({"q": "x"}) == "Final answer"

    def test_blocked_response_raises(self, fake_llm):
        reply = AIMessage(content="", response_metadata={"finish_reason": "SAFETY"})
        with pytest.raises(LLMBlockedResponseError):
            self._chain(fake_llm, reply).invoke({"q": "x"})

    def test_empty_response_raises(self, fake_llm):
        with pytest.raises(LLMEmptyResponseError):
            self._chain(fake_llm, AIMessage(content="")).invoke({"q": "x"})

    def test_max_tokens_without_text_explains(self, fake_llm):
        reply = AIMessage(content="", response_metadata={"finish_reason": "MAX_TOKENS"})
        with pytest.raises(LLMEmptyResponseError, match="output-token limit"):
            self._chain(fake_llm, reply).invoke({"q": "x"})


class TestRetry:
    def _flaky(self, errors):
        state = {"calls": 0}

        def run(_):
            state["calls"] += 1
            if errors:
                raise errors.pop(0)
            return "done"

        return RunnableLambda(run), state

    def test_retries_transient_503_then_succeeds(self, no_sleep):
        runnable, state = self._flaky([_server_error(), _server_error()])
        assert invoke_with_retry(runnable, None, task="t") == "done"
        assert state["calls"] == 3
        assert len(no_sleep) == 2 and no_sleep[1] > no_sleep[0]  # exponential backoff

    def test_honours_server_retry_delay(self, no_sleep):
        err = GoogleRateLimitError("429 RESOURCE_EXHAUSTED. {'retryDelay': '17s'}")
        runnable, _ = self._flaky([err])
        invoke_with_retry(runnable, None, task="t")
        assert no_sleep[0] >= 17

    def test_retry_delay_is_capped(self, no_sleep):
        err = GoogleRateLimitError("429 RESOURCE_EXHAUSTED. {'retryDelay': '900s'}")
        runnable, _ = self._flaky([err])
        invoke_with_retry(runnable, None, task="t")
        assert no_sleep[0] <= llm_module.MAX_RETRY_DELAY_SECONDS

    def test_auth_error_not_retried(self, no_sleep, fake_api_key):
        runnable, state = self._flaky([GoogleAuthenticationError("401 UNAUTHENTICATED bad key")])
        with pytest.raises(LLMServiceError, match="401"):
            invoke_with_retry(runnable, None, task="Summary")
        assert state["calls"] == 1 and no_sleep == []

    def test_model_not_found_not_retried(self, no_sleep, fake_api_key):
        runnable, state = self._flaky([GoogleModelNotFoundError("404 NOT_FOUND models/x")])
        with pytest.raises(LLMServiceError, match="not available"):
            invoke_with_retry(runnable, None, task="Summary")
        assert state["calls"] == 1

    def test_daily_quota_fails_fast(self, no_sleep, fake_api_key):
        err = GoogleRateLimitError(
            "429 RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier"
        )
        runnable, state = self._flaky([err])
        with pytest.raises(LLMServiceError, match="daily request quota"):
            invoke_with_retry(runnable, None, task="Summary")
        assert state["calls"] == 1

    def test_real_free_tier_daily_quota_message(self, no_sleep, fake_api_key):
        """Format copied from a real Gemini 429 response (free tier, 20 requests/day)."""
        err = GoogleRateLimitError(
            "Error calling model 'gemini-3.8-flash' (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. "
            "{'error': {'code': 429, 'message': '* Quota exceeded for metric: "
            "generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, "
            "model: gemini-3.8-flash, description: Generate content requests per day limit exceeded "
            "on Free Tier.', 'details': [{'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}, "
            "{'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '49656s'}]}}"
        )
        runnable, state = self._flaky([err])
        with pytest.raises(LLMServiceError) as info:
            invoke_with_retry(runnable, None, task="Meeting summary")
        message = str(info.value)
        assert "20 requests/day" in message and "13h 47m" in message
        assert state["calls"] == 1 and no_sleep == []

    def test_bounded_attempts(self, no_sleep, fake_api_key):
        runnable, state = self._flaky([_server_error() for _ in range(10)])
        with pytest.raises(LLMServiceError, match="overloaded"):
            invoke_with_retry(runnable, None, task="Summary", max_attempts=3)
        assert state["calls"] == 3

    def test_empty_response_is_retried(self, no_sleep):
        runnable, state = self._flaky([LLMEmptyResponseError("empty")])
        assert invoke_with_retry(runnable, None) == "done"
        assert state["calls"] == 2

    def test_blocked_response_not_retried(self, no_sleep):
        runnable, state = self._flaky([LLMBlockedResponseError("blocked")])
        with pytest.raises(LLMBlockedResponseError):
            invoke_with_retry(runnable, None)
        assert state["calls"] == 1

    def test_unknown_bug_not_retried(self, no_sleep, fake_api_key):
        runnable, state = self._flaky([KeyError("missing")])
        with pytest.raises(LLMServiceError):
            invoke_with_retry(runnable, None)
        assert state["calls"] == 1

    def test_timeouts_are_retryable(self):
        assert classify_error(TimeoutError()) == "retryable"
        assert classify_error(ConnectionError()) == "retryable"

    def test_api_key_is_redacted_from_errors(self, no_sleep, monkeypatch):
        secret = "AIzaSySECRETSECRETSECRET"
        monkeypatch.setenv("GOOGLE_API_KEY", secret)
        runnable, _ = self._flaky([RuntimeError(f"bad request with key={secret}")])
        with pytest.raises(LLMServiceError) as info:
            invoke_with_retry(runnable, None)
        assert secret not in str(info.value)
