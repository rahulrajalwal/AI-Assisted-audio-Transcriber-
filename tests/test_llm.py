"""OpenRouter configuration, status checks, response handling and retries (offline).

One live test at the bottom makes exactly ONE real OpenRouter request; it runs only
when LIVE_OPENROUTER=1 is set (it uses one request of the 50/day free quota).
"""

from __future__ import annotations

import os
import time

import httpx
import openai
import pytest
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

import core.llm as llm_module
from core.llm import (
    LLMBlockedResponseError,
    LLMConfigurationError,
    LLMEmptyResponseError,
    LLMServiceError,
    build_text_chain,
    check_openrouter,
    classify_error,
    ensure_llm_ready,
    get_llm,
    input_char_budget,
    invoke_with_retry,
    stream_with_retry,
)

REQUEST = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")


def api_error(cls, status: int, message: str, headers: dict | None = None):
    return cls(message, response=httpx.Response(status, headers=headers or {}, request=REQUEST), body=None)


class TestConfiguration:
    def test_builds_chat_openai_against_openrouter(self):
        model = get_llm()
        assert type(model).__name__ == "ChatOpenAI"
        assert model.model_name == "openai/gpt-4o-mini"
        assert model.openai_api_base == "http://127.0.0.1:9/api/v1"  # test override of OPENROUTER_BASE_URL
        assert model.temperature == 0
        assert model.max_retries == 0  # retries limited by invoke_with_retry (free-tier quota)

    def test_default_endpoint_and_model(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
        monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
        model = get_llm()
        assert model.openai_api_base == "https://openrouter.ai/api/v1"
        assert model.model_name == "openai/gpt-4o-mini"  # default: GPT-4o mini

    def test_model_configurable(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free")
        assert get_llm().model_name == "google/gemma-4-26b-a4b-it:free"

    def test_edited_env_file_applies_without_restart(self, monkeypatch, tmp_path):
        """Regression: the running app kept the old model after .env was edited."""
        import core.config

        env_file = tmp_path / ".env"
        env_file.write_text("OPENROUTER_MODEL=google/gemma-4-31b-it:free\n", encoding="utf-8")
        monkeypatch.setattr(core.config, "ENV_FILE", env_file)
        assert core.config.reload_environment()
        assert get_llm().model_name == "google/gemma-4-31b-it:free"
        env_file.write_text("OPENROUTER_MODEL=openai/gpt-4o-mini\n", encoding="utf-8")
        core.config.reload_environment()
        assert get_llm().model_name == "openai/gpt-4o-mini"

    def test_pacing_only_for_free_models(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr(llm_module, "MIN_REQUEST_INTERVAL_SECONDS", 3.1)
        monkeypatch.setattr(llm_module.time, "sleep", lambda s: sleeps.append(s))
        llm_module._throttle(); llm_module._throttle()
        assert sleeps == []  # paid default model: no 20/minute pacing
        monkeypatch.setenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")
        llm_module._throttle(); llm_module._throttle()
        assert sleeps and sleeps[0] > 0

    def test_model_cached_and_no_network_on_init(self):
        assert get_llm() is get_llm()

    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(LLMConfigurationError, match="OPENROUTER_API_KEY"):
            get_llm()
        status = check_openrouter()
        assert not status.key_configured and not status.ready and "OPENROUTER_API_KEY" in status.error

    @pytest.mark.parametrize("provider", ["gemini", "mistral"])
    def test_legacy_providers_are_inactive(self, monkeypatch, provider):
        monkeypatch.setenv("LLM_PROVIDER", provider)
        with pytest.raises(LLMConfigurationError, match="inactive"):
            get_llm()
        with pytest.raises(LLMConfigurationError, match="INACTIVE"):
            ensure_llm_ready()

    def test_no_ollama_gemini_or_mistral_imported(self):
        import sys

        get_llm()
        for module in ("langchain_ollama", "ollama", "langchain_mistralai", "langchain_google_genai"):
            assert module not in sys.modules, module

    def test_input_budget_keeps_requests_few_but_bounded(self):
        budget = input_char_budget()
        assert 60_000 <= budget <= 100_000  # a one-hour meeting fits in one summary request


class TestStatusCheck:
    def test_unreachable(self):
        status = check_openrouter(timeout=0.5)
        assert status.key_configured and not status.ready
        assert "Could not reach OpenRouter" in status.error

    def _fake_http(self, monkeypatch, key_status=200, remaining=40, model_status=200, endpoints=1):
        def fake_get(url, headers=None, timeout=None):
            if url.endswith("/key"):
                data = {"data": {"free_model_daily_requests": {"used": 50 - remaining, "limit": 50, "remaining": remaining}}}
                return httpx.Response(key_status, json=data, request=httpx.Request("GET", url))
            data = {"data": {"endpoints": [{}] * endpoints}}
            return httpx.Response(model_status, json=data, request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx, "get", fake_get)

    def test_ready_with_quota(self, monkeypatch):
        self._fake_http(monkeypatch)
        status = check_openrouter()
        assert status.ready and (status.daily_used, status.daily_limit, status.daily_remaining) == (10, 50, 40)

    def test_invalid_key(self, monkeypatch):
        self._fake_http(monkeypatch, key_status=401)
        status = check_openrouter()
        assert not status.ready and "rejected the API key" in status.error

    def test_model_unavailable_no_fallback(self, monkeypatch):
        self._fake_http(monkeypatch, model_status=404)
        status = check_openrouter()
        assert not status.ready and "not available" in status.error and "no other model" in status.error

    def test_free_quota_does_not_block_paid_model(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_MODEL", "google/gemma-4-31b-it")
        self._fake_http(monkeypatch, remaining=0)
        assert check_openrouter().ready

    def test_daily_quota_exhausted(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")
        self._fake_http(monkeypatch, remaining=0)
        status = check_openrouter()
        assert not status.ready and "used up" in status.error
        with pytest.raises(LLMConfigurationError, match="used up"):
            ensure_llm_ready()


class TestResponseHandling:
    def _chain(self, fake_llm, reply):
        fake_llm(reply)
        return build_text_chain(ChatPromptTemplate.from_messages([("human", "{q}")]))

    def test_plain_string_output(self, fake_llm):
        assert self._chain(fake_llm, "  Hello  ").invoke({"q": "x"}) == "Hello"

    def test_leaked_think_block_is_removed(self, fake_llm):
        assert self._chain(fake_llm, "<think>reasoning</think>\nFinal").invoke({"q": "x"}) == "Final"

    def test_output_limit_without_text(self, fake_llm):
        reply = AIMessage(content="", response_metadata={"finish_reason": "length"})
        with pytest.raises(LLMEmptyResponseError, match="output-token limit"):
            self._chain(fake_llm, reply).invoke({"q": "x"})

    def test_blocked_response(self, fake_llm):
        reply = AIMessage(content="", response_metadata={"finish_reason": "content_filter"})
        with pytest.raises(LLMBlockedResponseError):
            self._chain(fake_llm, reply).invoke({"q": "x"})


class TestRetries:
    def _flaky(self, errors):
        state = {"calls": 0}

        def run(_):
            state["calls"] += 1
            if errors:
                raise errors.pop(0)
            return "done"

        return RunnableLambda(run), state

    def test_per_minute_rate_limit_retried_once_honouring_retry_after(self, no_sleep):
        err = api_error(openai.RateLimitError, 429, "Rate limit exceeded: free-models-per-min", {"Retry-After": "7"})
        runnable, state = self._flaky([err])
        assert invoke_with_retry(runnable, None) == "done"
        assert state["calls"] == 2 and no_sleep[0] >= 7

    def test_upstream_provider_limit_explained(self, no_sleep):
        """Real OpenRouter body seen live: the free model's upstream provider was saturated."""
        body = {"error": {"code": 429, "message": "Provider returned error", "metadata": {
            "provider_name": "Google AI Studio",
            "raw": "google/gemma-4-31b-it:free is temporarily rate-limited upstream. Please retry shortly"}}}
        errors = [openai.RateLimitError("Error code: 429 - Provider returned error",
                                        response=httpx.Response(429, request=REQUEST), body=body) for _ in range(3)]
        runnable, state = self._flaky(errors)
        with pytest.raises(LLMServiceError, match="rate-limited by its provider") as info:
            invoke_with_retry(runnable, None)
        assert "not your daily quota" in str(info.value)
        assert state["calls"] == 2  # retried once, then a clear message (no model fallback)

    def test_daily_quota_fails_fast_without_retry(self, no_sleep):
        err = api_error(openai.RateLimitError, 429, "Rate limit exceeded: free-models-per-day")
        runnable, state = self._flaky([err])
        with pytest.raises(LLMServiceError, match="limit for today") as info:
            invoke_with_retry(runnable, None, task="Meeting summary")
        assert "transcript and meeting data are kept" in str(info.value)
        assert state["calls"] == 1 and no_sleep == []

    def test_long_reset_header_treated_as_daily_quota(self, no_sleep):
        reset_ms = str(int((time.time() + 6 * 3600) * 1000))
        err = api_error(openai.RateLimitError, 429, "Rate limit exceeded", {"X-RateLimit-Reset": reset_ms})
        runnable, state = self._flaky([err])
        with pytest.raises(LLMServiceError, match="limit for today"):
            invoke_with_retry(runnable, None)
        assert state["calls"] == 1

    def test_retries_are_bounded(self, no_sleep):
        errors = [api_error(openai.InternalServerError, 503, "upstream busy") for _ in range(5)]
        runnable, state = self._flaky(errors)
        with pytest.raises(LLMServiceError, match="temporarily unavailable"):
            invoke_with_retry(runnable, None)
        assert state["calls"] == 2  # at most one retry by default

    def test_bad_key_not_retried(self, no_sleep):
        runnable, state = self._flaky([api_error(openai.AuthenticationError, 401, "No auth credentials found")])
        with pytest.raises(LLMServiceError, match="rejected the API key"):
            invoke_with_retry(runnable, None)
        assert state["calls"] == 1

    def test_unknown_model_not_retried_and_no_fallback(self, no_sleep):
        runnable, state = self._flaky([api_error(openai.NotFoundError, 404, "No endpoints found")])
        with pytest.raises(LLMServiceError, match="not available on OpenRouter") as info:
            invoke_with_retry(runnable, None)
        assert "no other model is used" in str(info.value) and state["calls"] == 1

    def test_network_error_retried(self, no_sleep):
        runnable, state = self._flaky([openai.APIConnectionError(request=REQUEST)])
        assert invoke_with_retry(runnable, None) == "done"
        assert state["calls"] == 2

    def test_empty_response_retried(self, no_sleep):
        runnable, state = self._flaky([LLMEmptyResponseError("empty")])
        assert invoke_with_retry(runnable, None) == "done"

    def test_unknown_bug_not_retried(self, no_sleep):
        runnable, state = self._flaky([KeyError("missing")])
        with pytest.raises(LLMServiceError):
            invoke_with_retry(runnable, None)
        assert state["calls"] == 1

    def test_classification(self):
        assert classify_error(TimeoutError()) == "retryable"
        assert classify_error(api_error(openai.BadRequestError, 400, "bad")) == "permanent"

    def test_api_key_redacted(self, no_sleep, monkeypatch):
        secret = "sk-or-v1-SECRETSECRETSECRET"
        monkeypatch.setenv("OPENROUTER_API_KEY", secret)
        runnable, _ = self._flaky([RuntimeError(f"bad request key={secret}")])
        with pytest.raises(LLMServiceError) as info:
            invoke_with_retry(runnable, None)
        assert secret not in str(info.value)


class TestStreamRetry:
    class Flaky:
        def __init__(self, plan):
            self.plan, self.calls = plan, 0

        def stream(self, payload):
            step = self.plan[self.calls]
            self.calls += 1
            for item in step:
                if isinstance(item, Exception):
                    raise item
                yield item

    def test_retries_before_first_chunk(self, no_sleep):
        runnable = self.Flaky([[openai.APIConnectionError(request=REQUEST)], ["Hel", "lo"]])
        assert "".join(stream_with_retry(runnable, None)) == "Hello" and runnable.calls == 2

    def test_no_retry_after_text_was_shown(self, no_sleep):
        runnable = self.Flaky([["Partial ", openai.APIConnectionError(request=REQUEST)], ["never"]])
        produced = []
        with pytest.raises(LLMServiceError):
            for chunk in stream_with_retry(runnable, None):
                produced.append(chunk)
        assert produced == ["Partial "] and runnable.calls == 1

    def test_empty_stream_is_an_error(self, no_sleep):
        with pytest.raises(LLMServiceError):
            list(stream_with_retry(self.Flaky([[], []]), None))


@pytest.mark.openrouter
@pytest.mark.skipif(not os.getenv("LIVE_OPENROUTER"), reason="set LIVE_OPENROUTER=1 (sends 1 real request)")
def test_live_openrouter_generation(monkeypatch):
    """Exactly one real request through the project's LCEL chain."""
    from dotenv import dotenv_values

    env = dotenv_values(os.path.join(os.path.dirname(__file__), "..", ".env"))
    monkeypatch.setenv("OPENROUTER_API_KEY", env.get("OPENROUTER_API_KEY") or "")
    monkeypatch.setenv("OPENROUTER_BASE_URL", env.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENROUTER_MODEL", env.get("OPENROUTER_MODEL") or "openai/gpt-4o-mini")
    monkeypatch.setattr(llm_module, "DEFAULT_MAX_ATTEMPTS", 1)
    chain = build_text_chain(ChatPromptTemplate.from_messages([("human", "Reply with exactly the word: READY")]))
    assert "READY" in invoke_with_retry(chain, {}).upper()
