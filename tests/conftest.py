"""Shared fixtures. All tests run offline: Gemini, Sarvam and YouTube are mocked,
and every test gets its own temporary ChromaDB / meeting-history / work directories
so the user's real data is never touched."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any, Callable

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core.config  # noqa: E402  (loads .env once; tests then override env vars)


class FakeGemini(BaseChatModel):  # name kept for history; stands in for any chat model
    """Chat model stand-in that records every prompt and returns scripted replies."""

    responder: Any = None  # str | AIMessage | Exception | callable(messages) -> any of these
    calls: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-gemini"

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls.append(messages)
        reply = self.responder(messages) if callable(self.responder) else self.responder
        if isinstance(reply, Exception):
            raise reply
        message = reply if isinstance(reply, AIMessage) else AIMessage(content=str(reply))
        return ChatResult(generations=[ChatGeneration(message=message)])

    # helpers for assertions
    def system_prompts(self) -> list[str]:
        return [str(msgs[0].content) for msgs in self.calls]

    def last_human(self) -> str:
        return str(self.calls[-1][-1].content)


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    """Temporary storage folders and NO access to the real OpenRouter API."""
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "vector_db"))
    monkeypatch.setenv("MEETINGS_DIR", str(tmp_path / "meetings"))
    monkeypatch.setenv("WORK_DIR", str(tmp_path / "work"))
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key-1234567890")
    # A closed local port: unit tests must never send requests (or spend quota) on OpenRouter.
    monkeypatch.setenv("OPENROUTER_BASE_URL", "http://127.0.0.1:9/api/v1")
    monkeypatch.setattr(core.config, "_env_loaded", True)
    import core.llm as llm_module

    monkeypatch.setattr(llm_module, "MIN_REQUEST_INTERVAL_SECONDS", 0)
    yield tmp_path


def _status(ready: bool = True, **overrides):
    from core.llm import OpenRouterStatus

    data = dict(
        key_configured=True,
        model="openai/gpt-4o-mini",
        reachable=ready,
        key_valid=ready,
        model_available=ready,
        daily_used=3 if ready else None,
        daily_limit=50 if ready else None,
        daily_remaining=47 if ready else None,
        error="" if ready else "Could not reach OpenRouter (ConnectError). Check your internet connection.",
    )
    data.update(overrides)
    return OpenRouterStatus(**data)


@pytest.fixture
def llm_ready(monkeypatch):
    """Pretend OpenRouter is reachable, the key valid and the model available."""
    import core.llm

    monkeypatch.setattr(core.llm, "check_openrouter", lambda timeout=10.0: _status(True))


@pytest.fixture
def llm_down(monkeypatch):
    """Pretend OpenRouter cannot be reached."""
    import core.llm

    monkeypatch.setattr(core.llm, "check_openrouter", lambda timeout=10.0: _status(False))


@pytest.fixture
def fake_llm(monkeypatch, llm_ready) -> Callable[..., FakeGemini]:
    """Install a fake chat model as the shared LLM. Usage: llm = fake_llm("reply")."""

    def install(responder: Any = "ok") -> FakeGemini:
        model = FakeGemini(responder=responder)
        import core.llm

        monkeypatch.setattr(core.llm, "get_llm", lambda thinking_level=None: model)
        monkeypatch.setattr(core.llm.time, "sleep", lambda seconds: None)
        return model

    return install


@pytest.fixture
def fake_embeddings(monkeypatch):
    """Deterministic offline embeddings instead of downloading MiniLM."""
    import core.vector_store as vs

    embeddings = DeterministicFakeEmbedding(size=64)
    monkeypatch.setattr(vs, "get_embeddings", lambda: embeddings)
    vs._open_store.cache_clear()
    yield embeddings
    vs._open_store.cache_clear()


@pytest.fixture
def no_sleep(monkeypatch):
    import core.llm

    delays: list[float] = []
    monkeypatch.setattr(core.llm.time, "sleep", lambda seconds: delays.append(seconds))
    return delays


requires_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg not installed")


SAMPLE_TRANSCRIPT = (
    "Good morning everyone. This is the weekly product sync. First topic, the mobile app launch. "
    "After some discussion, we decided to launch the mobile app on the 15th of August. "
    "Priya will prepare the press kit by Friday. Rahul will fix the login bug before the launch. "
    "Second topic, the marketing budget. We have not decided whether to increase the marketing budget. "
    "Anil asked who will approve the extra spending, and that question is still open. "
    "We will discuss it again next week. Thanks everyone."
)


FIXTURES = Path(__file__).resolve().parent / "fixtures"
