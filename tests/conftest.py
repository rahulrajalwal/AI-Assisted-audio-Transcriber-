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


class FakeGemini(BaseChatModel):
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
    """Point every persistent directory at a temporary location."""
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "vector_db"))
    monkeypatch.setenv("MEETINGS_DIR", str(tmp_path / "meetings"))
    monkeypatch.setenv("WORK_DIR", str(tmp_path / "work"))
    monkeypatch.setenv("GEMINI_MIN_REQUEST_INTERVAL", "0")
    monkeypatch.setattr(core.config, "_env_loaded", True)
    yield tmp_path


@pytest.fixture
def fake_api_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-1234567890")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")


@pytest.fixture
def fake_llm(monkeypatch, fake_api_key) -> Callable[..., FakeGemini]:
    """Install a FakeGemini as the shared model. Usage: llm = fake_llm("reply")."""

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
