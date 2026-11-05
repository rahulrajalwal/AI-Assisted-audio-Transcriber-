"""Streamlit UI smoke tests using streamlit.testing (headless, backend mocked)."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import core.rag_engine
from core.meeting_store import save_meeting
from tests.test_exporter import make_record

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def app(fake_api_key, monkeypatch) -> AppTest:
    import core.vector_store

    monkeypatch.setattr(core.vector_store, "warm_up_embeddings", lambda: None)
    return AppTest.from_file(APP, default_timeout=60)


def button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


def test_home_renders(app):
    at = app.run()
    assert not at.exception
    labels = [b.label for b in at.button]
    assert "Analyse meeting" in labels
    assert at.radio(key="language").options == ["English — Whisper (local)", "Hinglish — Sarvam AI"]
    assert any("Turn every meeting" in m.value for m in at.markdown)


def test_invalid_youtube_url_shows_error(app):
    at = app.run()
    at.text_input(key="youtube_url").set_value("https://vimeo.com/123")
    button(at, "Analyse meeting").click().run()
    assert any("valid YouTube link" in e.value for e in at.error)
    assert at.session_state["job"] is None


def test_upload_required(app):
    at = app.run()
    at.segmented_control(key="source_mode").set_value("Upload file").run()
    button(at, "Analyse meeting").click().run()
    assert any("upload an audio or video file" in e.value for e in at.error)


def test_missing_gemini_key_blocks_processing(app, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    at = app.run()
    at.text_input(key="youtube_url").set_value("https://youtu.be/abc")
    button(at, "Analyse meeting").click().run()
    assert any("GOOGLE_API_KEY" in e.value for e in at.error)


def test_history_and_meeting_tabs(app):
    record = make_record(title="Weekly Product Sync")
    save_meeting(record)
    at = app.run()
    assert "Weekly Product Sync" in [b.label for b in at.button]
    button(at, "Open").click().run()
    assert not at.exception
    assert any("Weekly Product Sync" in m.value for m in at.markdown)
    assert [m.label for m in at.metric][:2] == ["Duration", "Words"]

    for tab in ("Transcript", "Summary", "Action Items", "Key Decisions", "Open Questions"):
        at.session_state["meeting_tab"] = tab
        at.run()
        assert not at.exception, tab


def test_chat_uses_rag_chain(app, monkeypatch):
    save_meeting(make_record(title="Weekly Product Sync"))
    seen = {}

    def fake_ask(chain, question, history=None):
        seen["question"], seen["history"] = question, list(history or [])
        return {"answer": "Priya prepares the press kit [Source 1].", "sources": [
            {"number": 1, "title": "Weekly Product Sync", "chunk_index": 0, "excerpt": "Priya will prepare the press kit", "meeting_id": "x"}
        ]}

    monkeypatch.setattr(core.rag_engine, "ask_question", fake_ask)
    monkeypatch.setattr(core.rag_engine, "load_rag_chain", lambda meeting_id=None, persist_directory=None: object())

    at = app.run()
    button(at, "Open").click().run()
    at.session_state["meeting_tab"] = "Ask Your Meeting"
    at.run()
    at.chat_input[0].set_value("Who prepares the press kit?").run()
    assert not at.exception
    assert seen["question"] == "Who prepares the press kit?"
    chats = at.session_state["chats"]
    history = next(iter(chats.values()))
    assert history[-1]["content"].startswith("Priya")
    assert history[-1]["sources"][0]["title"] == "Weekly Product Sync"
