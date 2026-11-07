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
def app(llm_ready, monkeypatch) -> AppTest:
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


def test_openrouter_unreachable_blocks_processing(app, llm_down):
    at = app.run()
    at.text_input(key="youtube_url").set_value("https://youtu.be/abc")
    button(at, "Analyse meeting").click().run()
    assert any("Could not reach OpenRouter" in e.value for e in at.error)
    assert at.session_state["job"] is None


def test_sidebar_shows_openrouter_provider(app, llm_ready):
    at = app.run()
    sidebar_html = " ".join(m.value for m in at.sidebar.markdown)
    assert "OpenRouter" in sidebar_html and "openai/gpt-4o-mini" in sidebar_html
    assert "n/a (paid model)" in sidebar_html
    assert "inactive (Gemini, Mistral preserved)" in sidebar_html
    assert "Ollama" not in sidebar_html and "Qwen" not in sidebar_html


def test_history_and_meeting_tabs(app):
    record = make_record(title="Weekly Product Sync")
    save_meeting(record)
    at = app.run()
    assert "Weekly Product Sync" in [b.label for b in at.button]
    button(at, "Open").click().run()
    assert not at.exception
    assert any("Weekly Product Sync" in m.value for m in at.markdown)
    assert [m.label for m in at.metric][:2] == ["Duration", "Words"]

    for tab in ("Ask Your Meeting", "Transcript", "Summary", "Action Items", "Key Decisions",
                "Open Questions", "Downloads"):
        at.session_state["meeting_tab"] = tab
        at.run()
        assert not at.exception, tab
    labels = [b.label for b in at.get("download_button")]
    for expected in ("Download full report", "Download everything", "Download transcript", "Download summary",
                     "Download action items", "Download key decisions", "Download open questions", "Download q&a chat"):
        assert expected in labels, expected


def _fake_stream(seen):
    def fake_stream(question, **kwargs):
        seen["question"], seen["kwargs"] = question, kwargs
        info = {"scope_note": "", "sources": [
            {"number": 1, "title": "Weekly Product Sync", "chunk_index": 0, "start_seconds": 65.0, "end_seconds": 90.0,
             "excerpt": "Priya will prepare the press kit", "meeting_id": "x", "meeting_date": "2026-11-05"}
        ]}
        return info, iter(["Priya prepares ", "the press kit ", "[Source 1]."])

    return fake_stream


def test_chat_streams_answer_and_saves_history(app, monkeypatch):
    record = make_record(title="Weekly Product Sync")
    save_meeting(record)
    seen = {}
    monkeypatch.setattr(core.rag_engine, "stream_answer", _fake_stream(seen))

    at = app.run()
    button(at, "Open").click().run()
    at.session_state["meeting_tab"] = "Ask Your Meeting"
    at.run()
    at.chat_input[0].set_value("Who prepares the press kit?").run()
    assert not at.exception
    assert seen["question"] == "Who prepares the press kit?"
    assert seen["kwargs"]["meeting_id"] == record.meeting_id  # "This meeting" scope
    history = at.session_state["chats"][record.meeting_id]["This meeting"]
    assert history[-1]["content"] == "Priya prepares the press kit [Source 1]."
    assert history[-1]["sources"][0]["start_seconds"] == 65.0

    from core.meeting_store import load_chats

    assert load_chats(record.meeting_id)["This meeting"][-1]["content"].startswith("Priya")  # survives restarts


def test_suggested_question_button_asks(app, monkeypatch):
    record = make_record(title="Weekly Product Sync")
    save_meeting(record)
    seen = {}
    monkeypatch.setattr(core.rag_engine, "stream_answer", _fake_stream(seen))
    at = app.run()
    button(at, "Open").click().run()
    button(at, "What were the main topics discussed?").click().run()
    assert not at.exception
    assert seen["question"] == "What were the main topics discussed?"
    assert at.session_state["meeting_tab"] == "Ask Your Meeting"


def test_history_shows_original_name_and_ai_title(app):
    save_meeting(make_record(meeting_id="1" * 16, title="Launch Planning Sync", source_title="Me at the zoo"))
    save_meeting(make_record(meeting_id="2" * 16, title="Budget Review", source="C:/rec/q2-budget.m4a",
                             source_type="local_file"))
    at = app.run()
    labels = [b.label for b in at.button]
    assert "Me at the zoo" in labels and "q2-budget.m4a" in labels  # original names, not AI titles
    next(b for b in at.button if b.label == "Me at the zoo").click().run()
    assert not at.exception
    heads = [m.value for m in at.markdown if m.value.startswith("<div class=\"meeting-head\"")]
    assert heads and "<h1>Me at the zoo</h1>" in heads[0] and "AI title: Launch Planning Sync" in heads[0]
    assert at.session_state["meeting_id"] == "1" * 16
    assert at.session_state["chat_scope"] == "This meeting"  # meeting-specific chat by default


def test_unfinished_job_can_be_resumed(app, monkeypatch):
    import core.pipeline
    from core.meeting_store import MeetingCheckpoint

    checkpoint = MeetingCheckpoint("3" * 16)
    checkpoint.start("file:abc", source_title="Long Board Meeting.mp4", language="english", status="failed",
                     error="Whisper failed on audio part 10 of 12", chunk_plan=[f"c{i}" for i in range(12)])
    for index in range(9):
        checkpoint.save_chunk(index, [{"start": 0.0, "end": 1.0, "text": "x"}])
    resumed = []
    monkeypatch.setattr(core.pipeline, "resume_meeting", lambda meeting_id, progress=None: resumed.append(meeting_id))
    at = app.run()
    resume = next(b for b in at.button if b.label == "Resume: Long Board Meeting.mp4")
    assert "9 of 12 audio parts transcribed" in resume.help
    resume.click().run()
    assert resumed == ["3" * 16]


def test_failed_download_is_not_reported_as_saved_audio(app):
    from core.meeting_store import MeetingCheckpoint

    MeetingCheckpoint("4" * 16).start("youtube:x", source_title="Some talk", language="hinglish", status="failed",
                                      error="Could not download the YouTube audio: HTTP Error 403")
    at = app.run()
    resume = next(b for b in at.button if b.label == "Resume: Some talk")
    assert "audio not downloaded yet" in resume.help and "audio saved" not in resume.help
