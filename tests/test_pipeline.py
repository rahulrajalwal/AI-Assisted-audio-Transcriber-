"""Pipeline orchestration, progress events, caching and failure handling (offline)."""

from __future__ import annotations

import pytest

import core.pipeline as pipeline
from core.llm import LLMServiceError
from core.meeting_store import list_meetings, load_meeting
from core.pipeline import PipelineError, analyse_transcript, run_pipeline, transcribe_source
from core.vector_store import compute_meeting_id, load_vector_store
from tests.conftest import SAMPLE_TRANSCRIPT, requires_ffmpeg


def scripted_reply(messages):
    system = str(messages[0].content)
    if "final summary" in system:
        return "### Overview\nWeekly product sync about the app launch."
    if "title" in system:
        return "Weekly Product Sync"
    if "ACTION ITEM" in system:
        return "1. **Prepare the press kit**\n   - **Owner:** Priya\n   - **Deadline:** Friday"
    if "KEY DECISION" in system:
        return "1. **Launch on 15 August**"
    if "OPEN QUESTION" in system:
        return "1. **Who approves the extra spending?**"
    return "unexpected"


def test_analyse_transcript_full_flow(fake_llm, fake_embeddings):
    llm = fake_llm(scripted_reply)
    events = []
    record = analyse_transcript(SAMPLE_TRANSCRIPT, source="meeting.mp4", language="english", progress=events.append)

    assert record.title == "Weekly Product Sync"
    assert record.summary.startswith("### Overview")
    assert "Priya" in record.action_items
    assert record.key_decisions == "1. **Launch on 15 August**"
    assert record.model == "openai/gpt-4o-mini"
    assert len(llm.calls) == 5  # summary, title, 3 extractions

    finished = [(e.stage, e.status) for e in events if e.status != "running"]
    assert finished == [("analysis", "done"), ("extraction", "done"), ("indexing", "done")]
    assert load_meeting(record.meeting_id).title == "Weekly Product Sync"
    assert load_vector_store().get(where={"meeting_id": record.meeting_id}, include=[])["ids"]


def test_reprocessing_reuses_saved_analysis(fake_llm, fake_embeddings):
    llm = fake_llm(scripted_reply)
    first = analyse_transcript(SAMPLE_TRANSCRIPT)
    calls = len(llm.calls)
    chunks = len(load_vector_store().get(include=[])["ids"])

    second = analyse_transcript(SAMPLE_TRANSCRIPT)
    assert second.meeting_id == first.meeting_id
    assert len(llm.calls) == calls  # no duplicate Gemini calls
    assert len(load_vector_store().get(include=[])["ids"]) == chunks  # no duplicate chunks

    analyse_transcript(SAMPLE_TRANSCRIPT, force=True)
    assert len(llm.calls) == calls + 5


def test_failed_extraction_is_not_reported_as_success(fake_llm, fake_embeddings):
    def reply(messages):
        if "KEY DECISION" in str(messages[0].content):
            raise LLMServiceError("Key-decision extraction failed: quota")
        return scripted_reply(messages)

    fake_llm(reply)
    events = []
    with pytest.raises(PipelineError) as info:
        analyse_transcript(SAMPLE_TRANSCRIPT, progress=events.append)

    assert info.value.stage == "extraction"
    assert info.value.transcript == SAMPLE_TRANSCRIPT  # allows retry without re-transcribing
    statuses = {(e.stage, e.status) for e in events}
    assert ("extraction", "error") in statuses
    assert ("extraction", "done") not in statuses
    assert ("indexing", "running") not in statuses
    assert list_meetings() == []  # nothing saved
    assert load_meeting(compute_meeting_id(SAMPLE_TRANSCRIPT)) is None


def test_openrouter_unreachable_stops_before_analysis(llm_down, fake_embeddings):
    with pytest.raises(PipelineError, match="Could not reach OpenRouter") as info:
        analyse_transcript(SAMPLE_TRANSCRIPT, segments=[{"start": 0, "end": 1, "char_start": 0, "char_end": 5}])
    assert info.value.stage == "analysis"
    assert info.value.transcript == SAMPLE_TRANSCRIPT and info.value.segments  # retry keeps timestamps


def test_force_reanalysis_refreshes_only_this_meetings_chunks(fake_llm, fake_embeddings):
    fake_llm(scripted_reply)
    analyse_transcript(SAMPLE_TRANSCRIPT)
    analyse_transcript("A different meeting about the office move. We agreed to move in June.")
    before = len(load_vector_store().get(include=[])["ids"])
    analyse_transcript(SAMPLE_TRANSCRIPT, force=True)
    assert len(load_vector_store().get(include=[])["ids"]) == before


def test_missing_file_fails_in_input_stage(tmp_path):
    events = []
    with pytest.raises(PipelineError) as info:
        transcribe_source(str(tmp_path / "nope.mp4"), "english", events.append)
    assert info.value.stage == "input"
    assert events[-1].status == "error"


def test_invalid_language_fails_in_input_stage(tmp_path):
    with pytest.raises(PipelineError, match="Unsupported language") as info:
        transcribe_source("https://youtu.be/abc", "french")
    assert info.value.stage == "input"


@requires_ffmpeg
def test_run_pipeline_with_mocked_transcription(fake_llm, fake_embeddings, monkeypatch, tmp_path):
    from pydub.generators import Sine

    audio = tmp_path / "meeting.mp3"
    Sine(440).to_audio_segment(duration=2_000).export(audio, format="mp3")
    segments = [{"start": 0.0, "end": 2.0, "audio_chunk": 0, "char_start": 0, "char_end": len(SAMPLE_TRANSCRIPT)}]
    monkeypatch.setattr(
        pipeline, "transcribe_all_with_segments", lambda chunks, language, progress=None: (SAMPLE_TRANSCRIPT, segments)
    )
    fake_llm(scripted_reply)

    events = []
    record = run_pipeline(
        str(audio), "english", progress=events.append, source_label="meeting.mp3", meeting_date="2026-03-10"
    )
    done = [e.stage for e in events if e.status == "done"]
    assert done == ["input", "audio", "transcription", "analysis", "extraction", "indexing"]
    assert record.source == "meeting.mp3" and record.source_type == "local_file"
    assert record.meeting_date == "2026-03-10" and record.segments == segments
    meta = load_vector_store().get(where={"meeting_id": record.meeting_id}, include=["metadatas"])["metadatas"]
    assert meta and meta[0]["meeting_date"] == "2026-03-10" and meta[0]["start_seconds"] == 0.0
    assert record.duration_seconds and 1.5 < record.duration_seconds < 2.5
    # temporary audio is cleaned up
    from core.config import work_dir

    assert not any(work_dir().glob("run_*"))


def test_chat_history_saved_per_meeting():
    from core.meeting_store import load_chats, save_chats

    meeting_id = "a" * 16
    assert load_chats(meeting_id) == {}
    save_chats(meeting_id, {"This meeting": [{"role": "user", "content": "hi"}]})
    assert load_chats(meeting_id)["This meeting"][0]["content"] == "hi"
    from core.meeting_store import list_meetings

    assert list_meetings() == []  # chat files are never mistaken for meetings
