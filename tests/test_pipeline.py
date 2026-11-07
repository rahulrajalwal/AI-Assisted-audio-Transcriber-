"""Pipeline orchestration, progress events, caching and failure handling (offline)."""

from __future__ import annotations

from pathlib import Path

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
        pipeline, "transcribe_all_with_segments", lambda chunks, language, **kwargs: (SAMPLE_TRANSCRIPT, segments)
    )
    fake_llm(scripted_reply)

    events = []
    record = run_pipeline(
        str(audio), "english", progress=events.append, source_label="meeting.mp3", meeting_date="2026-03-10"
    )
    done = [e.stage for e in events if e.status == "done"]
    assert done == ["input", "audio", "transcription", "analysis", "extraction", "indexing"]
    assert record.source == "meeting.mp3" and record.source_type == "local_file"
    assert record.source_title == "meeting.mp3" and record.display_name == "meeting.mp3"
    assert record.title == "Weekly Product Sync"  # AI title kept separately
    assert record.status == "complete" and record.vector_chunks >= 1
    assert record.meeting_date == "2026-03-10" and record.segments == segments
    meta = load_vector_store().get(where={"meeting_id": record.meeting_id}, include=["metadatas"])["metadatas"]
    assert meta and meta[0]["meeting_date"] == "2026-03-10" and meta[0]["start_seconds"] == 0.0
    assert record.duration_seconds and 1.5 < record.duration_seconds < 2.5
    # temporary audio is cleaned up
    from core.config import work_dir

    assert not any(work_dir().glob("run_*"))
    assert not any((work_dir() / "jobs").glob("*"))  # checkpoint removed after success


def test_chat_history_saved_per_meeting():
    from core.meeting_store import load_chats, save_chats

    meeting_id = "a" * 16
    assert load_chats(meeting_id) == {}
    save_chats(meeting_id, {"This meeting": [{"role": "user", "content": "hi"}]})
    assert load_chats(meeting_id)["This meeting"][0]["content"] == "hi"
    from core.meeting_store import list_meetings

    assert list_meetings() == []  # chat files are never mistaken for meetings


# ── Checkpoints: resume long recordings without repeating finished work ─────────


@pytest.fixture
def chunked_audio(tmp_path, monkeypatch):
    """A 12-second recording split into ~3-second parts, with a scripted fake speech model."""
    import core.transcriber as transcriber
    from pydub.generators import Sine

    from utils.audio_processor import chunk_audio

    audio = tmp_path / "Quarterly Review.mp3"
    Sine(440).to_audio_segment(duration=12_000).export(audio, format="mp3")
    monkeypatch.setattr(
        pipeline, "chunk_audio", lambda wav, output_dir=None: chunk_audio(wav, chunk_minutes=0.05, output_dir=output_dir)
    )
    monkeypatch.setattr(transcriber, "load_model", lambda: None)
    calls: list[int] = []
    fail_on: set[int] = set()

    def fake_segments(chunk_path, language="english"):
        index = int(Path(chunk_path).stem.split("_")[-1])
        calls.append(index)
        if index in fail_on:
            fail_on.discard(index)  # fails once, works on retry
            raise RuntimeError("simulated network error")
        return [{"start": 0.0, "end": 2.5, "text": f"Part {index} speech."}]

    monkeypatch.setattr(transcriber, "transcribe_chunk_segments", fake_segments)
    return audio, calls, fail_on


@requires_ffmpeg
def test_failed_chunk_resumes_from_that_chunk(fake_llm, fake_embeddings, chunked_audio):
    from core.meeting_store import MeetingCheckpoint, list_checkpoints

    audio, calls, fail_on = chunked_audio
    fake_llm(scripted_reply)
    fail_on.add(2)
    with pytest.raises(PipelineError) as info:
        run_pipeline(str(audio), "english")
    assert info.value.stage == "transcription" and "part 3 of" in str(info.value)
    assert calls == [0, 1, 2]
    meeting_id = info.value.meeting_id
    total = len(MeetingCheckpoint(meeting_id).data["chunk_plan"])
    assert total >= 4
    job = list_checkpoints()[0]
    assert job["meeting_id"] == meeting_id and job["status"] == "failed"
    assert (job["chunks_done"], job["chunks_total"]) == (2, total)
    assert job["source_title"] == "Quarterly Review.mp3"

    calls.clear()
    events = []
    record = pipeline.resume_meeting(meeting_id, progress=events.append)
    assert calls == list(range(2, total))  # parts 1-2 reused, never transcribed again
    assert any("Reusing" in e.message for e in events if e.stage == "audio")
    assert record.meeting_id == meeting_id
    assert record.transcript.startswith("Part 0 speech. Part 1 speech. Part 2")
    starts = [seg["start"] for seg in record.segments]
    assert starts == sorted(starts) and starts[-1] > 6  # chronological, meeting-relative timestamps
    assert record.source_title == "Quarterly Review.mp3" and record.title == "Weekly Product Sync"
    assert list_checkpoints() == []


@requires_ffmpeg
def test_llm_failure_retry_skips_transcription_and_finished_llm_steps(fake_llm, fake_embeddings, chunked_audio):
    audio, calls, _ = chunked_audio
    failures = {"KEY DECISION": 1}

    def reply(messages):
        system = str(messages[0].content)
        for marker in list(failures):
            if marker in system and failures[marker]:
                failures[marker] -= 1
                raise LLMServiceError("rate limited")
        return scripted_reply(messages)

    llm = fake_llm(reply)
    with pytest.raises(PipelineError) as info:
        run_pipeline(str(audio), "english")
    assert info.value.stage == "extraction"
    transcribed = len(calls)
    done_prompts = len(llm.calls)

    record = run_pipeline(str(audio), "english")  # same source again = continue
    assert len(calls) == transcribed  # no re-transcription
    assert len(llm.calls) - done_prompts == 2  # only key decisions (retried) + open questions
    assert record.key_decisions == "1. **Launch on 15 August**"


@requires_ffmpeg
def test_upload_keeps_original_filename_and_survives_cleanup(fake_llm, fake_embeddings, chunked_audio, tmp_path):
    import shutil as sh

    audio, _, fail_on = chunked_audio
    upload_dir = tmp_path / "upload_x"
    upload_dir.mkdir()
    staged = upload_dir / "input.mp3"
    sh.copy(audio, staged)
    fake_llm(scripted_reply)
    fail_on.add(1)
    with pytest.raises(PipelineError) as info:
        run_pipeline(str(staged), "english", source_label="Board Meeting – March.mp3", source_type="upload")
    sh.rmtree(upload_dir)  # the app deletes its upload folder after every run

    record = pipeline.resume_meeting(info.value.meeting_id)
    assert record.source_title == "Board Meeting – March.mp3" == record.display_name
    assert record.source_type == "upload" and record.title == "Weekly Product Sync"


def test_youtube_title_and_stable_id(fake_llm, fake_embeddings, monkeypatch):
    monkeypatch.setattr(pipeline, "ensure_ffmpeg", lambda: None)
    monkeypatch.setattr(
        pipeline,
        "fetch_youtube_info",
        lambda url: {
            "id": "jNQXAC9IVRw", "title": "Me at the zoo", "duration": 19,
            "url": "https://www.youtube.com/watch?v=jNQXAC9IVRw",
        },
    )
    monkeypatch.setattr(pipeline, "download_youtube_audio", lambda url, output_dir: str(Path(output_dir) / "a.m4a"))
    monkeypatch.setattr(pipeline, "convert_to_wav", lambda src, dst: dst)
    monkeypatch.setattr(pipeline, "get_duration_seconds", lambda path: 19.0)
    monkeypatch.setattr(pipeline, "chunk_audio", lambda wav, output_dir=None: [str(Path(output_dir) / "chunk_000.wav")])
    transcriptions = []

    def fake_transcribe(chunks, language, **kwargs):
        transcriptions.append(chunks)
        return SAMPLE_TRANSCRIPT, []

    monkeypatch.setattr(pipeline, "transcribe_all_with_segments", fake_transcribe)
    fake_llm(scripted_reply)

    first = run_pipeline("https://youtu.be/jNQXAC9IVRw", "english")
    assert first.source_title == "Me at the zoo" and first.title == "Weekly Product Sync"
    assert first.source == "https://www.youtube.com/watch?v=jNQXAC9IVRw" and first.source_type == "youtube"
    second = run_pipeline("https://www.youtube.com/watch?v=jNQXAC9IVRw", "english")
    assert second.meeting_id == first.meeting_id and len(transcriptions) == 1  # opened, not reprocessed
    hinglish = pipeline.source_identity("https://youtu.be/jNQXAC9IVRw", "hinglish")
    assert hinglish["meeting_id"] != first.meeting_id  # other language = other transcript
    meta = load_vector_store().get(where={"meeting_id": first.meeting_id}, include=["metadatas"])["metadatas"]
    assert meta[0]["source_title"] == "Me at the zoo" and meta[0]["title"] == "Weekly Product Sync"
