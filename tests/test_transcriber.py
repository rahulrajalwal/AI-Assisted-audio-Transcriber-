"""Whisper / Sarvam routing and Sarvam request handling (network mocked)."""

from __future__ import annotations

import pytest
from pydub.generators import Sine

import core.transcriber as tr


@pytest.fixture
def wav_60s(tmp_path):
    path = tmp_path / "chunk.wav"
    Sine(300).to_audio_segment(duration=60_000).set_channels(1).set_frame_rate(16_000).export(path, format="wav")
    return str(path)


class FakeResponse:
    def __init__(self, status, payload=None, headers=None):
        self.status_code = status
        self.ok = 200 <= status < 300
        self._payload = payload or {}
        self.headers = headers or {}
        self.text = str(payload)

    def json(self):
        return self._payload


def _segs(text, start=0.0, end=1.0):
    return [{"start": start, "end": end, "text": text}]


def test_routing(monkeypatch):
    monkeypatch.setattr(tr, "whisper_segments", lambda p: _segs("whisper"))
    monkeypatch.setattr(tr, "sarvam_segments", lambda p: _segs("sarvam"))
    assert tr.transcribe_chunk("x.wav", "English") == "whisper"
    assert tr.transcribe_chunk("x.wav", "hinglish") == "sarvam"


def test_invalid_language():
    with pytest.raises(ValueError, match="Unsupported language"):
        tr.transcribe_chunk("x.wav", "french")


def test_transcribe_all_joins_and_reports(monkeypatch):
    monkeypatch.setattr(tr, "load_model", lambda: None)
    monkeypatch.setattr(tr, "whisper_segments", lambda p: _segs(f"text-{p}"))
    monkeypatch.setattr(tr, "_chunk_duration_seconds", lambda p: 600.0)
    events = []
    text = tr.transcribe_all(["a", "b"], "english", progress=lambda m, f: events.append((m, f)))
    assert text == "text-a text-b"
    assert any("part 2/2" in m for m, _ in events)


def test_segments_get_meeting_offsets_and_char_positions(monkeypatch):
    monkeypatch.setattr(tr, "load_model", lambda: None)
    per_chunk = {
        "a": [{"start": 0.0, "end": 4.0, "text": "Hello team."}, {"start": 4.0, "end": 9.5, "text": "Second line."}],
        "b": [{"start": 1.0, "end": 3.0, "text": "After the break."}],
    }
    monkeypatch.setattr(tr, "whisper_segments", lambda p: per_chunk[p])
    monkeypatch.setattr(tr, "_chunk_duration_seconds", lambda p: 600.0)
    text, segments = tr.transcribe_all_with_segments(["a", "b"], "english")
    assert text == "Hello team. Second line. After the break."
    for seg in segments:  # char offsets point into the joined transcript
        assert text[seg["char_start"] : seg["char_end"]] in ("Hello team.", "Second line.", "After the break.")
    assert segments[2]["start"] == 601.0 and segments[2]["audio_chunk"] == 1


def test_empty_transcript_raises(monkeypatch):
    monkeypatch.setattr(tr, "load_model", lambda: None)
    monkeypatch.setattr(tr, "whisper_segments", lambda p: _segs("   "))
    monkeypatch.setattr(tr, "_chunk_duration_seconds", lambda p: 1.0)
    with pytest.raises(tr.TranscriptionError, match="No speech"):
        tr.transcribe_all(["a"], "english")


def test_missing_sarvam_key(monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    with pytest.raises(tr.TranscriptionError, match="SARVAM_API_KEY"):
        tr.transcribe_all(["a"], "hinglish")


def test_sarvam_reads_env_at_call_time(monkeypatch):
    """Regression: the key used to be read at import time, before .env was loaded."""
    monkeypatch.setenv("SARVAM_API_KEY", "late-key")
    monkeypatch.setenv("SARVAM_STT_MODEL", "saaras:v2.5")
    assert tr._sarvam_config() == ("late-key", "saaras:v2.5")


def test_sarvam_pieces_are_short_contiguous_and_ordered(monkeypatch, wav_60s):
    monkeypatch.setenv("SARVAM_API_KEY", "k")
    monkeypatch.setenv("SARVAM_CONCURRENCY", "3")
    import threading
    import time as _time

    calls, lock = [], threading.Lock()

    def fake_post(url, headers, files, data, timeout):
        name = files["file"][0]  # chunk.wav_sv_<index>.wav
        index = int(name.rsplit("_", 1)[1].split(".")[0])
        _time.sleep(0.05 * (3 - index))  # later pieces finish first: order must still be kept
        with lock:
            calls.append((index, headers, data))
        return FakeResponse(200, {"transcript": f"piece{index}"})

    monkeypatch.setattr(tr.requests, "post", fake_post)
    segments = tr.sarvam_segments(wav_60s)
    assert [s["text"] for s in segments] == [f"piece{i}" for i in range(len(segments))]
    assert segments[0]["start"] == 0.0 and segments[-1]["end"] == 60.0
    assert all(b["start"] == a["end"] for a, b in zip(segments, segments[1:]))  # contiguous
    assert all(s["end"] - s["start"] <= 29.0 for s in segments)  # Sarvam's 30 s limit
    assert len(calls) == len(segments) >= 3
    assert calls[0][1] == {"api-subscription-key": "k"} and calls[0][2]["model"]


def test_completed_chunks_are_not_transcribed_again(monkeypatch):
    monkeypatch.setattr(tr, "load_model", lambda: None)
    monkeypatch.setattr(tr, "_chunk_duration_seconds", lambda p: 600.0)
    calls, saved = [], {}
    monkeypatch.setattr(tr, "whisper_segments", lambda p: calls.append(p) or _segs(f"text-{p}"))
    done = {0: _segs("saved-a"), 1: _segs("saved-b")}
    text, segments = tr.transcribe_all_with_segments(
        ["a", "b", "c"], "english", completed=done, on_chunk_done=lambda i, s: saved.update({i: s})
    )
    assert calls == ["c"] and list(saved) == [2]
    assert text == "saved-a saved-b text-c"
    assert segments[2]["start"] == 1200.0 and segments[2]["audio_chunk"] == 2


def test_failed_chunk_reports_resume_point(monkeypatch):
    monkeypatch.setattr(tr, "load_model", lambda: None)
    monkeypatch.setattr(tr, "_chunk_duration_seconds", lambda p: 600.0)

    def flaky(path):
        if path == "c":
            raise RuntimeError("network down")
        return _segs(f"text-{path}")

    monkeypatch.setattr(tr, "whisper_segments", flaky)
    saved = {}
    with pytest.raises(tr.TranscriptionError, match="part 3 of 4") as info:
        tr.transcribe_all_with_segments(["a", "b", "c", "d"], "english", on_chunk_done=lambda i, s: saved.update({i: s}))
    assert sorted(saved) == [0, 1]  # chunks 1-2 were saved before the failure
    assert "continue from part 3" in str(info.value)


def test_sarvam_retries_rate_limit(monkeypatch, wav_60s):
    monkeypatch.setenv("SARVAM_API_KEY", "k")
    monkeypatch.setattr(tr.time, "sleep", lambda s: None)
    responses = [FakeResponse(429, {"error": "rate"}), FakeResponse(200, {"transcript": "ok"})]
    monkeypatch.setattr(tr.requests, "post", lambda *a, **k: responses.pop(0) if responses else FakeResponse(200, {"transcript": "ok"}))
    assert tr._send_to_sarvam(wav_60s, "k", "m") == "ok"


def test_sarvam_auth_error_not_retried(monkeypatch, wav_60s):
    calls = []
    monkeypatch.setattr(tr.requests, "post", lambda *a, **k: calls.append(1) or FakeResponse(401, {"error": "bad"}))
    with pytest.raises(tr.TranscriptionError, match="rejected"):
        tr._send_to_sarvam(wav_60s, "k", "m")
    assert len(calls) == 1
