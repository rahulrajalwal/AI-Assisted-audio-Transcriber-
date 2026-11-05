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


def test_routing(monkeypatch):
    monkeypatch.setattr(tr, "transcribe_chunk_whisper", lambda p: "whisper")
    monkeypatch.setattr(tr, "transcribe_chunk_sarvam", lambda p: "sarvam")
    assert tr.transcribe_chunk("x.wav", "English") == "whisper"
    assert tr.transcribe_chunk("x.wav", "hinglish") == "sarvam"


def test_invalid_language():
    with pytest.raises(ValueError, match="Unsupported language"):
        tr.transcribe_chunk("x.wav", "french")


def test_transcribe_all_joins_and_reports(monkeypatch):
    monkeypatch.setattr(tr, "load_model", lambda: None)
    monkeypatch.setattr(tr, "transcribe_chunk_whisper", lambda p: f"text-{p}")
    events = []
    text = tr.transcribe_all(["a", "b"], "english", progress=lambda m, f: events.append((m, f)))
    assert text == "text-a text-b"
    assert any("part 2/2" in m for m, _ in events)


def test_empty_transcript_raises(monkeypatch):
    monkeypatch.setattr(tr, "load_model", lambda: None)
    monkeypatch.setattr(tr, "transcribe_chunk_whisper", lambda p: "   ")
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


def test_sarvam_splits_into_25s_pieces(monkeypatch, wav_60s):
    monkeypatch.setenv("SARVAM_API_KEY", "k")
    calls = []

    def fake_post(url, headers, files, data, timeout):
        calls.append((url, headers, data))
        return FakeResponse(200, {"transcript": f"piece{len(calls)}"})

    monkeypatch.setattr(tr.requests, "post", fake_post)
    assert tr.transcribe_chunk_sarvam(wav_60s) == "piece1 piece2 piece3"
    assert len(calls) == 3  # 25s + 25s + 10s
    assert calls[0][1] == {"api-subscription-key": "k"}
    assert calls[0][2]["model"]


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
