"""Speech-to-text: local OpenAI Whisper (English) and Sarvam AI (Hinglish).

* ``english``  → Whisper, running locally (model from ``WHISPER_MODEL``, default ``small``).
* ``hinglish`` → Sarvam AI ``speech-to-text-translate`` (returns English text).

Configuration is read when a transcription starts (not at import time), so the
values from ``.env`` are always picked up.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable

import requests
from pydub import AudioSegment

from core.config import load_environment

logger = logging.getLogger(__name__)

# Sarvam's sync STT-translate API rejects audio longer than 30s.
# Each chunk is sliced into 25s pieces (5s safety margin) before sending.
SARVAM_PIECE_SECONDS = 25
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_DEFAULT_MODEL = "saaras:v2.5"
SARVAM_MAX_ATTEMPTS = 3
SARVAM_TIMEOUT_SECONDS = 120

LANGUAGE_ENGINES = {
    "english": "Whisper (local)",
    "hinglish": "Sarvam AI",
}

ProgressFn = Callable[[str, float | None], None]

_model = None
_model_name: str | None = None
_model_lock = threading.Lock()


class TranscriptionError(RuntimeError):
    """Raised when audio cannot be transcribed."""


def normalize_language(language: str) -> str:
    value = (language or "").strip().lower()
    if value not in LANGUAGE_ENGINES:
        raise ValueError(
            f"Unsupported language '{language}'. Choose one of: {', '.join(LANGUAGE_ENGINES)}."
        )
    return value


def get_whisper_model_name() -> str:
    load_environment()
    return (os.getenv("WHISPER_MODEL") or "small").strip().lower()


def load_model():
    """Load the Whisper model once per process (thread-safe)."""
    global _model, _model_name
    name = get_whisper_model_name()
    with _model_lock:
        if _model is None or _model_name != name:
            import whisper  # imported lazily: heavy (torch)

            logger.info("Loading Whisper model '%s'…", name)
            try:
                _model = whisper.load_model(name)
                _model_name = name
            except Exception as exc:
                raise TranscriptionError(
                    f"Failed to load Whisper model '{name}'. Check WHISPER_MODEL in .env "
                    f"(e.g. tiny, base, small, medium). Original error: {exc}"
                ) from exc
    return _model


Segment = dict  # {"start": float, "end": float, "text": str} — seconds, relative to the chunk


def whisper_segments(chunk_path: str) -> list[Segment]:
    """Transcribe one chunk with Whisper and return its timed segments."""
    import torch

    model = load_model()
    result = model.transcribe(
        chunk_path,
        task="transcribe",
        language="en",
        fp16=torch.cuda.is_available(),
        verbose=None,  # no console progress bar
    )
    segments = [
        {"start": float(seg["start"]), "end": float(seg["end"]), "text": seg["text"].strip()}
        for seg in result.get("segments") or []
        if seg.get("text", "").strip()
    ]
    if not segments and (result.get("text") or "").strip():
        segments = [{"start": 0.0, "end": 0.0, "text": result["text"].strip()}]
    return segments


def transcribe_chunk_whisper(chunk_path: str) -> str:
    return " ".join(seg["text"] for seg in whisper_segments(chunk_path)).strip()


def _sarvam_config() -> tuple[str, str]:
    load_environment()
    api_key = (os.getenv("SARVAM_API_KEY") or "").strip()
    if not api_key:
        raise TranscriptionError(
            "SARVAM_API_KEY is not set. Add it to the .env file to use Hinglish transcription."
        )
    model = (os.getenv("SARVAM_STT_MODEL") or SARVAM_DEFAULT_MODEL).strip()
    return api_key, model


def _send_to_sarvam(piece_path: str, api_key: str | None = None, model: str | None = None) -> str:
    """Send one ≤30s WAV file to Sarvam and return the English transcript."""
    if api_key is None or model is None:
        api_key, model = _sarvam_config()
    headers = {"api-subscription-key": api_key}

    for attempt in range(1, SARVAM_MAX_ATTEMPTS + 1):
        try:
            with open(piece_path, "rb") as audio_file:
                files = {"file": (os.path.basename(piece_path), audio_file, "audio/wav")}
                data = {"model": model, "with_diarization": "false"}
                response = requests.post(
                    SARVAM_STT_TRANSLATE_URL,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=SARVAM_TIMEOUT_SECONDS,
                )
        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt == SARVAM_MAX_ATTEMPTS:
                raise TranscriptionError(f"Could not reach Sarvam AI: {exc}") from exc
            time.sleep(2**attempt)
            continue

        if response.ok:
            return (response.json().get("transcript") or "").strip()

        retryable = response.status_code == 429 or response.status_code >= 500
        if retryable and attempt < SARVAM_MAX_ATTEMPTS:
            retry_after = response.headers.get("Retry-After", "")
            delay = float(retry_after) if retry_after.isdigit() else 2**attempt
            logger.warning("Sarvam returned %s; retrying in %.0fs.", response.status_code, delay)
            time.sleep(min(delay, 30))
            continue

        hint = {
            401: "the SARVAM_API_KEY was rejected",
            403: "the SARVAM_API_KEY is not allowed to use this model",
            429: "the Sarvam rate limit / credits are exhausted",
        }.get(response.status_code, "the request failed")
        body = response.text[:300]
        raise TranscriptionError(
            f"Sarvam AI returned HTTP {response.status_code}: {hint}. Response: {body}"
        )
    raise TranscriptionError("Sarvam AI transcription failed.")  # pragma: no cover


def sarvam_segments(chunk_path: str) -> list[Segment]:
    """Split a chunk into ≤25s pieces, send each to Sarvam; one timed segment per piece."""
    api_key, model = _sarvam_config()
    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000
    total_pieces = max(1, (len(audio) + piece_ms - 1) // piece_ms)

    segments: list[Segment] = []
    for index, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start : start + piece_ms]
        piece_path = f"{chunk_path}_sv_{index}.wav"
        piece.export(piece_path, format="wav")
        try:
            logger.info("Sarvam piece %d/%d", index + 1, total_pieces)
            text = _send_to_sarvam(piece_path, api_key, model)
            if text:
                segments.append({"start": start / 1000, "end": (start + len(piece)) / 1000, "text": text})
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)
    del audio  # free the decoded chunk promptly
    return segments


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    return " ".join(seg["text"] for seg in sarvam_segments(chunk_path)).strip()


def transcribe_chunk_segments(chunk_path: str, language: str = "english") -> list[Segment]:
    """Route one chunk to Whisper (english) or Sarvam (hinglish); return timed segments."""
    if normalize_language(language) == "hinglish":
        return sarvam_segments(chunk_path)
    return whisper_segments(chunk_path)


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """Route one chunk to Whisper (english) or Sarvam (hinglish); return plain text."""
    return " ".join(seg["text"] for seg in transcribe_chunk_segments(chunk_path, language)).strip()


def _chunk_duration_seconds(chunk_path: str) -> float:
    try:
        return len(AudioSegment.from_wav(chunk_path)) / 1000.0
    except Exception:
        return 0.0


def transcribe_all_with_segments(
    chunks: list[str],
    language: str = "english",
    progress: ProgressFn | None = None,
) -> tuple[str, list[dict]]:
    """Transcribe all audio chunks in order.

    Returns ``(transcript, segments)`` where each segment has meeting-relative
    ``start``/``end`` seconds, its ``audio_chunk`` index and its ``char_start``/
    ``char_end`` position in the transcript — used to timestamp the text chunks
    stored in ChromaDB.
    """
    language = normalize_language(language)
    if not chunks:
        raise TranscriptionError("No audio chunks to transcribe.")

    engine = LANGUAGE_ENGINES[language]
    if language == "english":
        if progress:
            progress(f"Loading Whisper model '{get_whisper_model_name()}'", 0.0)
        load_model()
    else:
        _sarvam_config()  # fail fast if the key is missing

    transcript_parts: list[str] = []
    segments: list[dict] = []
    cursor = 0  # character position in the joined transcript
    offset = 0.0  # seconds from the start of the meeting
    for index, chunk in enumerate(chunks):
        if progress:
            progress(f"{engine}: transcribing part {index + 1}/{len(chunks)}", index / len(chunks))
        try:
            chunk_segments = transcribe_chunk_segments(chunk, language=language)
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(f"{engine} failed on audio part {index + 1}/{len(chunks)}: {exc}") from exc
        for seg in chunk_segments:
            text = seg["text"].strip()
            if not text:
                continue
            if transcript_parts:
                cursor += 1  # the joining space
            segments.append(
                {
                    "start": round(offset + seg["start"], 2),
                    "end": round(offset + seg["end"], 2),
                    "audio_chunk": index,
                    "char_start": cursor,
                    "char_end": cursor + len(text),
                }
            )
            transcript_parts.append(text)
            cursor += len(text)
        offset += _chunk_duration_seconds(chunk)

    transcript = " ".join(transcript_parts).strip()
    if not transcript:
        raise TranscriptionError(
            "No speech was detected in the audio. Check that the recording contains audible speech "
            "and that the correct language was selected."
        )
    return transcript, segments


def transcribe_all(
    chunks: list[str],
    language: str = "english",
    progress: ProgressFn | None = None,
) -> str:
    """Transcribe all chunks in order and return the full transcript."""
    return transcribe_all_with_segments(chunks, language, progress)[0]
