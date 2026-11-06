"""End-to-end meeting pipeline shared by the Streamlit app and the CLI.

Stages (each reports real progress events through an optional callback):

1. input          – input validation and audio extraction (yt-dlp / local file)
2. audio          – conversion to mono 16 kHz WAV and AUDIO chunking (FFmpeg + pydub)
3. transcription  – Whisper (English) or Sarvam AI (Hinglish), with timed segments
4. analysis       – LLM (OpenRouter) Map-Reduce summary and meeting title
5. extraction     – action items, key decisions, open questions
6. indexing       – TEXT chunking → MiniLM embeddings → persistent ChromaDB (for RAG)

A failure in any stage raises :class:`PipelineError`; nothing is reported as
successful after a failed stage. If transcription succeeded, the error carries
the transcript (and timestamps) so the analysis can be retried without
re-transcribing.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.llm import LLMError, ensure_llm_ready, get_model_name
from core.meeting_store import MeetingRecord, load_meeting, save_meeting
from core.summarizer import generate_title, summarize
from core.transcriber import (
    LANGUAGE_ENGINES,
    TranscriptionError,
    normalize_language,
    transcribe_all_with_segments,
)
from core.vector_store import build_vector_store, compute_meeting_id
from utils.audio_processor import (
    AudioProcessingError,
    chunk_audio,
    cleanup_work_dir,
    convert_to_wav,
    download_youtube_audio,
    ensure_ffmpeg,
    get_duration_seconds,
    is_url,
    new_work_dir,
    validate_local_file,
    validate_youtube_url,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Stage:
    key: str
    label: str


STAGES: tuple[Stage, ...] = (
    Stage("input", "Input validation & audio extraction"),
    Stage("audio", "Audio conversion & chunking"),
    Stage("transcription", "Speech-to-text transcription"),
    Stage("analysis", "LLM meeting analysis (summary & title)"),
    Stage("extraction", "Information extraction (actions, decisions, questions)"),
    Stage("indexing", "Vector indexing & RAG initialisation"),
)
STAGE_LABELS = {stage.key: stage.label for stage in STAGES}


@dataclass
class ProgressEvent:
    stage: str
    status: str  # "running" | "done" | "error"
    message: str = ""
    fraction: float | None = None  # progress within the stage, 0..1, when known


ProgressCallback = Callable[[ProgressEvent], None]


class PipelineError(RuntimeError):
    def __init__(
        self,
        stage: str,
        message: str,
        *,
        transcript: str | None = None,
        duration_seconds: float | None = None,
        segments: list[dict] | None = None,
    ):
        super().__init__(message)
        self.stage = stage
        self.stage_label = STAGE_LABELS.get(stage, stage)
        self.transcript = transcript
        self.duration_seconds = duration_seconds
        self.segments = segments or []


_EXPECTED_ERRORS = (AudioProcessingError, TranscriptionError, LLMError, ValueError)


def _emit(progress: ProgressCallback | None, stage: str, status: str, message: str = "", fraction: float | None = None) -> None:
    if progress:
        try:
            progress(ProgressEvent(stage, status, message, fraction))
        except Exception:  # a UI callback must never break the pipeline
            logger.exception("Progress callback failed")


def _sub_progress(progress: ProgressCallback | None, stage: str) -> Callable[[str, float | None], None]:
    def report(message: str, fraction: float | None = None) -> None:
        _emit(progress, stage, "running", message, fraction)

    return report


def _run_stage(
    progress: ProgressCallback | None,
    stage: str,
    func: Callable[[], Any],
    *,
    start_message: str = "",
    done_message: Callable[[Any], str] | str = "",
    transcript: str | None = None,
    duration_seconds: float | None = None,
    segments: list[dict] | None = None,
) -> Any:
    _emit(progress, stage, "running", start_message, 0.0)
    context = {"transcript": transcript, "duration_seconds": duration_seconds, "segments": segments}
    try:
        result = func()
    except PipelineError:
        raise
    except _EXPECTED_ERRORS as exc:
        message = str(exc)
        _emit(progress, stage, "error", message)
        raise PipelineError(stage, message, **context) from exc
    except MemoryError as exc:
        message = "The computer ran out of memory. Close other applications or use smaller models (see README)."
        _emit(progress, stage, "error", message)
        raise PipelineError(stage, message, **context) from exc
    except Exception as exc:
        logger.exception("Unexpected error in stage %s", stage)
        message = f"Unexpected error: {type(exc).__name__}: {exc}"
        _emit(progress, stage, "error", message)
        raise PipelineError(stage, message, **context) from exc
    text = done_message(result) if callable(done_message) else done_message
    _emit(progress, stage, "done", text, 1.0)
    return result


def _format_duration(seconds: float | None) -> str:
    if not seconds:
        return ""
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {secs:02d}s" if hours else f"{minutes}m {secs:02d}s"


def validate_source(source: str) -> str:
    """Fast validation before any heavy work starts."""
    source = (source or "").strip()
    if not source:
        raise AudioProcessingError("Please provide a YouTube link or an audio/video file.")
    if is_url(source):
        return validate_youtube_url(source)
    validate_local_file(source)
    return source


def infer_source_type(source: str) -> str:
    return "youtube" if is_url(source) else "local_file"


def transcribe_source(
    source: str,
    language: str,
    progress: ProgressCallback | None = None,
) -> tuple[str, float | None, list[dict]]:
    """Stages 1–3: return ``(transcript, duration_seconds, segments)``. Temporary audio is always deleted."""
    work_directory: Path | None = None
    try:

        def acquire() -> str:
            nonlocal work_directory
            lang = normalize_language(language)
            cleaned = validate_source(source)
            ensure_ffmpeg()
            work_directory = new_work_dir()
            if is_url(cleaned):
                _emit(progress, "input", "running", "Downloading audio from YouTube…")
                path = download_youtube_audio(cleaned, work_directory)
            else:
                path = cleaned
            _emit(progress, "input", "running", f"Language: {lang} → {LANGUAGE_ENGINES[lang]}")
            return path

        raw_path = _run_stage(progress, "input", acquire, start_message="Validating input", done_message="Audio source ready")

        def prepare() -> tuple[list[str], float]:
            _emit(progress, "audio", "running", "Converting to mono 16 kHz WAV", 0.2)
            wav_path = convert_to_wav(raw_path, str(Path(work_directory) / "audio_16k_mono.wav"))
            duration = get_duration_seconds(wav_path)
            _emit(progress, "audio", "running", "Splitting audio into chunks", 0.7)
            chunks = chunk_audio(wav_path, output_dir=work_directory)
            gc.collect()  # drop decoded audio before the speech model loads
            return chunks, duration

        chunks, duration = _run_stage(
            progress, "audio", prepare,
            done_message=lambda r: f"{len(r[0])} audio chunk(s), {_format_duration(r[1])} of audio",
        )

        def transcribe() -> tuple[str, list[dict]]:
            return transcribe_all_with_segments(chunks, language, progress=_sub_progress(progress, "transcription"))

        transcript, segments = _run_stage(
            progress, "transcription", transcribe,
            start_message=f"Starting {LANGUAGE_ENGINES.get(language.lower(), '')}",
            done_message=lambda r: f"{len(r[0].split()):,} words transcribed",
            duration_seconds=duration,
        )
        return transcript, duration, segments
    finally:
        cleanup_work_dir(work_directory)


def analyse_transcript(
    transcript: str,
    *,
    source: str = "",
    source_type: str = "",
    language: str = "english",
    duration_seconds: float | None = None,
    segments: list[dict] | None = None,
    meeting_date: str = "",
    progress: ProgressCallback | None = None,
    force: bool = False,
) -> MeetingRecord:
    """Stages 4–6: LLM analysis, extraction and vector indexing."""
    transcript = (transcript or "").strip()
    common = {"transcript": transcript, "duration_seconds": duration_seconds, "segments": segments}
    if not transcript:
        raise PipelineError("analysis", "The transcript is empty.", **common)

    meeting_id = compute_meeting_id(transcript)
    record = None if force else load_meeting(meeting_id)

    if record is not None:
        message = "Reused saved analysis for this meeting (no LLM calls)"
        _emit(progress, "analysis", "done", message, 1.0)
        _emit(progress, "extraction", "done", message, 1.0)
    else:
        try:
            ensure_llm_ready()  # OpenRouter key valid, model available, free quota left?
        except LLMError as exc:
            _emit(progress, "analysis", "error", str(exc))
            raise PipelineError("analysis", str(exc), **common) from exc

        def analyse() -> tuple[str, str]:
            summary = summarize(transcript, progress=_sub_progress(progress, "analysis"))
            _emit(progress, "analysis", "running", "Generating meeting title", 0.9)
            return summary, generate_title(transcript, summary)

        summary, title = _run_stage(
            progress, "analysis", analyse,
            start_message=f"Summarising with {get_model_name()} via OpenRouter",
            done_message=lambda r: f"Title: {r[1]}",
            **common,
        )

        def extract() -> tuple[str, str, str]:
            report = _sub_progress(progress, "extraction")
            report("Extracting action items", 0.0)
            actions = extract_action_items(transcript, progress=report)
            report("Extracting key decisions", 1 / 3)
            decisions = extract_key_decisions(transcript, progress=report)
            report("Extracting open questions", 2 / 3)
            questions = extract_questions(transcript, progress=report)
            return actions, decisions, questions

        actions, decisions, questions = _run_stage(
            progress, "extraction", extract,
            done_message="Action items, decisions and open questions extracted",
            **common,
        )

        processed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        record = MeetingRecord(
            meeting_id=meeting_id,
            title=title,
            transcript=transcript,
            summary=summary,
            action_items=actions,
            key_decisions=decisions,
            open_questions=questions,
            source=source,
            language=normalize_language(language),
            processed_at=processed_at,
            model=get_model_name(),
            duration_seconds=duration_seconds,
            meeting_date=meeting_date or date.today().isoformat(),
            source_type=source_type or infer_source_type(source),
            segments=segments or [],
        )
        # Saved before indexing so a retry after an indexing failure needs no LLM calls.
        save_meeting(record)

    final_record = record
    _run_stage(
        progress, "indexing",
        lambda: build_vector_store(
            transcript,
            meeting_id=final_record.meeting_id,
            title=final_record.title,
            source=final_record.source,
            source_type=final_record.source_type,
            language=final_record.language,
            processed_at=final_record.processed_at,
            meeting_date=final_record.meeting_date,
            segments=final_record.segments,
            replace=force,  # intentional re-analysis refreshes this meeting's chunks only
        ),
        start_message="Splitting transcript into text chunks and embedding with all-MiniLM-L6-v2",
        done_message="Transcript stored in ChromaDB — ready for questions",
        **common,
    )
    return record


def run_pipeline(
    source: str,
    language: str = "english",
    progress: ProgressCallback | None = None,
    *,
    source_label: str | None = None,
    source_type: str = "",
    meeting_date: str = "",
    force: bool = False,
) -> MeetingRecord:
    """Run all six stages for a YouTube URL or local file path."""
    transcript, duration, segments = transcribe_source(source, language, progress)
    return analyse_transcript(
        transcript,
        source=source_label or source,
        source_type=source_type or infer_source_type(source),
        language=language,
        duration_seconds=duration,
        segments=segments,
        meeting_date=meeting_date,
        progress=progress,
        force=force,
    )
