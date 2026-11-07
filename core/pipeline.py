"""End-to-end meeting pipeline shared by the Streamlit app and the CLI.

Stages (each reports real progress events through an optional callback):

1. input          – validation, meeting identity + original title, audio download (yt-dlp) / local file
2. audio          – FFmpeg conversion to mono 16 kHz WAV and silence-aware AUDIO chunking
3. transcription  – Whisper (English) or Sarvam AI (Hinglish), with timed segments
4. analysis       – LLM (OpenRouter) Map-Reduce summary and AI title
5. extraction     – action items, key decisions, open questions
6. indexing       – TEXT chunking → MiniLM embeddings → persistent ChromaDB (for RAG)

Long recordings are checkpointed (``core.meeting_store.MeetingCheckpoint``): the
downloaded audio, the chunk plan, every transcribed chunk and every finished LLM
request are saved as soon as they complete. Running the same source again (or
:func:`resume_meeting`) continues from the first unfinished step — e.g. if part 10
of 12 failed, parts 1–9 are reused and transcription restarts at part 10.

Meeting identity: the meeting ID comes from the *source* (YouTube video ID, or the
file's content hash) plus the language, so it is known before transcription and
does not change on retry. The original YouTube title / filename is kept as
``source_title``; the AI-generated title is stored separately in ``title``.

A failure in any stage raises :class:`PipelineError`; nothing is reported as
successful after a failed stage.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.llm import LLMError, ensure_llm_ready, get_model_name
from core.meeting_store import MeetingCheckpoint, MeetingRecord, load_meeting, save_meeting
from core.summarizer import generate_title, summarize
from core.transcriber import (
    LANGUAGE_ENGINES,
    TranscriptionError,
    normalize_language,
    transcribe_all_with_segments,
)
from core.vector_store import build_vector_store, compute_meeting_id, count_meeting_chunks
from utils.audio_processor import (
    AudioProcessingError,
    chunk_audio,
    convert_to_wav,
    download_youtube_audio,
    ensure_ffmpeg,
    fetch_youtube_info,
    file_fingerprint,
    get_duration_seconds,
    is_url,
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
    """A failed stage. ``meeting_id`` is set once the meeting is known, so it can be resumed."""

    def __init__(
        self,
        stage: str,
        message: str,
        *,
        transcript: str | None = None,
        duration_seconds: float | None = None,
        segments: list[dict] | None = None,
        meeting_id: str | None = None,
    ):
        super().__init__(message)
        self.stage = stage
        self.stage_label = STAGE_LABELS.get(stage, stage)
        self.transcript = transcript
        self.duration_seconds = duration_seconds
        self.segments = segments or []
        self.meeting_id = meeting_id


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
    **context: Any,  # transcript / duration_seconds / segments / meeting_id for PipelineError
) -> Any:
    _emit(progress, stage, "running", start_message, 0.0)
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


@contextmanager
def _remember_failure() -> Iterator[None]:
    """Record a failure in the meeting's checkpoint so it can be listed and resumed later."""
    try:
        yield
    except PipelineError as exc:
        if exc.meeting_id:
            checkpoint = MeetingCheckpoint(exc.meeting_id)
            if checkpoint.data:
                checkpoint.update(status="failed", error=str(exc), failed_stage=exc.stage)
        raise


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


def source_identity(source: str, language: str, source_label: str | None = None, source_type: str = "") -> dict:
    """Stable meeting ID and original name of a source, known before any heavy work.

    YouTube: keyed by the video ID and named by the original video title (yt-dlp metadata).
    Files: keyed by a content hash (renaming the file does not change it) and named by
    the original filename. The language is part of the ID because English (Whisper) and
    Hinglish (Sarvam) produce different transcripts of the same audio.
    """
    if is_url(source):
        info = fetch_youtube_info(source)
        key, name, kind, origin = f"youtube:{info['id']}", info["title"], "youtube", info["url"]
    else:
        path = validate_local_file(source)
        origin = source_label or str(path)
        key, name, kind = f"file:{file_fingerprint(path)}", Path(origin).name, source_type or "local_file"
    return {
        "meeting_id": hashlib.sha256(f"{key}|{language}".encode("utf-8")).hexdigest()[:16],
        "fingerprint": key,
        "source_title": name or origin,
        "source_type": kind,
        "source": origin,
    }


@dataclass
class SourceTranscript:
    """Result of stages 1–3 (or the stored transcript of an already known meeting)."""

    meeting_id: str
    transcript: str
    duration_seconds: float | None
    segments: list[dict]
    source: str
    source_title: str
    source_type: str
    existing: MeetingRecord | None = None  # saved record for this source, if any


def transcribe_source(
    source: str,
    language: str,
    progress: ProgressCallback | None = None,
    *,
    source_label: str | None = None,
    source_type: str = "",
    meeting_date: str = "",
) -> SourceTranscript:
    """Stages 1–3 with checkpoints: identify the meeting, prepare audio, transcribe.

    Reuses whatever an earlier attempt finished: the stored transcript of a known
    meeting, the downloaded audio, the chunk files and every transcribed chunk.
    """
    state: dict[str, Any] = {}

    def acquire() -> str | None:
        lang = state["lang"] = normalize_language(language)  # unsupported → clear error, no download
        cleaned = validate_source(source)
        _emit(progress, "input", "running", "Reading source details", 0.1)
        ident = state["ident"] = source_identity(cleaned, lang, source_label, source_type)
        existing = load_meeting(ident["meeting_id"])
        if existing and existing.transcript:
            state["existing"] = existing
            return None  # transcript already stored — no audio work needed
        ensure_ffmpeg()
        checkpoint = state["checkpoint"] = MeetingCheckpoint(ident["meeting_id"])
        resumed = checkpoint.start(
            ident["fingerprint"], source=ident["source"], source_title=ident["source_title"],
            source_type=ident["source_type"], language=lang, meeting_date=meeting_date,
            source_url=cleaned if is_url(cleaned) else "", status="processing", error="",
        )
        if checkpoint.data.get("transcript"):
            return None  # transcribed by an earlier attempt that failed later
        saved = checkpoint.data.get("source_file")
        if saved and Path(saved).exists():
            _emit(progress, "input", "running", "Reusing the audio saved by the previous attempt", 0.5)
            return saved
        if resumed:
            _emit(progress, "input", "running", "Resuming the previous attempt", 0.3)
        if is_url(cleaned):
            _emit(progress, "input", "running", f"Downloading audio: {ident['source_title']}", 0.4)
            path = download_youtube_audio(cleaned, checkpoint.dir)
        elif ident["source_type"] == "upload":
            # Keep the uploaded file with the checkpoint so a retry (even after a restart) can resume.
            target = checkpoint.dir / f"source{Path(cleaned).suffix.lower()}"
            shutil.move(cleaned, target)
            path = str(target)
        else:
            path = cleaned  # the user's own file on disk is used in place — never moved or deleted
        checkpoint.update(source_file=path)
        _emit(progress, "input", "running", f"Language: {lang} → {LANGUAGE_ENGINES[lang]}")
        return path

    try:
        raw_path = _run_stage(
            progress, "input", acquire, start_message="Validating input",
            done_message=lambda _: f"Source: {state['ident']['source_title']}",
        )
    except PipelineError as exc:
        exc.meeting_id = state.get("ident", {}).get("meeting_id")
        raise
    ident, existing, lang = state["ident"], state.get("existing"), state["lang"]
    names = {key: ident[key] for key in ("source", "source_title", "source_type")}

    if existing is not None:
        note = "Reused the stored transcript (no audio processing needed)"
        _emit(progress, "audio", "done", note, 1.0)
        _emit(progress, "transcription", "done", note, 1.0)
        return SourceTranscript(
            ident["meeting_id"], existing.transcript, existing.duration_seconds, existing.segments,
            existing.source or ident["source"], ident["source_title"], ident["source_type"], existing,
        )

    checkpoint: MeetingCheckpoint = state["checkpoint"]
    if raw_path is None:
        note = "Reused the transcript saved by the previous attempt"
        _emit(progress, "audio", "done", note, 1.0)
        _emit(progress, "transcription", "done", note, 1.0)
        data = checkpoint.data
        return SourceTranscript(ident["meeting_id"], data["transcript"], data.get("duration"), data.get("segments") or [], **names)

    context = {"meeting_id": ident["meeting_id"]}

    def prepare() -> tuple[list[str], float]:
        plan = checkpoint.data.get("chunk_plan") or []
        if plan and all(Path(chunk).exists() for chunk in plan):
            _emit(progress, "audio", "running", f"Reusing {len(plan)} saved audio chunk(s)", 0.9)
            return plan, checkpoint.data.get("duration")
        _emit(progress, "audio", "running", "Converting to mono 16 kHz WAV (FFmpeg)", 0.2)
        wav_path = convert_to_wav(raw_path, str(checkpoint.dir / "audio_16k_mono.wav"))
        duration = get_duration_seconds(wav_path)
        _emit(progress, "audio", "running", f"Splitting {_format_duration(duration)} of audio at pauses", 0.6)
        chunks = chunk_audio(wav_path, output_dir=checkpoint.dir)
        Path(wav_path).unlink(missing_ok=True)  # the chunk files hold the same audio
        gc.collect()  # drop decoded audio before the speech model loads
        # A new chunk plan invalidates chunk transcripts saved for an older plan.
        checkpoint.update(chunk_plan=chunks, duration=duration, chunks={})
        return chunks, duration

    chunks, duration = _run_stage(
        progress, "audio", prepare,
        done_message=lambda r: f"{len(r[0])} audio chunk(s), {_format_duration(r[1])} of audio",
        **context,
    )

    def transcribe() -> tuple[str, list[dict]]:
        return transcribe_all_with_segments(
            chunks, lang, progress=_sub_progress(progress, "transcription"),
            completed=checkpoint.completed_chunks(), on_chunk_done=checkpoint.save_chunk,
        )

    transcript, segments = _run_stage(
        progress, "transcription", transcribe,
        start_message=f"Starting {LANGUAGE_ENGINES[lang]}",
        done_message=lambda r: f"{len(r[0].split()):,} words transcribed",
        duration_seconds=duration, **context,
    )
    # The finished transcript replaces the per-chunk results; the audio is no longer needed.
    checkpoint.update(transcript=transcript, segments=segments, chunks={}, chunk_plan=[], status="analysing")
    checkpoint.discard_files()
    return SourceTranscript(ident["meeting_id"], transcript, duration, segments, **names)


def analyse_transcript(
    transcript: str,
    *,
    source: str = "",
    source_type: str = "",
    source_title: str = "",
    language: str = "english",
    duration_seconds: float | None = None,
    segments: list[dict] | None = None,
    meeting_date: str = "",
    progress: ProgressCallback | None = None,
    force: bool = False,
    meeting_id: str | None = None,
) -> MeetingRecord:
    """Stages 4–6: LLM analysis, extraction and vector indexing.

    Every finished LLM result is cached in the meeting's checkpoint, so a retry after a
    failure (e.g. an OpenRouter rate limit) repeats only the requests that did not finish.
    ``force`` re-runs the analysis of an already processed meeting.
    """
    transcript = (transcript or "").strip()
    language = normalize_language(language)
    meeting_id = meeting_id or compute_meeting_id(transcript)
    common = {
        "transcript": transcript, "duration_seconds": duration_seconds, "segments": segments, "meeting_id": meeting_id,
    }
    if not transcript:
        raise PipelineError("analysis", "The transcript is empty.", **common)

    record = None if force else load_meeting(meeting_id)
    checkpoint = MeetingCheckpoint(meeting_id)
    if record is None or record.status != "complete":
        # Keep everything needed to resume the analysis later, even after a restart.
        if not checkpoint.data:
            checkpoint.start(f"transcript:{meeting_id}")
        checkpoint.update(
            transcript=transcript, segments=segments or [], duration=duration_seconds, source=source,
            source_title=source_title, source_type=source_type, language=language,
            meeting_date=meeting_date, force=force, status="analysing", error="",
        )

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
            summary = summarize(transcript, progress=_sub_progress(progress, "analysis"), cache=checkpoint)
            _emit(progress, "analysis", "running", "Generating AI title", 0.9)
            return summary, generate_title(transcript, summary, cache=checkpoint)

        summary, title = _run_stage(
            progress, "analysis", analyse,
            start_message=f"Summarising with {get_model_name()} via OpenRouter",
            done_message=lambda r: f"AI title: {r[1]}",
            **common,
        )

        def extract() -> tuple[str, str, str]:
            report = _sub_progress(progress, "extraction")
            report("Extracting action items", 0.0)
            actions = extract_action_items(transcript, progress=report, cache=checkpoint)
            report("Extracting key decisions", 1 / 3)
            decisions = extract_key_decisions(transcript, progress=report, cache=checkpoint)
            report("Extracting open questions", 2 / 3)
            questions = extract_questions(transcript, progress=report, cache=checkpoint)
            return actions, decisions, questions

        actions, decisions, questions = _run_stage(
            progress, "extraction", extract,
            done_message="Action items, decisions and open questions extracted",
            **common,
        )

        record = MeetingRecord(
            meeting_id=meeting_id,
            title=title,
            transcript=transcript,
            summary=summary,
            action_items=actions,
            key_decisions=decisions,
            open_questions=questions,
            source=source,
            language=language,
            processed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            model=get_model_name(),
            duration_seconds=duration_seconds,
            meeting_date=meeting_date or date.today().isoformat(),
            source_type=source_type or infer_source_type(source),
            segments=segments or [],
            source_title=source_title,
            status="indexing",
        )
        # Saved before indexing so a retry after an indexing failure needs no LLM calls.
        save_meeting(record)

    final = record

    def index() -> int:
        store = build_vector_store(
            transcript,
            meeting_id=final.meeting_id,
            title=final.title,
            source_title=final.display_name,
            source=final.source,
            source_type=final.source_type,
            language=final.language,
            processed_at=final.processed_at,
            meeting_date=final.meeting_date,
            segments=final.segments,
            replace=force,  # intentional re-analysis refreshes this meeting's chunks only
        )
        return count_meeting_chunks(final.meeting_id, store)

    count = _run_stage(
        progress, "indexing", index,
        start_message="Splitting transcript into text chunks and embedding with all-MiniLM-L6-v2",
        done_message=lambda n: f"{n} transcript chunks stored in ChromaDB — ready for questions",
        **common,
    )
    if record.status != "complete" or record.vector_chunks != count:
        record.status, record.vector_chunks = "complete", count
        save_meeting(record)
    checkpoint.remove()  # fully processed: the checkpoint is no longer needed
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
    """Run all six stages for a YouTube URL or local file, resuming any saved progress.

    An already processed source opens its saved meeting (no work repeated) unless
    ``force`` is set, which re-runs the analysis on the stored transcript.
    """
    with _remember_failure():
        result = transcribe_source(
            source, language, progress, source_label=source_label, source_type=source_type,
            meeting_date=meeting_date,
        )
        existing = result.existing
        if existing and existing.status == "complete" and not force:
            note = "Already processed — opened the saved meeting (no work repeated)"
            for stage in ("analysis", "extraction", "indexing"):
                _emit(progress, stage, "done", note, 1.0)
            if not existing.source_title:  # older record: remember the original name now
                existing.source_title = result.source_title
                save_meeting(existing)
            return existing
        return analyse_transcript(
            result.transcript,
            source=result.source,
            source_type=result.source_type,
            source_title=result.source_title,
            language=language,
            duration_seconds=result.duration_seconds,
            segments=result.segments,
            meeting_date=meeting_date or (existing.meeting_date if existing else ""),
            progress=progress,
            force=force,
            meeting_id=result.meeting_id,
        )


def resume_meeting(meeting_id: str, progress: ProgressCallback | None = None) -> MeetingRecord:
    """Continue an unfinished meeting from its checkpoint (also works after an app restart)."""
    data = MeetingCheckpoint(meeting_id).data
    record = load_meeting(meeting_id)
    with _remember_failure():
        if data.get("transcript") or (record and record.transcript):
            info = record.to_dict() if record else {}
            info.update({key: value for key, value in data.items() if value not in (None, "")})
            return analyse_transcript(
                info["transcript"],
                source=info.get("source", ""),
                source_type=info.get("source_type", ""),
                source_title=info.get("source_title", ""),
                language=info.get("language", "english"),
                duration_seconds=info.get("duration") or info.get("duration_seconds"),
                segments=info.get("segments"),
                meeting_date=info.get("meeting_date", ""),
                progress=progress,
                force=bool(data.get("force")),
                meeting_id=meeting_id,
            )
        source = data.get("source_url") or data.get("source_file")
        if not source:
            raise PipelineError("input", "Nothing to resume: no saved progress was found for this meeting.")
        return run_pipeline(
            source, data.get("language", "english"), progress, source_label=data.get("source"),
            source_type=data.get("source_type", ""), meeting_date=data.get("meeting_date", ""),
        )
