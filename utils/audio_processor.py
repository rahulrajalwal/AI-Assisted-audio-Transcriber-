"""Audio acquisition and preparation: yt-dlp (YouTube), FFmpeg + pydub (conversion/chunking).

Every input — a YouTube URL or a local audio/video file — ends up as mono,
16 kHz WAV chunks (default 10 minutes each) inside a working directory that
the caller can delete once transcription is finished.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from pydub import AudioSegment

from core.config import work_dir

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1
DEFAULT_CHUNK_MINUTES = 10

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".amr"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".wmv", ".flv", ".mpeg", ".mpg", ".3gp"}
SUPPORTED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be", "www.youtu.be"}

ProgressFn = Callable[[str, float | None], None]


class AudioProcessingError(RuntimeError):
    """Raised for invalid input or failed download/conversion."""


# ── Validation ──────────────────────────────────────────────────────────────────


def ensure_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise AudioProcessingError(
            "FFmpeg was not found on PATH. Install it (e.g. `winget install Gyan.FFmpeg`) "
            "and restart the terminal/app."
        )


def is_url(source: str) -> bool:
    return (source or "").strip().lower().startswith(("http://", "https://"))


def validate_youtube_url(url: str) -> str:
    """Return the cleaned URL or raise AudioProcessingError."""
    url = (url or "").strip()
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https") or host not in YOUTUBE_HOSTS:
        raise AudioProcessingError("Please enter a valid YouTube link (youtube.com or youtu.be).")
    if host.endswith("youtu.be"):
        if not parsed.path.strip("/"):
            raise AudioProcessingError("The youtu.be link is missing a video ID.")
    elif not (
        "v=" in (parsed.query or "")
        or re.match(r"^/(shorts|live|embed)/[\w-]+", parsed.path or "")
    ):
        raise AudioProcessingError("The YouTube link does not point to a video.")
    return url


def validate_local_file(path: str) -> Path:
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise AudioProcessingError(f"File not found: {file_path.name}")
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise AudioProcessingError(
            f"Unsupported file type '{file_path.suffix}'. Supported: "
            + ", ".join(sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS))
        )
    if file_path.stat().st_size == 0:
        raise AudioProcessingError("The file is empty.")
    return file_path


def new_work_dir() -> Path:
    """Create a unique working directory for one processing run."""
    path = work_dir() / f"run_{uuid.uuid4().hex[:12]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cleanup_work_dir(path: str | Path | None) -> None:
    """Delete a run's working directory (only directories inside ``work_dir()``)."""
    if not path:
        return
    path = Path(path).resolve()
    root = work_dir().resolve()
    if path != root and root in path.parents and path.exists():
        shutil.rmtree(path, ignore_errors=True)


# ── Download / conversion / chunking ────────────────────────────────────────────


def download_youtube_audio(url: str, output_dir: str | Path | None = None) -> str:
    """Download the best audio stream and extract it to WAV with FFmpeg."""
    import yt_dlp

    url = validate_youtube_url(url)
    ensure_ffmpeg()
    output_dir = Path(output_dir or new_work_dir())
    output_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "format": "bestaudio/best",
        # Video ID as the filename: avoids Windows-illegal characters from titles.
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            downloaded = Path(ydl.prepare_filename(info))
    except Exception as exc:  # yt_dlp.utils.DownloadError and friends
        message = re.sub(r"\x1b\[[0-9;]*m", "", str(exc))
        raise AudioProcessingError(f"Could not download the YouTube audio: {message}") from exc

    wav_path = downloaded.with_suffix(".wav")
    if not wav_path.exists():
        candidates = sorted(output_dir.glob(f"{info.get('id', '*')}*.wav"))
        if not candidates:
            raise AudioProcessingError("The download finished but no WAV file was produced.")
        wav_path = candidates[0]
    return str(wav_path)


def convert_to_wav(input_path: str, output_path: str | None = None) -> str:
    """Convert any audio/video file to mono 16 kHz WAV using pydub/FFmpeg."""
    ensure_ffmpeg()
    source = Path(input_path)
    output = Path(output_path) if output_path else source.with_name(f"{source.stem}_converted.wav")
    try:
        audio = AudioSegment.from_file(str(source))
    except Exception as exc:
        raise AudioProcessingError(
            f"Could not read audio from '{source.name}'. The file may be corrupt or contain no audio track."
        ) from exc
    if len(audio) == 0:
        raise AudioProcessingError(f"'{source.name}' contains no audio.")
    audio = audio.set_channels(TARGET_CHANNELS).set_frame_rate(TARGET_SAMPLE_RATE)
    audio.export(str(output), format="wav")
    return str(output)


def chunk_audio(
    wav_path: str,
    chunk_minutes: int = DEFAULT_CHUNK_MINUTES,
    output_dir: str | Path | None = None,
) -> list[str]:
    """Split a WAV file into consecutive chunks of ``chunk_minutes`` minutes."""
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000
    target_dir = Path(output_dir) if output_dir else Path(wav_path).parent
    stem = Path(wav_path).stem

    chunks: list[str] = []
    for index, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk_path = target_dir / f"{stem}_chunk_{index:03d}.wav"
        audio[start : start + chunk_ms].export(str(chunk_path), format="wav")
        chunks.append(str(chunk_path))
    return chunks


def get_duration_seconds(wav_path: str) -> float:
    return len(AudioSegment.from_wav(wav_path)) / 1000.0


def process_input(
    source: str,
    work_directory: str | Path | None = None,
    progress: ProgressFn | None = None,
) -> list[str]:
    """YouTube URL or local file → list of mono 16 kHz WAV chunk paths.

    Files are written to ``work_directory`` (a new run directory if omitted);
    the caller is responsible for cleaning it up.
    """

    def report(message: str, fraction: float | None = None) -> None:
        if progress:
            progress(message, fraction)

    work_directory = Path(work_directory or new_work_dir())
    if is_url(source):
        report("Downloading audio from YouTube", None)
        raw_path = download_youtube_audio(source, work_directory)
    else:
        local = validate_local_file(source)
        raw_path = str(local)

    report("Converting audio to mono 16 kHz WAV", None)
    wav_path = convert_to_wav(raw_path, str(work_directory / "audio_16k_mono.wav"))

    report("Splitting audio into chunks", None)
    chunks = chunk_audio(wav_path, output_dir=work_directory)
    logger.info("Audio ready — %d chunk(s) created.", len(chunks))
    return chunks
