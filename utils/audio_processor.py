"""Audio acquisition and preparation: yt-dlp (YouTube), FFmpeg + pydub (conversion/chunking).

Every input — a YouTube URL or a local audio/video file — ends up as mono,
16 kHz WAV chunks (about 10 minutes each) inside a working directory.

Long recordings (2 h+): conversion is streamed by FFmpeg straight to a 16 kHz mono
WAV (≈ 115 MB per hour) instead of decoding the whole file into Python memory, and
chunk boundaries are placed at the quietest moment near each cut point so words are
not split between chunks.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import uuid
import wave
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from pydub import AudioSegment

from core.config import work_dir

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1
DEFAULT_CHUNK_MINUTES = 10
CUT_SEARCH_WINDOW_MS = 30_000  # look this far either side of a chunk boundary for silence

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


def fetch_youtube_info(url: str) -> dict:
    """Video metadata (id, original title, duration) without downloading anything."""
    import yt_dlp

    url = validate_youtube_url(url)
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "noplaylist": True, "skip_download": True}) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:  # yt_dlp.utils.DownloadError and friends
        message = re.sub(r"\x1b\[[0-9;]*m", "", str(exc))
        raise AudioProcessingError(f"Could not read the YouTube video: {message}") from exc
    return {
        "id": info.get("id", ""),
        "title": info.get("title") or "",
        "duration": info.get("duration"),
        "url": info.get("webpage_url") or url,
    }


def download_youtube_audio(url: str, output_dir: str | Path | None = None) -> str:
    """Download the best audio stream as-is (m4a/webm); conversion happens in convert_to_wav."""
    import yt_dlp

    url = validate_youtube_url(url)
    ensure_ffmpeg()
    output_dir = Path(output_dir or new_work_dir())
    output_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "format": "bestaudio/best",
        # Video ID as the filename: avoids Windows-illegal characters from titles.
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 5,
        "fragment_retries": 5,
        "continuedl": True,  # resume a partially downloaded file after a failure
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            downloaded = Path(ydl.prepare_filename(info))
    except Exception as exc:  # yt_dlp.utils.DownloadError and friends
        message = re.sub(r"\x1b\[[0-9;]*m", "", str(exc))
        raise AudioProcessingError(f"Could not download the YouTube audio: {message}") from exc

    if not downloaded.exists():
        candidates = [c for c in sorted(output_dir.glob(f"{info.get('id', '*')}.*")) if not c.name.endswith(".part")]
        if not candidates:
            raise AudioProcessingError("The download finished but no audio file was produced.")
        downloaded = candidates[0]
    return str(downloaded)


def convert_to_wav(input_path: str, output_path: str | None = None) -> str:
    """Convert any audio/video file to mono 16 kHz WAV, streamed by FFmpeg (low memory)."""
    ensure_ffmpeg()
    source = Path(input_path)
    output = Path(output_path) if output_path else source.with_name(f"{source.stem}_converted.wav")
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
        "-vn", "-ac", str(TARGET_CHANNELS), "-ar", str(TARGET_SAMPLE_RATE), "-sample_fmt", "s16", str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not output.exists():
        output.unlink(missing_ok=True)
        raise AudioProcessingError(
            f"Could not read audio from '{source.name}'. The file may be corrupt or contain no audio track."
        )
    if get_duration_seconds(str(output)) <= 0:
        output.unlink(missing_ok=True)
        raise AudioProcessingError(f"'{source.name}' contains no audio.")
    return str(output)


def quiet_split_points(audio: AudioSegment, min_ms: int, max_ms: int, frame_ms: int = 100) -> list[tuple[int, int]]:
    """Split ``audio`` into consecutive (start, end) ranges between min_ms and max_ms long.

    Each cut is placed on the quietest ``frame_ms`` frame inside the allowed window, so
    boundaries fall in pauses rather than in the middle of a word.
    """
    ranges: list[tuple[int, int]] = []
    start, total = 0, len(audio)
    while total - start > max_ms:
        window_start, window_end = start + min_ms, start + max_ms
        best_cut, best_rms = window_end, None
        for frame_start in range(window_start, window_end - frame_ms + 1, frame_ms):
            rms = audio[frame_start : frame_start + frame_ms].rms
            if best_rms is None or rms < best_rms:
                best_cut, best_rms = frame_start + frame_ms // 2, rms
        ranges.append((start, best_cut))
        start = best_cut
    ranges.append((start, total))
    return ranges


def chunk_audio(
    wav_path: str,
    chunk_minutes: int = DEFAULT_CHUNK_MINUTES,
    output_dir: str | Path | None = None,
) -> list[str]:
    """Split a WAV file into ~``chunk_minutes`` chunks, cutting at the quietest nearby moment."""
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = int(chunk_minutes * 60 * 1000)
    window = min(CUT_SEARCH_WINDOW_MS, chunk_ms // 4)
    target_dir = Path(output_dir) if output_dir else Path(wav_path).parent
    stem = Path(wav_path).stem

    chunks: list[str] = []
    for index, (start, end) in enumerate(quiet_split_points(audio, chunk_ms - window, chunk_ms + window)):
        chunk_path = target_dir / f"{stem}_chunk_{index:03d}.wav"
        audio[start:end].export(str(chunk_path), format="wav")
        chunks.append(str(chunk_path))
    del audio  # free the decoded recording before transcription starts
    return chunks


def get_duration_seconds(wav_path: str) -> float:
    """Duration from the WAV header (no decoding)."""
    with wave.open(str(wav_path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate() or 1)


def file_fingerprint(path: str | Path, block_size: int = 1 << 20) -> str:
    """SHA-256 of a file's content — identifies an upload even if it is renamed."""
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


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
