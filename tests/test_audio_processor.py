"""yt-dlp / FFmpeg / pydub audio handling (no network: yt-dlp is mocked)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from pydub import AudioSegment
from pydub.generators import Sine

import utils.audio_processor as ap
from core.config import work_dir
from tests.conftest import requires_ffmpeg


@pytest.fixture
def tone_wav(tmp_path) -> Path:
    path = tmp_path / "tone.wav"
    Sine(440).to_audio_segment(duration=3_000).set_channels(2).set_frame_rate(44_100).export(path, format="wav")
    return path


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/HW31sw4zCWg",
        "https://www.youtube.com/shorts/abc123",
        "https://m.youtube.com/watch?v=abc&t=10s",
    ],
)
def test_valid_youtube_urls(url):
    assert ap.validate_youtube_url(url) == url


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/watch?v=abc",
        "ftp://youtube.com/watch?v=abc",
        "https://www.youtube.com/playlist?list=abc",
        "https://youtu.be/",
        "not a url",
    ],
)
def test_invalid_youtube_urls(url):
    with pytest.raises(ap.AudioProcessingError):
        ap.validate_youtube_url(url)


def test_local_file_validation(tmp_path):
    with pytest.raises(ap.AudioProcessingError, match="not found"):
        ap.validate_local_file(str(tmp_path / "missing.mp3"))
    bad = tmp_path / "notes.txt"
    bad.write_text("hi")
    with pytest.raises(ap.AudioProcessingError, match="Unsupported"):
        ap.validate_local_file(str(bad))
    empty = tmp_path / "empty.mp3"
    empty.touch()
    with pytest.raises(ap.AudioProcessingError, match="empty"):
        ap.validate_local_file(str(empty))


@requires_ffmpeg
def test_convert_to_mono_16k(tone_wav, tmp_path):
    out = ap.convert_to_wav(str(tone_wav), str(tmp_path / "out.wav"))
    audio = AudioSegment.from_wav(out)
    assert audio.channels == 1 and audio.frame_rate == 16_000
    assert abs(len(audio) - 3_000) < 50


@requires_ffmpeg
def test_convert_video_file(tmp_path, tone_wav):
    video = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:r=5",
         "-i", str(tone_wav), "-shortest", "-c:v", "libx264", "-c:a", "aac", str(video)],
        check=True,
    )
    audio = AudioSegment.from_wav(ap.convert_to_wav(str(video), str(tmp_path / "v.wav")))
    assert audio.channels == 1 and audio.frame_rate == 16_000 and len(audio) > 2_500


@requires_ffmpeg
def test_corrupt_file_raises(tmp_path):
    bad = tmp_path / "bad.mp3"
    bad.write_text("this is not audio")
    with pytest.raises(ap.AudioProcessingError, match="Could not read audio"):
        ap.convert_to_wav(str(bad))


@requires_ffmpeg
def test_chunking_covers_audio_in_order(tmp_path):
    wav = tmp_path / "long.wav"
    Sine(300).to_audio_segment(duration=130_000).set_channels(1).set_frame_rate(16_000).export(wav, format="wav")
    chunks = ap.chunk_audio(str(wav), chunk_minutes=1, output_dir=tmp_path)
    lengths = [len(AudioSegment.from_wav(c)) for c in chunks]
    assert sum(lengths) == 130_000  # nothing lost or duplicated
    assert all(45_000 <= n <= 75_000 for n in lengths[:-1])  # ~1 minute each (cut window +-15 s)


def test_cuts_land_in_silence():
    from pydub import AudioSegment as Seg

    tone = Sine(440).to_audio_segment(duration=1_000).set_frame_rate(16_000).set_channels(1)
    # 58 s of "speech", a 1-second pause, then more speech: the cut must fall inside the pause
    audio = tone * 58 + Seg.silent(duration=1_000, frame_rate=16_000) + tone * 30
    (start, cut), (cut2, end) = ap.quiet_split_points(audio, 45_000, 75_000)
    assert start == 0 and cut == cut2 and end == len(audio)
    assert 58_000 <= cut <= 59_000


@requires_ffmpeg
def test_duration_from_header(tone_wav):
    assert abs(ap.get_duration_seconds(str(tone_wav)) - 3.0) < 0.01


@requires_ffmpeg
def test_process_input_local_file(tone_wav):
    run_dir = ap.new_work_dir()
    chunks = ap.process_input(str(tone_wav), run_dir)
    assert len(chunks) == 1 and Path(chunks[0]).parent == run_dir
    ap.cleanup_work_dir(run_dir)
    assert not run_dir.exists()


def test_cleanup_refuses_paths_outside_work_dir(tmp_path):
    outside = tmp_path / "keep_me"
    outside.mkdir()
    ap.cleanup_work_dir(outside)
    ap.cleanup_work_dir(work_dir())
    assert outside.exists()


@requires_ffmpeg
def test_youtube_download_keeps_original_stream(monkeypatch, tmp_path, tone_wav):
    """yt-dlp is mocked: verify the WAV produced by FFmpegExtractAudio is located correctly."""
    import yt_dlp

    captured = {}

    class FakeYDL:
        def __init__(self, opts):
            captured["opts"] = opts

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download):
            out_dir = Path(captured["opts"]["outtmpl"]).parent
            (out_dir / "abc123.webm").write_bytes(tone_wav.read_bytes())
            return {"id": "abc123", "ext": "webm", "title": "Original Title"}

        def prepare_filename(self, info):
            return str(Path(captured["opts"]["outtmpl"]).parent / "abc123.webm")

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    path = ap.download_youtube_audio("https://youtu.be/abc123", tmp_path)
    assert path.endswith("abc123.webm") and Path(path).exists()  # original stream, no huge WAV
    assert captured["opts"]["noplaylist"] is True and "postprocessors" not in captured["opts"]


@requires_ffmpeg
def test_youtube_download_error_is_wrapped(monkeypatch, tmp_path):
    import yt_dlp

    class FailingYDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download):
            raise yt_dlp.utils.DownloadError("Video unavailable")

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FailingYDL)
    with pytest.raises(ap.AudioProcessingError, match="Video unavailable"):
        ap.download_youtube_audio("https://youtu.be/abc123", tmp_path)


def _client_ydl(attempts: list, refuse: set, out_dir: Path):
    """Fake YoutubeDL that refuses (HTTP 403) the given player clients and records each attempt."""
    import yt_dlp

    class ClientYDL:
        def __init__(self, opts):
            self.client = (opts.get("extractor_args") or {}).get("youtube", {}).get("player_client", [None])[0]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download):
            attempts.append(self.client)
            if self.client in refuse:
                (out_dir / "abc123.webm.part").write_bytes(b"partial")
                raise yt_dlp.utils.DownloadError("ERROR: unable to download video data: HTTP Error 403: Forbidden")
            (out_dir / "abc123.webm").write_bytes(b"audio")
            return {"id": "abc123", "ext": "webm"}

        def prepare_filename(self, info):
            return str(out_dir / "abc123.webm")

    return ClientYDL


@requires_ffmpeg
def test_youtube_403_retries_other_player_clients(monkeypatch, tmp_path):
    """Streamlit Cloud: YouTube refuses some clients (HTTP 403) for data-centre IPs — try the next ones."""
    import yt_dlp

    attempts: list = []
    monkeypatch.setattr(yt_dlp, "YoutubeDL", _client_ydl(attempts, {None, "tv_simply"}, tmp_path))
    path = ap.download_youtube_audio("https://youtu.be/abc123", tmp_path)
    assert attempts == [None, "tv_simply", "tv"]  # default clients first, then fallbacks in order
    assert Path(path).read_bytes() == b"audio" and not list(tmp_path.glob("*.part"))  # no stale partials


@requires_ffmpeg
def test_youtube_403_everywhere_gives_actionable_error(monkeypatch, tmp_path):
    import yt_dlp

    attempts: list = []
    monkeypatch.setattr(yt_dlp, "YoutubeDL", _client_ydl(attempts, set(ap.YOUTUBE_FALLBACK_CLIENTS), tmp_path))
    with pytest.raises(ap.AudioProcessingError, match="403.*Deno.*upload the audio or video file"):
        ap.download_youtube_audio("https://youtu.be/abc123", tmp_path)
    assert attempts == list(ap.YOUTUBE_FALLBACK_CLIENTS)


@requires_ffmpeg
def test_youtube_non_403_error_is_not_retried(monkeypatch, tmp_path):
    import yt_dlp

    calls = []

    class Unavailable:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download):
            calls.append(1)
            raise yt_dlp.utils.DownloadError("Private video")

    monkeypatch.setattr(yt_dlp, "YoutubeDL", Unavailable)
    with pytest.raises(ap.AudioProcessingError, match="Private video"):
        ap.download_youtube_audio("https://youtu.be/abc123", tmp_path)
    assert calls == [1]
