"""AI Meeting Assistant — Streamlit frontend.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import html
import os
import re
import shutil
import uuid
from pathlib import Path

import streamlit as st

from core.config import load_environment, work_dir

load_environment()  # must happen before modules that read configuration are used

st.set_page_config(
    page_title="AI Meeting Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from core.extractor import count_items  # noqa: E402
from core.llm import LLMError, get_model_name, is_llm_configured  # noqa: E402
from core.meeting_store import MeetingRecord, list_meetings, load_meeting  # noqa: E402
from core.pipeline import (  # noqa: E402
    STAGE_LABELS,
    STAGES,
    PipelineError,
    ProgressEvent,
    analyse_transcript,
    run_pipeline,
)
from core.rag_engine import ask_question, load_rag_chain  # noqa: E402
from core.transcriber import LANGUAGE_ENGINES, get_whisper_model_name  # noqa: E402
from core.vector_store import warm_up_embeddings  # noqa: E402
from utils.audio_processor import (  # noqa: E402
    SUPPORTED_EXTENSIONS,
    AudioProcessingError,
    cleanup_work_dir,
    validate_youtube_url,
)
from utils.exporter import (  # noqa: E402
    build_pdf_report,
    format_date,
    safe_filename,
    summary_to_txt,
    transcript_to_txt,
)

# ── Styling ─────────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1200px;}
.hero {padding: 2.1rem 2.3rem; border: 1px solid #252C3B; border-radius: 16px; margin-bottom: 1.3rem;
       background: radial-gradient(120% 150% at 0% 0%, rgba(124,131,255,.17), transparent 55%), #10141D;}
.eyebrow {display: inline-flex; align-items: center; gap: .4rem; font-size: .72rem; font-weight: 600;
          letter-spacing: .07em; text-transform: uppercase; color: #B3B8FF; background: rgba(124,131,255,.12);
          border: 1px solid rgba(124,131,255,.3); padding: .25rem .7rem; border-radius: 999px;}
.hero h1 {font-size: 2.25rem; line-height: 1.15; margin: .85rem 0 .45rem; padding: 0;}
.hero p {color: #A3AABB; font-size: 1.02rem; max-width: 740px; margin: 0;}
.meeting-head {padding: 1.5rem 1.8rem; border: 1px solid #252C3B; border-radius: 16px; margin-bottom: 1rem;
               background: radial-gradient(120% 160% at 0% 0%, rgba(79,209,197,.11), transparent 55%), #10141D;}
.meeting-head h1 {font-size: 1.75rem; line-height: 1.2; margin: .6rem 0 0; padding: 0;}
.chips {display: flex; flex-wrap: wrap; gap: .4rem; margin-top: .8rem;}
.chip {font-size: .78rem; color: #B9BFCC; background: #171D29; border: 1px solid #273042;
       padding: .2rem .65rem; border-radius: 999px;}
.feature {border: 1px solid #252C3B; background: #11161F; border-radius: 12px; padding: 1rem 1.1rem; height: 100%;}
.feature .icon {font-size: 1.3rem;}
.feature .name {font-weight: 600; margin: .4rem 0 .2rem;}
.feature .desc {color: #9AA1B2; font-size: .86rem; line-height: 1.5;}
.section-title {font-size: .76rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
                color: #8D94A6; margin: 1.7rem 0 .65rem;}
.stage-list {display: flex; flex-direction: column; gap: .35rem; margin: .1rem 0 .7rem;}
.stage {display: flex; gap: .75rem; align-items: flex-start; padding: .55rem .75rem; border-radius: 10px;
        border: 1px solid transparent;}
.stage.running {background: rgba(124,131,255,.08); border-color: rgba(124,131,255,.32);}
.stage.error {background: rgba(255,92,92,.08); border-color: rgba(255,92,92,.38);}
.stage .dot {width: 1.4rem; height: 1.4rem; border-radius: 50%; display: flex; align-items: center;
             justify-content: center; font-size: .72rem; font-weight: 700; flex-shrink: 0;}
.stage.pending .dot {border: 1px solid #3A4254; color: #6B7385;}
.stage.running .dot {background: #7C83FF; color: #0B0E14; animation: pulse 1.4s ease-in-out infinite;}
.stage.done .dot {background: #2BB673; color: #0B0E14;}
.stage.error .dot {background: #FF5C5C; color: #0B0E14;}
.stage .label {font-weight: 550; font-size: .92rem;}
.stage.pending .label {color: #7D8496;}
.stage .detail {color: #9AA1B2; font-size: .8rem; margin-top: .1rem; word-break: break-word;}
@keyframes pulse {0%, 100% {opacity: 1;} 50% {opacity: .45;}}
.transcript {line-height: 1.8; color: #C9CEDA; font-size: .95rem;}
.transcript p {margin: 0 0 .95rem;}
.transcript mark {background: rgba(255,196,0,.3); color: inherit; padding: 0 .1rem; border-radius: 3px;}
.brand {display: flex; gap: .65rem; align-items: center; margin: .2rem 0 .9rem;}
.brand .logo {width: 2.2rem; height: 2.2rem; border-radius: 10px; display: flex; align-items: center;
              justify-content: center; font-size: 1.1rem; background: linear-gradient(135deg, #7C83FF, #4FD1C5);}
.brand .name {font-weight: 700; font-size: 1.02rem; line-height: 1.15;}
.brand .tag {font-size: .74rem; color: #8D94A6;}
.status-row {display: flex; justify-content: space-between; gap: .5rem; font-size: .82rem; padding: .16rem 0; color: #AEB4C2;}
.ok {color: #2BB673;} .bad {color: #FF6B6B;} .muted {color: #7D8496;}
[data-testid="stSidebar"] .stButton button {justify-content: flex-start; text-align: left;}
[data-testid="stSidebar"] .stButton button p {white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
</style>
""",
    unsafe_allow_html=True,
)

TAB_NAMES = [
    "Overview",
    "Transcript",
    "Summary",
    "Action Items",
    "Key Decisions",
    "Open Questions",
    "Ask Your Meeting",
]
SCOPE_THIS, SCOPE_ALL = "This meeting", "All meetings"
SUGGESTED_QUESTIONS = [
    "What were the main topics discussed?",
    "What decisions were taken?",
    "Who is responsible for each action item?",
    "What deadlines were mentioned?",
    "What problems or risks were discussed?",
]
LANGUAGE_CAPTIONS = {
    "english": "Transcribed locally with OpenAI Whisper",
    "hinglish": "Transcribed and translated to English by Sarvam AI",
}
UPLOAD_TYPES = sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS)


# ── Helpers ─────────────────────────────────────────────────────────────────────


def esc(value: object) -> str:
    """HTML-escape any user/LLM text before it is placed inside custom HTML."""
    return html.escape(str(value or ""), quote=True).replace("$", "&#36;")


_MD_SPECIAL = re.compile(r"([\\`*_\[\]#<>~$|])")


def escape_md(value: object) -> str:
    """Escape text so st.markdown shows it literally."""
    return _MD_SPECIAL.sub(r"\\\1", str(value or ""))


def format_duration(seconds: float | None) -> str:
    if not seconds:
        return "—"
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m {secs:02d}s"


def engine_label(language: str) -> str:
    return f"{(language or 'english').title()} · {LANGUAGE_ENGINES.get(language, '—')}"


def split_items(markdown_list: str) -> list[str]:
    """Split a numbered Markdown list into its top-level items."""
    items, current = [], []
    for line in (markdown_list or "").splitlines():
        if re.match(r"^\d+\.\s", line):
            if current:
                items.append("\n".join(current))
            current = [line]
        elif current:
            current.append(line)
    if current:
        items.append("\n".join(current))
    return items


def summary_section(summary: str, heading: str) -> str:
    match = re.search(
        rf"^#+\s*{re.escape(heading)}\s*$(.*?)(?=^#+\s|\Z)", summary or "", flags=re.M | re.S | re.I
    )
    return match.group(1).strip() if match else ""


def sarvam_configured() -> bool:
    return bool((os.getenv("SARVAM_API_KEY") or "").strip())


# ── Session state ───────────────────────────────────────────────────────────────


def init_state() -> None:
    defaults = {
        "view": "home",  # "home" | "meeting"
        "meeting_id": None,
        "chats": {},  # "<meeting_id>|<scope>" -> list of messages
        "job": None,  # pending processing job (runs on the next script run)
        "failure": None,  # last pipeline failure (with transcript, when available)
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def open_meeting(meeting_id: str) -> None:
    st.session_state.view = "meeting"
    st.session_state.meeting_id = meeting_id
    st.session_state.failure = None
    st.session_state["meeting_tab"] = "Overview"


def go_home() -> None:
    st.session_state.view = "home"
    st.session_state.meeting_id = None


@st.cache_resource(show_spinner=False)
def start_background_warmup() -> bool:
    """Once per server process: preload the embedding model so indexing/chat start fast."""
    warm_up_embeddings()
    return True


@st.cache_resource(show_spinner=False)
def get_rag_chain(meeting_id: str | None):
    """RAG chains hold no per-user state (history is passed per call), so they are shared."""
    return load_rag_chain(meeting_id=meeting_id)


@st.cache_data(show_spinner=False, max_entries=20)
def pdf_report_bytes(meeting_id: str, processed_at: str) -> bytes:
    record = load_meeting(meeting_id)
    return build_pdf_report(record) if record else b""


# ── Progress UI (driven by real pipeline events) ────────────────────────────────


class ProgressTracker:
    def __init__(self, status_container, completed_stages: tuple[str, ...] = (), note: str = ""):
        self.status = status_container
        self.state = {stage.key: {"status": "pending", "message": ""} for stage in STAGES}
        self.fraction: dict[str, float] = {}
        for key in completed_stages:
            self.state[key] = {"status": "done", "message": note}
        with self.status:
            self.list_placeholder = st.empty()
            self.bar = st.progress(0.0)
        self.render()

    def __call__(self, event: ProgressEvent) -> None:
        entry = self.state.setdefault(event.stage, {"status": "pending", "message": ""})
        entry["status"] = event.status
        if event.message:
            entry["message"] = event.message
        if event.fraction is not None:
            self.fraction[event.stage] = event.fraction
        if event.status == "running":
            self.status.update(
                label=f"{STAGE_LABELS.get(event.stage, event.stage)}…", state="running", expanded=True
            )
        self.render()

    def overall(self) -> float:
        done = sum(1 for entry in self.state.values() if entry["status"] == "done")
        running = sum(
            self.fraction.get(key, 0.0)
            for key, entry in self.state.items()
            if entry["status"] == "running"
        )
        return min((done + running) / len(STAGES), 1.0)

    def render(self) -> None:
        icons = {"pending": "", "running": "•", "done": "✓", "error": "✕"}
        rows = []
        for number, stage in enumerate(STAGES, start=1):
            entry = self.state[stage.key]
            status = entry["status"]
            dot = icons.get(status) or str(number)
            detail = f'<div class="detail">{esc(entry["message"])}</div>' if entry["message"] else ""
            rows.append(
                f'<div class="stage {status}"><div class="dot">{dot}</div>'
                f'<div><div class="label">{esc(stage.label)}</div>{detail}</div></div>'
            )
        self.list_placeholder.markdown(
            f'<div class="stage-list">{"".join(rows)}</div>', unsafe_allow_html=True
        )
        done = sum(1 for entry in self.state.values() if entry["status"] == "done")
        self.bar.progress(self.overall(), text=f"{done} of {len(STAGES)} stages complete")


# ── Sidebar ─────────────────────────────────────────────────────────────────────


def render_sidebar(meetings: list[MeetingRecord], busy: bool) -> None:
    with st.sidebar:
        st.markdown(
            '<div class="brand"><div class="logo">🎙️</div><div><div class="name">AI Meeting Assistant</div>'
            f'<div class="tag">Powered by {esc(get_model_name())}</div></div></div>',
            unsafe_allow_html=True,
        )
        st.button("＋  New meeting", width="stretch", type="primary", on_click=go_home, disabled=busy)

        st.markdown('<div class="section-title">Meeting history</div>', unsafe_allow_html=True)
        if not meetings:
            st.caption("Processed meetings will appear here.")
        for record in meetings[:30]:
            active = st.session_state.view == "meeting" and st.session_state.meeting_id == record.meeting_id
            st.button(
                record.title,
                key=f"history_{record.meeting_id}",
                help=f"{format_date(record.processed_at)} · {engine_label(record.language)}",
                width="stretch",
                type="secondary" if not active else "tertiary",
                icon=":material/description:",
                on_click=open_meeting,
                args=(record.meeting_id,),
                disabled=busy,
            )

        st.markdown('<div class="section-title">System status</div>', unsafe_allow_html=True)
        rows = [
            ("Gemini API key", is_llm_configured(), "configured", "missing"),
            ("Sarvam API key", sarvam_configured(), "configured", "missing"),
            ("FFmpeg", bool(shutil.which("ffmpeg")), "found", "not found"),
        ]
        html_rows = "".join(
            f'<div class="status-row"><span>{name}</span>'
            f'<span class="{"ok" if ok else "bad"}">{"✓ " + good if ok else "✕ " + bad}</span></div>'
            for name, ok, good, bad in rows
        )
        html_rows += (
            f'<div class="status-row"><span>LLM</span><span class="muted">{esc(get_model_name())}</span></div>'
            f'<div class="status-row"><span>Whisper model</span><span class="muted">{esc(get_whisper_model_name())}</span></div>'
        )
        st.markdown(html_rows, unsafe_allow_html=True)


# ── Home ────────────────────────────────────────────────────────────────────────


def queue_job_from_form(source_mode: str, url: str, upload, language: str, force: bool) -> str | None:
    """Validate inputs and queue a processing job. Returns an error message or None."""
    if not is_llm_configured():
        return "Gemini API key not found. Add GOOGLE_API_KEY to the .env file and restart the app."
    if language == "hinglish" and not sarvam_configured():
        return "SARVAM_API_KEY is not set in .env — it is required for Hinglish transcription."
    if not shutil.which("ffmpeg"):
        return "FFmpeg was not found on PATH. Install it and restart the app (see README)."

    if source_mode == "YouTube link":
        try:
            source = validate_youtube_url(url)
        except AudioProcessingError as exc:
            return str(exc)
        job = {"kind": "full", "source": source, "label": source, "upload_dir": None}
    else:
        if upload is None:
            return "Please upload an audio or video file."
        suffix = Path(upload.name).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            return f"Unsupported file type '{suffix}'."
        upload_dir = work_dir() / f"upload_{uuid.uuid4().hex[:12]}"
        upload_dir.mkdir(parents=True, exist_ok=True)
        saved = upload_dir / f"input{suffix}"
        with open(saved, "wb") as handle:
            shutil.copyfileobj(upload, handle)
        job = {"kind": "full", "source": str(saved), "label": upload.name, "upload_dir": str(upload_dir)}

    job.update(language=language, force=force)
    st.session_state.job = job
    st.session_state.failure = None
    return None


def render_hero() -> None:
    st.markdown(
        f"""
<div class="hero">
  <span class="eyebrow">✦ {esc(get_model_name())} · Whisper · Sarvam AI</span>
  <h1>Turn every meeting into clear decisions</h1>
  <p>Drop in a recording or a YouTube link. Get a faithful transcript, a structured summary,
  action items, decisions and open questions — then ask follow-up questions grounded in what was actually said.</p>
</div>""",
        unsafe_allow_html=True,
    )


def render_input_card(busy: bool) -> None:
    with st.container(border=True):
        st.markdown("#### New meeting")
        source_mode = st.segmented_control(
            "Source",
            ["YouTube link", "Upload file"],
            default="YouTube link",
            key="source_mode",
            disabled=busy,
            required=True,
        )
        url, upload = "", None
        if source_mode == "YouTube link":
            url = st.text_input(
                "YouTube URL",
                placeholder="https://www.youtube.com/watch?v=…  or  https://youtu.be/…",
                key="youtube_url",
                disabled=busy,
            )
        else:
            upload = st.file_uploader(
                "Audio or video file",
                type=UPLOAD_TYPES,
                key="upload_file",
                disabled=busy,
                help="Audio: mp3, wav, m4a, flac, ogg… · Video: mp4, mkv, mov, webm…",
            )

        language = st.radio(
            "Spoken language",
            list(LANGUAGE_ENGINES),
            format_func=lambda lang: f"{lang.title()} — {LANGUAGE_ENGINES[lang]}",
            captions=[LANGUAGE_CAPTIONS[lang] for lang in LANGUAGE_ENGINES],
            horizontal=True,
            key="language",
            disabled=busy,
        )
        with st.expander("Advanced options"):
            force = st.checkbox(
                "Re-run Gemini analysis even if this meeting was analysed before",
                key="force_reanalyse",
                disabled=busy,
                help="By default a previously analysed meeting reuses its saved results (no extra Gemini calls).",
            )

        if st.button("Analyse meeting", type="primary", icon=":material/auto_awesome:", disabled=busy):
            error = queue_job_from_form(source_mode, url, upload, language, force)
            if error:
                st.error(error, icon=":material/error:")
            else:
                st.rerun()


def render_features() -> None:
    features = [
        ("🎧", "Any source", "YouTube links or local audio and video files, converted to clean 16 kHz audio."),
        ("🗣️", "English & Hinglish", "Whisper runs locally for English; Sarvam AI handles Hinglish speech."),
        ("✨", "Gemini analysis", "Map-Reduce summaries plus action items, decisions and open questions."),
        ("💬", "Ask your meetings", "Grounded answers with sources from one meeting or your whole history."),
    ]
    columns = st.columns(len(features))
    for column, (icon, name, description) in zip(columns, features):
        column.markdown(
            f'<div class="feature"><div class="icon">{icon}</div><div class="name">{name}</div>'
            f'<div class="desc">{description}</div></div>',
            unsafe_allow_html=True,
        )


def render_recent(meetings: list[MeetingRecord], busy: bool) -> None:
    st.markdown('<div class="section-title">Recent meetings</div>', unsafe_allow_html=True)
    if not meetings:
        st.caption("No meetings yet — your processed meetings will be listed here.")
        return
    columns = st.columns(3)
    for index, record in enumerate(meetings[:6]):
        with columns[index % 3].container(border=True):
            st.markdown(f"**{escape_md(record.title)}**")
            st.caption(
                f"{format_date(record.processed_at)} · {format_duration(record.duration_seconds)} · "
                f"{record.language.title()} · {count_items(record.action_items)} action items"
            )
            st.button(
                "Open",
                key=f"recent_{record.meeting_id}",
                icon=":material/arrow_forward:",
                on_click=open_meeting,
                args=(record.meeting_id,),
                disabled=busy,
            )


def render_failure(busy: bool) -> None:
    failure = st.session_state.failure
    if not failure or busy:
        return
    st.error(f"**Processing failed during: {failure['stage']}**\n\n{failure['message']}", icon=":material/error:")
    if failure.get("transcript"):
        st.info(
            "Transcription finished successfully, so you can retry the Gemini analysis without "
            "transcribing the audio again.",
            icon=":material/info:",
        )
        col_retry, col_download = st.columns(2)
        if col_retry.button("Retry analysis", type="primary", icon=":material/refresh:", width="stretch"):
            st.session_state.job = {
                "kind": "retry",
                "transcript": failure["transcript"],
                "label": failure["source"],
                "language": failure["language"],
                "duration": failure.get("duration"),
                "force": False,
                "upload_dir": None,
            }
            st.session_state.failure = None
            st.rerun()
        col_download.download_button(
            "Download transcript",
            data=failure["transcript"],
            file_name="transcript.txt",
            mime="text/plain",
            icon=":material/download:",
            width="stretch",
        )


def run_job(container) -> None:
    """Execute the queued job, streaming real progress events into the UI."""
    job = st.session_state.job
    with container:
        status = st.status("Starting…", expanded=True)
        if job["kind"] == "retry":
            tracker = ProgressTracker(
                status, ("input", "audio", "transcription"), "Reused the existing transcript"
            )
        else:
            tracker = ProgressTracker(status)
    try:
        if job["kind"] == "retry":
            record = analyse_transcript(
                job["transcript"],
                source=job["label"],
                language=job["language"],
                duration_seconds=job.get("duration"),
                progress=tracker,
                force=job["force"],
            )
        else:
            record = run_pipeline(
                job["source"], job["language"], progress=tracker, source_label=job["label"], force=job["force"]
            )
    except PipelineError as exc:
        status.update(label=f"Failed: {exc.stage_label}", state="error", expanded=True)
        st.session_state.failure = {
            "stage": exc.stage_label,
            "message": str(exc),
            "transcript": exc.transcript or job.get("transcript"),
            "source": job["label"],
            "language": job["language"],
            "duration": exc.duration_seconds or job.get("duration"),
        }
        st.session_state.job = None
    except Exception as exc:  # last-resort boundary: never show a success state after a failure
        status.update(label="Failed: unexpected error", state="error", expanded=True)
        st.session_state.failure = {
            "stage": "Processing",
            "message": f"{type(exc).__name__}: {exc}",
            "transcript": job.get("transcript"),
            "source": job["label"],
            "language": job["language"],
        }
        st.session_state.job = None
    else:
        status.update(label="Meeting processed successfully", state="complete", expanded=False)
        st.session_state.job = None
        open_meeting(record.meeting_id)
    finally:
        cleanup_work_dir(job.get("upload_dir"))
    st.rerun()


def render_home(meetings: list[MeetingRecord], busy: bool) -> None:
    render_hero()
    left, right = st.columns([3, 2], gap="large")
    with left:
        render_input_card(busy)
        progress_area = st.container()
        with progress_area:
            render_failure(busy)
    with right:
        with st.container(border=True):
            st.markdown("#### How it works")
            for number, stage in enumerate(STAGES, start=1):
                st.markdown(f"**{number}.** {stage.label}")
            st.caption("Progress for each stage is reported live while your meeting is processed.")
    st.markdown('<div class="section-title">What you get</div>', unsafe_allow_html=True)
    render_features()
    render_recent(meetings, busy)

    if busy:
        run_job(progress_area)


# ── Meeting view ────────────────────────────────────────────────────────────────


def render_meeting_header(record: MeetingRecord) -> None:
    chips = [
        f"📅 {format_date(record.processed_at)}",
        f"⏱ {format_duration(record.duration_seconds)}",
        f"🗣 {engine_label(record.language)}",
        f"✦ {record.model or get_model_name()}",
    ]
    source = record.source or ""
    if source:
        chips.append(f"🔗 {source if len(source) <= 60 else source[:57] + '…'}")
    chips_html = "".join(f'<span class="chip">{esc(chip)}</span>' for chip in chips)
    st.markdown(
        f'<div class="meeting-head"><span class="eyebrow">Meeting report</span>'
        f"<h1>{esc(record.title)}</h1><div class=\"chips\">{chips_html}</div></div>",
        unsafe_allow_html=True,
    )


def render_metrics_and_exports(record: MeetingRecord) -> None:
    metrics = st.columns(5)
    metrics[0].metric("Duration", format_duration(record.duration_seconds), border=True)
    metrics[1].metric("Words", f"{len(record.transcript.split()):,}", border=True)
    metrics[2].metric("Action items", count_items(record.action_items), border=True)
    metrics[3].metric("Decisions", count_items(record.key_decisions), border=True)
    metrics[4].metric("Open questions", count_items(record.open_questions), border=True)

    exports = st.columns([1, 1, 1, 2])
    exports[0].download_button(
        "Transcript (TXT)",
        data=transcript_to_txt(record),
        file_name=safe_filename(record.title, "transcript.txt"),
        mime="text/plain",
        icon=":material/description:",
        width="stretch",
    )
    exports[1].download_button(
        "Summary (TXT)",
        data=summary_to_txt(record),
        file_name=safe_filename(record.title, "summary.txt"),
        mime="text/plain",
        icon=":material/summarize:",
        width="stretch",
    )
    exports[2].download_button(
        "Report (PDF)",
        data=lambda: pdf_report_bytes(record.meeting_id, record.processed_at),
        file_name=safe_filename(record.title, "report.pdf"),
        mime="application/pdf",
        icon=":material/picture_as_pdf:",
        width="stretch",
    )


def render_preview(column, title: str, content: str, empty_text: str) -> None:
    items = split_items(content)
    with column.container(border=True, height="stretch"):
        st.markdown(f"**{title}** · {len(items)}")
        if not items:
            st.caption(empty_text)
        for item in items[:3]:
            first_line = re.sub(r"^\d+\.\s*", "", item.splitlines()[0])
            st.markdown(f"- {first_line}")
        if len(items) > 3:
            st.caption(f"+{len(items) - 3} more in the {title} tab")


def render_overview(record: MeetingRecord) -> None:
    left, right = st.columns([3, 2], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("#### At a glance")
            overview = summary_section(record.summary, "Overview")
            st.markdown(overview or record.summary.split("\n\n")[0])
            topics = summary_section(record.summary, "Main Topics Discussed")
            if topics:
                st.markdown("**Main topics**")
                names = re.findall(r"^\s*[-*]\s*\*\*(.+?)\*\*", topics, flags=re.M)
                st.markdown("\n".join(f"- {name}" for name in names) if names else topics)
    with right:
        with st.container(border=True):
            st.markdown("#### Details")
            details = [
                ("Processed", format_date(record.processed_at)),
                ("Duration", format_duration(record.duration_seconds)),
                ("Language", engine_label(record.language)),
                ("Model", record.model or get_model_name()),
                ("Source", record.source or "—"),
                ("Meeting ID", record.meeting_id),
            ]
            for label, value in details:
                st.markdown(f"**{label}:** {escape_md(value)}")
    c1, c2, c3 = st.columns(3)
    render_preview(c1, "Action Items", record.action_items, "No action items found.")
    render_preview(c2, "Key Decisions", record.key_decisions, "No key decisions found.")
    render_preview(c3, "Open Questions", record.open_questions, "No open questions found.")


def transcript_html(transcript: str, query: str = "") -> tuple[str, int]:
    sentences = re.split(r"(?<=[.!?])\s+", transcript.strip())
    paragraphs = [" ".join(sentences[i : i + 5]) for i in range(0, len(sentences), 5)]
    matches = 0
    rendered = []
    for paragraph in paragraphs:
        safe = esc(paragraph)
        if query:
            pattern = re.compile(re.escape(esc(query)), re.I)
            safe, found = pattern.subn(lambda m: f"<mark>{m.group(0)}</mark>", safe)
            matches += found
        rendered.append(f"<p>{safe}</p>")
    return f'<div class="transcript">{"".join(rendered)}</div>', matches


def render_transcript(record: MeetingRecord) -> None:
    query = st.text_input(
        "Find in transcript", placeholder="Search for a word or phrase…", key=f"find_{record.meeting_id}"
    ).strip()
    body, matches = transcript_html(record.transcript, query)
    if query:
        st.caption(f"{matches} match{'es' if matches != 1 else ''} for “{query}”")
    with st.container(border=True, height=560):
        st.markdown(body, unsafe_allow_html=True)


def render_markdown_card(content: str) -> None:
    with st.container(border=True):
        st.markdown(content)


def render_chat(record: MeetingRecord) -> None:
    top_left, top_right = st.columns([3, 1])
    with top_left:
        scope = st.segmented_control(
            "Search scope",
            [SCOPE_THIS, SCOPE_ALL],
            default=SCOPE_THIS,
            key="chat_scope",
            required=True,
            help="Search only this meeting, or every meeting stored in the vector database.",
        )
    chat_key = f"{record.meeting_id}|{scope}"
    history: list[dict] = st.session_state.chats.setdefault(chat_key, [])
    with top_right:
        st.write("")
        if st.button("Clear chat", icon=":material/delete_sweep:", disabled=not history, width="stretch"):
            st.session_state.chats[chat_key] = []
            st.rerun()

    picked = st.pills(
        "Suggested questions",
        SUGGESTED_QUESTIONS,
        key=f"pill_{chat_key}_{len(history)}",
        label_visibility="collapsed",
    )

    box = st.container(height=470, border=True)
    with box:
        if not history:
            st.caption(
                "Ask anything about this meeting. Answers are generated by Gemini only from the "
                "retrieved transcript excerpts, with sources."
            )
        for message in history:
            render_chat_message(message)

    prompt = st.chat_input(
        "Ask about this meeting…" if scope == SCOPE_THIS else "Ask about any of your meetings…",
        key=f"chat_input_{record.meeting_id}",
    )
    question = (prompt or picked or "").strip()
    if not question:
        return

    with box:
        render_chat_message({"role": "user", "content": question})
        with st.chat_message("assistant", avatar=":material/smart_toy:"):
            with st.spinner("Searching the transcript and asking Gemini…"):
                try:
                    chain = get_rag_chain(record.meeting_id if scope == SCOPE_THIS else None)
                    result = ask_question(chain, question, history)
                    reply = {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
                except (LLMError, ValueError) as exc:
                    reply = {"role": "assistant", "content": f"⚠️ {exc}", "sources": [], "error": True}
    history.append({"role": "user", "content": question})
    history.append(reply)
    st.rerun()


def render_chat_message(message: dict) -> None:
    is_user = message["role"] == "user"
    avatar = ":material/person:" if is_user else ":material/smart_toy:"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(escape_md(message["content"]) if is_user else message["content"])
        sources = message.get("sources") or []
        if sources:
            with st.expander(f"Sources ({len(sources)})"):
                for source in sources:
                    segment = source.get("chunk_index")
                    where = f" · segment {int(segment) + 1}" if segment is not None else ""
                    st.markdown(f"**[Source {source['number']}]** {escape_md(source['title'])}{where}")
                    st.caption(escape_md(source["excerpt"]))


def render_meeting(record: MeetingRecord) -> None:
    render_meeting_header(record)
    render_metrics_and_exports(record)
    st.write("")
    tabs = st.tabs(TAB_NAMES, key="meeting_tab", on_change="rerun")
    renderers = {
        "Overview": lambda: render_overview(record),
        "Transcript": lambda: render_transcript(record),
        "Summary": lambda: render_markdown_card(record.summary),
        "Action Items": lambda: render_markdown_card(record.action_items),
        "Key Decisions": lambda: render_markdown_card(record.key_decisions),
        "Open Questions": lambda: render_markdown_card(record.open_questions),
        "Ask Your Meeting": lambda: render_chat(record),
    }
    for name, tab in zip(TAB_NAMES, tabs):
        if tab.open:
            with tab:
                renderers[name]()


# ── Main ────────────────────────────────────────────────────────────────────────


def main() -> None:
    init_state()
    start_background_warmup()
    busy = st.session_state.job is not None
    meetings = list_meetings()
    render_sidebar(meetings, busy)

    if st.session_state.view == "meeting" and st.session_state.meeting_id and not busy:
        record = load_meeting(st.session_state.meeting_id)
        if record is None:
            st.warning("That meeting could not be found. It may have been removed.")
            go_home()
        else:
            render_meeting(record)
            return
    render_home(meetings, busy)


main()
