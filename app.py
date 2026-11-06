"""AI Meeting Assistant — Streamlit frontend.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import html
import os
import re
import shutil
import uuid
from datetime import date
from pathlib import Path

import streamlit as st

from core.config import load_environment, reload_environment, work_dir

load_environment()  # must happen before modules that read configuration are used

st.set_page_config(
    page_title="AI Meeting Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from core.extractor import count_items  # noqa: E402
from core.llm import LLMError, check_openrouter, get_model_name, get_provider_label  # noqa: E402
from core.meeting_store import MeetingRecord, list_meetings, load_chats, load_meeting, save_chats  # noqa: E402
from core.pipeline import (  # noqa: E402
    STAGE_LABELS,
    STAGES,
    PipelineError,
    ProgressEvent,
    analyse_transcript,
    run_pipeline,
)
from core.rag_engine import format_timestamp, stream_answer  # noqa: E402
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
    chat_to_txt,
    export_bundle,
    format_date,
    safe_filename,
    section_to_txt,
    timestamped_paragraphs,
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

/* ── v2 polish ── */
.hero h1 {background: linear-gradient(90deg, #FFFFFF 0%, #C9CCFF 55%, #7FE3D7 100%); -webkit-background-clip: text;
          background-clip: text; -webkit-text-fill-color: transparent;}
.hero-pills {display: flex; flex-wrap: wrap; gap: .45rem; margin-top: 1.1rem;}
.hero-pills span {font-size: .8rem; color: #C3C8D6; background: rgba(255,255,255,.04); border: 1px solid #2A3244;
                  padding: .3rem .7rem; border-radius: 999px;}
.stepper {border: 1px solid #252C3B; background: #11161F; border-radius: 14px; padding: 1.1rem 1.2rem;}
.stepper-title {font-weight: 650; font-size: 1.05rem; margin-bottom: .7rem;}
.step {display: flex; gap: .7rem; align-items: center; padding: .42rem 0; color: #C9CEDA; font-size: .9rem;}
.step .num {width: 1.6rem; height: 1.6rem; border-radius: 8px; flex-shrink: 0; display: flex; align-items: center;
            justify-content: center; font-weight: 700; font-size: .78rem; color: #0B0E14;
            background: linear-gradient(135deg, #7C83FF, #4FD1C5);}
.stepper-foot {color: #7D8496; font-size: .78rem; margin-top: .6rem;}
.feature {transition: border-color .15s, transform .15s;}
.feature:hover {border-color: #3B4560; transform: translateY(-2px);}
[data-testid="stMetric"] {background: #11161F;}
/* tabs as pills */
[data-testid="stTabs"] [role="tablist"] {gap: .4rem; flex-wrap: wrap; padding-bottom: .5rem;}
[data-testid="stTab"] {background: #121722; border: 1px solid #222938 !important; border-radius: 999px;
                       padding: .32rem 1rem !important; height: auto; transition: border-color .15s, background .15s;}
[data-testid="stTab"]:hover {border-color: #3B4560 !important;}
[data-testid="stTab"][aria-selected="true"] {background: rgba(124,131,255,.18); border-color: rgba(124,131,255,.6) !important;}
[data-testid="stTab"][aria-selected="true"] p {color: #DADCFF; font-weight: 600;}
/* transcript with timestamps */
.tline {display: flex; gap: .85rem; align-items: flex-start; padding: .35rem 0; border-bottom: 1px dashed #1E2533;}
.tline:last-child {border-bottom: none;}
.tline p {margin: 0;}
.ts {flex-shrink: 0; font-family: ui-monospace, Consolas, monospace; font-size: .76rem; color: #A9AEFF;
     background: rgba(124,131,255,.1); border: 1px solid rgba(124,131,255,.28); border-radius: 6px;
     padding: .08rem .45rem; margin-top: .25rem;}
/* chat */
[data-testid="stChatMessage"] {border-radius: 14px; padding: .75rem .9rem; margin-bottom: .5rem;
                               border: 1px solid #1F2633; background: #10151E;}
[data-testid="stChatMessage"]:has(.u-msg) {background: rgba(124,131,255,.09); border-color: rgba(124,131,255,.3);
                                           width: 86%; margin-left: auto; box-sizing: border-box;}
.u-msg {display: none;}
.srcs {display: flex; flex-wrap: wrap; gap: .35rem; margin: .45rem 0 .1rem;}
.src {font-size: .74rem; color: #8FE3D9; background: rgba(79,209,197,.08); border: 1px solid rgba(79,209,197,.28);
      border-radius: 999px; padding: .1rem .55rem;}
.chat-empty {text-align: center; padding: 3.2rem 1.5rem; color: #C9CEDA; line-height: 1.6;}
.chat-empty .big {font-size: 2.4rem; margin-bottom: .4rem;}
.chat-empty .muted {max-width: 520px; margin: .4rem auto 0; font-size: .9rem;}
</style>
""",
    unsafe_allow_html=True,
)

CHAT_TAB = "Ask Your Meeting"
TAB_NAMES = [
    "Overview",
    CHAT_TAB,
    "Transcript",
    "Summary",
    "Action Items",
    "Key Decisions",
    "Open Questions",
    "Downloads",
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


@st.cache_data(ttl=120, show_spinner=False)
def llm_status() -> dict:
    """OpenRouter key / model / free-quota status (cached; uses no model requests)."""
    status = check_openrouter()
    return {
        "key_configured": status.key_configured,
        "key_valid": status.key_valid,
        "model_available": status.model_available,
        "daily_used": status.daily_used,
        "daily_limit": status.daily_limit,
        "daily_remaining": status.daily_remaining,
        "ready": status.ready,
        "error": status.error,
    }


def sarvam_configured() -> bool:
    return bool((os.getenv("SARVAM_API_KEY") or "").strip())


# ── Session state ───────────────────────────────────────────────────────────────


def init_state() -> None:
    defaults = {
        "view": "home",  # "home" | "meeting"
        "meeting_id": None,
        "chats": {},  # meeting_id -> {scope: [messages]} (also saved to disk)
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
            f'<div class="tag">OpenRouter · {esc(get_model_name())}</div></div></div>',
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

        status_box = st.expander("System status", icon=":material/monitor_heart:")
        llm = llm_status()
        rows = [
            ("OpenRouter API key", llm["key_configured"] and llm["key_valid"],
             "valid", "missing" if not llm["key_configured"] else "not verified"),
            ("LLM model", llm["model_available"], "available", "unavailable"),
            ("Sarvam API key", sarvam_configured(), "configured", "missing"),
            ("FFmpeg", bool(shutil.which("ffmpeg")), "found", "not found"),
        ]
        html_rows = "".join(
            f'<div class="status-row"><span>{name}</span>'
            f'<span class="{"ok" if ok else "bad"}">{"✓ " + good if ok else "✕ " + bad}</span></div>'
            for name, ok, good, bad in rows
        )
        if not get_model_name().endswith(":free"):
            quota = "n/a (paid model)"
        elif llm["daily_remaining"] is not None and llm["daily_limit"] is not None:
            quota = f"{llm['daily_remaining']} of {llm['daily_limit']} left"
        else:
            quota = "—"
        html_rows = (
            f'<div class="status-row"><span>Provider</span><span class="muted">{esc(get_provider_label())}</span></div>'
            f'<div class="status-row"><span>Model</span><span class="muted">{esc(get_model_name())}</span></div>'
            + html_rows
            + f'<div class="status-row"><span>Free requests today</span><span class="muted">{esc(quota)}</span></div>'
            f'<div class="status-row"><span>Whisper model</span><span class="muted">{esc(get_whisper_model_name())}</span></div>'
            '<div class="status-row"><span>Legacy LLMs</span><span class="muted">inactive (Gemini, Mistral preserved)</span></div>'
        )
        if llm["error"]:
            st.warning(llm["error"], icon=":material/warning:")
        status_box.markdown(html_rows, unsafe_allow_html=True)
        if status_box.button("Refresh status", icon=":material/refresh:", width="stretch", disabled=busy,
                     help="Re-reads .env (e.g. a changed OPENROUTER_MODEL) and re-checks OpenRouter."):
            reload_environment()
            llm_status.clear()
            st.rerun()


# ── Home ────────────────────────────────────────────────────────────────────────


def queue_job_from_form(
    source_mode: str, url: str, upload, language: str, force: bool, meeting_date: date | None = None
) -> str | None:
    """Validate inputs and queue a processing job. Returns an error message or None."""
    reload_environment()  # pick up .env edits (model, keys) without restarting the app
    llm_status.clear()
    status = llm_status()
    if not status["ready"]:
        return status["error"] or "The local LLM is not ready."
    if language == "hinglish" and not sarvam_configured():
        return "SARVAM_API_KEY is not set in .env — it is required for Hinglish transcription."
    if not shutil.which("ffmpeg"):
        return "FFmpeg was not found on PATH. Install it and restart the app (see README)."

    if source_mode == "YouTube link":
        try:
            source = validate_youtube_url(url)
        except AudioProcessingError as exc:
            return str(exc)
        job = {"kind": "full", "source": source, "label": source, "source_type": "youtube", "upload_dir": None}
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
        job = {
            "kind": "full",
            "source": str(saved),
            "label": upload.name,
            "source_type": "upload",
            "upload_dir": str(upload_dir),
        }

    job.update(language=language, force=force, meeting_date=(meeting_date or date.today()).isoformat())
    st.session_state.job = job
    st.session_state.failure = None
    return None


def render_hero() -> None:
    st.markdown(
        f"""
<div class="hero">
  <span class="eyebrow">✦ {esc(get_model_name())} via OpenRouter · Whisper · Sarvam AI</span>
  <h1>Turn every meeting into clear decisions</h1>
  <p>Drop in a recording or a YouTube link. Get a timestamped transcript, a structured summary,
  action items, decisions and open questions — then chat with the meeting and get answers grounded in what was actually said.</p>
  <div class="hero-pills"><span>🎧 Upload or YouTube</span><span>📝 Timestamped transcript</span>
  <span>✅ Actions &amp; decisions</span><span>💬 Chat with sources</span><span>⬇ PDF · TXT · ZIP</span></div>
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
            meeting_date = st.date_input(
                "Meeting date",
                value=date.today(),
                key="meeting_date",
                disabled=busy,
                help="Used for history and questions like 'what did we decide last week?'.",
            )
            force = st.checkbox(
                "Re-run the AI analysis even if this meeting was analysed before",
                key="force_reanalyse",
                disabled=busy,
                help="By default a previously analysed meeting reuses its saved results (no extra LLM calls). "
                "Re-running also refreshes this meeting's stored transcript chunks.",
            )

        if st.button("Analyse meeting", type="primary", icon=":material/auto_awesome:", disabled=busy):
            error = queue_job_from_form(source_mode, url, upload, language, force, meeting_date)
            if error:
                st.error(error, icon=":material/error:")
            else:
                st.rerun()


def render_features() -> None:
    features = [
        ("🎧", "Any source", "YouTube links or local audio and video files, converted to clean 16 kHz audio."),
        ("🗣️", "English & Hinglish", "Whisper runs locally for English; Sarvam AI handles Hinglish speech."),
        ("✨", "AI analysis", f"{get_model_name()} via OpenRouter: Map-Reduce summaries, actions, decisions, open questions."),
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
            "Transcription finished successfully, so you can retry the AI analysis without "
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
                "segments": failure.get("segments") or [],
                "source_type": failure.get("source_type", ""),
                "meeting_date": failure.get("meeting_date", ""),
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
                source_type=job.get("source_type", ""),
                language=job["language"],
                duration_seconds=job.get("duration"),
                segments=job.get("segments"),
                meeting_date=job.get("meeting_date", ""),
                progress=tracker,
                force=job["force"],
            )
        else:
            record = run_pipeline(
                job["source"],
                job["language"],
                progress=tracker,
                source_label=job["label"],
                source_type=job.get("source_type", ""),
                meeting_date=job.get("meeting_date", ""),
                force=job["force"],
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
            "segments": exc.segments or job.get("segments") or [],
            "source_type": job.get("source_type", ""),
            "meeting_date": job.get("meeting_date", ""),
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
        steps = "".join(
            f'<div class="step"><div class="num">{number}</div><div>{esc(stage.label)}</div></div>'
            for number, stage in enumerate(STAGES, start=1)
        )
        st.markdown(
            f'<div class="stepper"><div class="stepper-title">How it works</div>{steps}'
            '<div class="stepper-foot">Live progress for every stage · then chat with the transcript</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown('<div class="section-title">What you get</div>', unsafe_allow_html=True)
    render_features()
    render_recent(meetings, busy)

    if busy:
        run_job(progress_area)


# ── Meeting view ────────────────────────────────────────────────────────────────


def meeting_chats(meeting_id: str) -> dict[str, list[dict]]:
    """This meeting's conversations (loaded from disk once per session)."""
    chats = st.session_state.chats
    if meeting_id not in chats:
        chats[meeting_id] = load_chats(meeting_id)
    return chats[meeting_id]


def persist_chats(meeting_id: str) -> None:
    try:
        save_chats(meeting_id, meeting_chats(meeting_id))
    except OSError:
        pass  # saving history is best-effort; the conversation stays in the session


def all_messages(meeting_id: str) -> list[dict]:
    """Both scopes' conversations, in order, for exports."""
    chats = meeting_chats(meeting_id)
    return chats.get(SCOPE_THIS, []) + chats.get(SCOPE_ALL, [])


def ask_from_button(question: str) -> None:
    """Callback for suggestion buttons: queue the question and jump to the chat tab."""
    st.session_state["pending_question"] = question
    st.session_state["meeting_tab"] = CHAT_TAB


def render_meeting_header(record: MeetingRecord) -> None:
    chips = [
        f"📅 {record.meeting_date or format_date(record.processed_at)}",
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


def render_metrics_and_actions(record: MeetingRecord) -> None:
    metrics = st.columns(5)
    metrics[0].metric("Duration", format_duration(record.duration_seconds), border=True)
    metrics[1].metric("Words", f"{len(record.transcript.split()):,}", border=True)
    metrics[2].metric("Action items", count_items(record.action_items), border=True)
    metrics[3].metric("Decisions", count_items(record.key_decisions), border=True)
    metrics[4].metric("Open questions", count_items(record.open_questions), border=True)

    actions = st.columns([1.2, 1, 1, 2.2])
    actions[0].button(
        "Ask AI about this meeting", type="primary", icon=":material/forum:", width="stretch",
        on_click=lambda: st.session_state.update(meeting_tab=CHAT_TAB),
    )
    actions[1].download_button(
        "PDF report", data=lambda: build_pdf_report(record, chat=all_messages(record.meeting_id)),
        file_name=safe_filename(record.title, "report.pdf"), mime="application/pdf",
        icon=":material/picture_as_pdf:", width="stretch",
    )
    actions[2].download_button(
        "Everything (ZIP)", data=lambda: export_bundle(record, all_messages(record.meeting_id)),
        file_name=safe_filename(record.title, "meeting.zip"), mime="application/zip",
        icon=":material/folder_zip:", width="stretch",
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


def suggested_questions(record: MeetingRecord) -> list[str]:
    """Generic starters plus questions built from this meeting's own extractions."""
    questions = list(SUGGESTED_QUESTIONS)

    def first_title(markdown_list: str) -> str:
        items = split_items(markdown_list)
        if not items:
            return ""
        title = re.sub(r"^\d+\.\s*", "", items[0].splitlines()[0])
        return re.sub(r"\*\*|—.*$", "", title).strip().rstrip(".")[:80]

    decision = first_title(record.key_decisions)
    action = first_title(record.action_items)
    question = first_title(record.open_questions)
    if decision:
        questions.insert(0, f"Why was this decided: {decision}?")
    if action:
        questions.insert(1, f"Give me more detail on: {action}")
    if question:
        questions.insert(2, f"What was said about: {question}?")
    return questions[:7]


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
                st.markdown("\n".join(f"- {name.rstrip(':')}" for name in names) if names else topics)
    with right:
        with st.container(border=True):
            st.markdown("#### 💬 Ask about this meeting")
            st.caption("Answers come only from the transcript, with timestamped sources.")
            for index, question in enumerate(suggested_questions(record)[:4]):
                st.button(
                    question, key=f"ov_q_{index}", width="stretch", icon=":material/chat_bubble:",
                    on_click=ask_from_button, args=(question,),
                )
    c1, c2, c3 = st.columns(3)
    render_preview(c1, "Action Items", record.action_items, "No action items found.")
    render_preview(c2, "Key Decisions", record.key_decisions, "No key decisions found.")
    render_preview(c3, "Open Questions", record.open_questions, "No open questions found.")
    with st.expander("Meeting details"):
        details = [
            ("Meeting date", record.meeting_date or "—"),
            ("Processed", format_date(record.processed_at)),
            ("Duration", format_duration(record.duration_seconds)),
            ("Language", engine_label(record.language)),
            ("Model", record.model or get_model_name()),
            ("Source", (record.source or "—") + (f" ({record.source_type})" if record.source_type else "")),
            ("Meeting ID", record.meeting_id),
        ]
        for label, value in details:
            st.markdown(f"**{label}:** {escape_md(value)}")


def transcript_html(record: MeetingRecord, query: str = "") -> tuple[str, int]:
    """Timestamped transcript paragraphs as HTML, with search highlighting."""
    matches = 0
    rows = []
    pattern = re.compile(re.escape(esc(query)), re.I) if query else None
    for label, text in timestamped_paragraphs(record):
        safe = esc(text)
        if pattern:
            safe, found = pattern.subn(lambda m: f"<mark>{m.group(0)}</mark>", safe)
            matches += found
            if not found:
                continue  # while searching, show only matching paragraphs
        stamp = f'<span class="ts">{esc(label)}</span>' if label else ""
        rows.append(f'<div class="tline">{stamp}<p>{safe}</p></div>')
    if not rows:
        rows.append('<p class="muted">No matches.</p>')
    return f'<div class="transcript">{"".join(rows)}</div>', matches


def render_transcript(record: MeetingRecord) -> None:
    left, right = st.columns([4, 1])
    query = left.text_input(
        "Find in transcript", placeholder="Search for a word or phrase…", key=f"find_{record.meeting_id}",
        label_visibility="collapsed",
    ).strip()
    right.download_button(
        "Transcript", data=transcript_to_txt(record), file_name=safe_filename(record.title, "transcript.txt"),
        mime="text/plain", icon=":material/download:", width="stretch",
    )
    body, matches = transcript_html(record, query)
    if query:
        st.caption(f"{matches} match{'es' if matches != 1 else ''} for “{query}”")
    elif not record.segments:
        st.caption("Timestamps are not available for this meeting (processed before timestamps were recorded).")
    with st.container(border=True, height=560):
        st.markdown(body, unsafe_allow_html=True)


def render_section(record: MeetingRecord, section: str, content: str) -> None:
    with st.container(border=True):
        st.markdown(content)
    st.download_button(
        "Download this section", data=section_to_txt(record, section),
        file_name=safe_filename(record.title, f"{section}.txt"), mime="text/plain",
        icon=":material/download:", key=f"dl_{section}",
    )


def render_chat_message(message: dict, index: int | None = None, meeting_id: str = "") -> None:
    is_user = message["role"] == "user"
    avatar = ":material/person:" if is_user else ":material/smart_toy:"
    with st.chat_message(message["role"], avatar=avatar):
        if is_user:
            st.markdown('<span class="u-msg"></span>', unsafe_allow_html=True)  # CSS hook for user bubbles
        st.markdown(escape_md(message["content"]) if is_user else message["content"])
        if is_user or message.get("error"):
            return
        sources = message.get("sources") or []
        if sources:
            chips = []
            for source in sources:
                when = (
                    f" · {format_timestamp(source['start_seconds'])}"
                    if source.get("start_seconds") is not None else ""
                )
                chips.append(f'<span class="src">Source {source["number"]}{esc(when)}</span>')
            st.markdown(f'<div class="srcs">{"".join(chips)}</div>', unsafe_allow_html=True)
            with st.expander(f"Show the {len(sources)} transcript excerpt(s) used"):
                for source in sources:
                    details = []
                    if source.get("meeting_date"):
                        details.append(source["meeting_date"])
                    if source.get("start_seconds") is not None:
                        details.append(
                            f"~{format_timestamp(source['start_seconds'])}–{format_timestamp(source.get('end_seconds'))}"
                        )
                    where = f" · {' · '.join(details)}" if details else ""
                    st.markdown(f"**[Source {source['number']}]** {escape_md(source['title'])}{escape_md(where)}")
                    st.caption(escape_md(source["excerpt"]))
        if index is not None:
            tools = st.columns([1, 1, 6])
            with tools[0].popover("", icon=":material/content_copy:", help="Copy answer"):
                st.code(message["content"], language=None, wrap_lines=True)
            with tools[1]:
                rating = st.feedback("thumbs", key=f"fb_{meeting_id}_{index}")
                if rating is not None and message.get("rating") != rating:
                    message["rating"] = rating
                    persist_chats(meeting_id)


def render_chat(record: MeetingRecord) -> None:
    chats = meeting_chats(record.meeting_id)
    main_col, side_col = st.columns([3, 1.15], gap="large")

    with side_col:
        scope = st.segmented_control(
            "Search in", [SCOPE_THIS, SCOPE_ALL], default=SCOPE_THIS, key="chat_scope", required=True,
            help="Only this meeting, or every meeting stored in the vector database.",
        )
        history: list[dict] = chats.setdefault(scope, [])
        st.markdown('<div class="section-title" style="margin-top:.9rem">Try asking</div>', unsafe_allow_html=True)
        for index, question in enumerate(suggested_questions(record)):
            st.button(question, key=f"sq_{index}", width="stretch", on_click=ask_from_button, args=(question,))
        st.markdown('<div class="section-title" style="margin-top:.9rem">Conversation</div>', unsafe_allow_html=True)
        questions_asked = sum(1 for m in history if m["role"] == "user")
        st.caption(f"{questions_asked} question{'s' if questions_asked != 1 else ''} · saved with this meeting")
        st.download_button(
            "Download chat", data=chat_to_txt(record, history), file_name=safe_filename(record.title, "chat.txt"),
            mime="text/plain", icon=":material/download:", width="stretch", disabled=not history,
        )
        if st.button("Clear chat", icon=":material/delete_sweep:", width="stretch", disabled=not history):
            chats[scope] = []
            persist_chats(record.meeting_id)
            st.rerun()

    with main_col:
        box = st.container(height=540, border=True)
        with box:
            if not history:
                st.markdown(
                    '<div class="chat-empty"><div class="big">💬</div>'
                    "<div><b>Chat with your meeting</b></div>"
                    "<div class='muted'>Ask anything about what was said. Answers are grounded in the transcript "
                    "and show the excerpts (with timestamps) they came from. In <i>All meetings</i> you can also ask "
                    "about time periods, e.g. “What did we decide last week?”.</div></div>",
                    unsafe_allow_html=True,
                )
            for index, message in enumerate(history):
                render_chat_message(message, index if message["role"] == "assistant" else None, record.meeting_id)

        prompt = st.chat_input(
            "Ask about this meeting…" if scope == SCOPE_THIS else "Ask about any of your meetings…",
            key=f"chat_input_{record.meeting_id}",
        )
    question = (prompt or st.session_state.pop("pending_question", None) or "").strip()
    if not question:
        return

    with box:
        render_chat_message({"role": "user", "content": question})
        with st.chat_message("assistant", avatar=":material/smart_toy:"):
            reply: dict = {"role": "assistant", "sources": []}
            try:
                with st.spinner("Searching the transcript…"):
                    info, chunks = stream_answer(
                        question,
                        meeting_id=record.meeting_id if scope == SCOPE_THIS else None,
                        chat_history=history,
                        current_meeting_id=record.meeting_id,
                    )
                reply["content"] = st.write_stream(chunks)
                reply["sources"] = info["sources"]
                if isinstance(reply["content"], list):
                    reply["content"] = "".join(str(part) for part in reply["content"])
            except (LLMError, ValueError) as exc:
                reply.update(content=f"⚠️ {exc}", error=True)
    history.append({"role": "user", "content": question})
    history.append(reply)
    persist_chats(record.meeting_id)
    st.rerun()


def render_downloads(record: MeetingRecord) -> None:
    messages = all_messages(record.meeting_id)
    items = [
        ("Full report", "PDF with summary, action items, decisions, open questions, Q&A and the timestamped transcript.",
         lambda: build_pdf_report(record, chat=messages), "report.pdf", "application/pdf", ":material/picture_as_pdf:"),
        ("Everything", "ZIP with every file below plus the PDF and the raw meeting data (JSON).",
         lambda: export_bundle(record, messages), "meeting.zip", "application/zip", ":material/folder_zip:"),
        ("Transcript", "Full transcript with [mm:ss] timestamps.",
         transcript_to_txt(record), "transcript.txt", "text/plain", ":material/description:"),
        ("Summary", "Overview, topics, outcomes and next steps.",
         section_to_txt(record, "summary"), "summary.txt", "text/plain", ":material/summarize:"),
        ("Action items", "Tasks with owners and deadlines (only when stated).",
         section_to_txt(record, "action_items"), "action_items.txt", "text/plain", ":material/task_alt:"),
        ("Key decisions", "Decisions that were explicitly made.",
         section_to_txt(record, "key_decisions"), "key_decisions.txt", "text/plain", ":material/gavel:"),
        ("Open questions", "Unresolved questions and follow-ups.",
         section_to_txt(record, "open_questions"), "open_questions.txt", "text/plain", ":material/help:"),
        ("Q&A chat", f"Your questions and the answers with sources ({sum(m['role'] == 'user' for m in messages)} asked).",
         chat_to_txt(record, messages), "chat.txt", "text/plain", ":material/forum:"),
    ]
    columns = st.columns(2)
    for index, (name, description, data, suffix, mime, icon) in enumerate(items):
        with columns[index % 2].container(border=True):
            st.markdown(f"**{name}**")
            st.caption(description)
            st.download_button(
                f"Download {name.lower()}", data=data, file_name=safe_filename(record.title, suffix), mime=mime,
                icon=icon, key=f"dlall_{suffix}", width="stretch",
            )


def render_meeting(record: MeetingRecord) -> None:
    render_meeting_header(record)
    render_metrics_and_actions(record)
    st.write("")
    tabs = st.tabs(TAB_NAMES, key="meeting_tab", on_change="rerun")
    renderers = {
        "Overview": lambda: render_overview(record),
        CHAT_TAB: lambda: render_chat(record),
        "Transcript": lambda: render_transcript(record),
        "Summary": lambda: render_section(record, "summary", record.summary),
        "Action Items": lambda: render_section(record, "action_items", record.action_items),
        "Key Decisions": lambda: render_section(record, "key_decisions", record.key_decisions),
        "Open Questions": lambda: render_section(record, "open_questions", record.open_questions),
        "Downloads": lambda: render_downloads(record),
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
