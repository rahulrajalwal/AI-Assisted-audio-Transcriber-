"""TXT, PDF (fpdf2) and ZIP exports for processed meetings — every section downloadable."""

from __future__ import annotations

import io
import json
import re
import unicodedata
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from core.meeting_store import MeetingRecord

# Unicode TTF fonts tried in order (Windows, Linux, macOS). Falls back to the
# built-in Helvetica (Latin-1 only) with characters transliterated.
_FONT_CANDIDATES = (
    (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"),
    (r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\segoeuib.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf"),
)

_ASCII_REPLACEMENTS = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-",
    "\u2026": "...", "\u2022": "-", "\u2192": "->", "\u20b9": "Rs.", "\u00a0": " ",
}


def format_date(iso_value: str) -> str:
    try:
        return datetime.fromisoformat(iso_value).astimezone().strftime("%d %b %Y, %H:%M")
    except (TypeError, ValueError):
        return iso_value or ""


def safe_filename(title: str, suffix: str) -> str:
    slug = re.sub(r"[^\w\- ]+", "", title or "meeting").strip().replace(" ", "_")[:60]
    return f"{slug or 'meeting'}_{suffix}"


def _header_lines(record: MeetingRecord) -> list[str]:
    name = record.display_name
    lines = [name, "=" * min(len(name), 80)]
    if record.title and record.title != name:
        lines.append(f"AI title: {record.title}")
    if record.processed_at:
        lines.append(f"Processed: {format_date(record.processed_at)}")
    if record.source:
        lines.append(f"Source: {record.source}")
    if record.language:
        lines.append(f"Language: {record.language.title()}")
    return lines


def format_clock(seconds: Any) -> str:
    """Seconds → "mm:ss" or "h:mm:ss"."""
    try:
        total = int(float(seconds))
    except (TypeError, ValueError):
        return ""
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def timestamped_paragraphs(record: MeetingRecord, window_seconds: float = 45.0) -> list[tuple[str, str]]:
    """Split the transcript into readable paragraphs with a start timestamp each.

    Uses the timed segments from Whisper/Sarvam when available (``[mm:ss]`` labels);
    otherwise groups sentences, with an empty label.
    """
    transcript = record.transcript or ""
    segments = [s for s in (record.segments or []) if "char_start" in s and "char_end" in s]
    if not segments:
        sentences = re.split(r"(?<=[.!?])\s+", transcript.strip())
        return [("", " ".join(sentences[i : i + 5])) for i in range(0, len(sentences), 5) if sentences[i : i + 5]]
    paragraphs: list[tuple[str, str]] = []
    group_start, group_begin, group_end = None, 0, 0
    for seg in segments:
        if group_start is None:
            group_start, group_begin = seg.get("start", 0.0), seg["char_start"]
        group_end = seg["char_end"]
        if seg.get("end", 0.0) - group_start >= window_seconds:
            paragraphs.append((format_clock(group_start), transcript[group_begin:group_end].strip()))
            group_start = None
    if group_start is not None:
        paragraphs.append((format_clock(group_start), transcript[group_begin:group_end].strip()))
    return [(label, text) for label, text in paragraphs if text]


def transcript_to_txt(record: MeetingRecord) -> str:
    """Full transcript, with [mm:ss] timestamps when available."""
    paragraphs = timestamped_paragraphs(record)
    if any(label for label, _ in paragraphs):
        body = "\n\n".join(f"[{label}] {text}" for label, text in paragraphs)
    else:
        body = record.transcript.strip()
    return "\n".join(_header_lines(record) + ["", "FULL TRANSCRIPT", "", body, ""])


SECTIONS = {
    "summary": ("MEETING SUMMARY", "summary"),
    "action_items": ("ACTION ITEMS", "action_items"),
    "key_decisions": ("KEY DECISIONS", "key_decisions"),
    "open_questions": ("OPEN QUESTIONS", "open_questions"),
}


def section_to_txt(record: MeetingRecord, section: str) -> str:
    """One section (summary, action_items, key_decisions, open_questions) as plain text."""
    heading, attribute = SECTIONS[section]
    content = _strip_markdown(getattr(record, attribute) or "").strip()
    return "\n".join(_header_lines(record) + ["", heading, "-" * len(heading), content, ""])


def chat_to_txt(record: MeetingRecord, messages: list[dict[str, Any]]) -> str:
    """The Q&A conversation, with the sources each answer used."""
    lines = _header_lines(record) + ["", "QUESTIONS & ANSWERS", "-" * 19]
    number = 0
    for message in messages:
        if message.get("role") == "user":
            number += 1
            lines += ["", f"Q{number}: {message.get('content', '').strip()}"]
        else:
            lines += [f"A{number}: {_strip_markdown(message.get('content', '')).strip()}"]
            for source in message.get("sources") or []:
                when = format_clock(source.get("start_seconds")) if source.get("start_seconds") is not None else ""
                label = f"   [Source {source.get('number')}] {source.get('title', '')}" + (f" @ {when}" if when else "")
                lines.append(label)
    if number == 0:
        lines += ["", "No questions asked yet."]
    return "\n".join(lines + [""])


def summary_to_txt(record: MeetingRecord) -> str:
    sections = [
        ("MEETING SUMMARY", record.summary),
        ("ACTION ITEMS", record.action_items),
        ("KEY DECISIONS", record.key_decisions),
        ("OPEN QUESTIONS", record.open_questions),
    ]
    body: list[str] = []
    for heading, content in sections:
        body += ["", heading, "-" * len(heading), _strip_markdown(content).strip()]
    return "\n".join(_header_lines(record) + body + [""])


def _strip_markdown(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text or "")
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)
    return text


class _ReportPDF(FPDF):
    def __init__(self, title: str):
        super().__init__(format="A4")
        self.report_title = title
        self.unicode_font = self._register_font()
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(18, 18, 18)

    def _register_font(self) -> bool:
        for regular, bold in _FONT_CANDIDATES:
            if Path(regular).exists():
                self.add_font("Body", "", regular)
                self.add_font("Body", "B", bold if Path(bold).exists() else regular)
                return True
        return False

    def font(self, style: str = "", size: float = 10.5) -> None:
        self.set_font("Body" if self.unicode_font else "Helvetica", style, size)

    def clean(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text or "")
        if self.unicode_font:
            return text
        for char, replacement in _ASCII_REPLACEMENTS.items():
            text = text.replace(char, replacement)
        return text.encode("latin-1", "replace").decode("latin-1")

    def footer(self) -> None:
        self.set_y(-12)
        self.font("", 8)
        self.set_text_color(130, 130, 140)
        self.cell(0, 6, self.clean(f"{self.report_title}  -  Page {self.page_no()}"), align="C")


def _write_markdown(pdf: _ReportPDF, markdown: str) -> None:
    """Render the Markdown subset produced by the prompts (headings, lists, bold)."""
    for raw_line in (markdown or "").splitlines():
        if not raw_line.strip():
            pdf.ln(2)
            continue
        indent_spaces = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            pdf.ln(1.5)
            pdf.font("B", 12 if len(heading.group(1)) <= 3 else 11)
            pdf.set_text_color(40, 40, 60)
            pdf.multi_cell(0, 6.5, pdf.clean(heading.group(2).replace("**", "")), markdown=False,
                           new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            continue

        indent = min(indent_spaces // 2, 4) * 5
        bullet = ""
        bullet_match = re.match(r"^([-*+])\s+(.*)$", line)
        number_match = re.match(r"^(\d+[.)])\s+(.*)$", line)
        if bullet_match:
            bullet, line = ("•" if pdf.unicode_font else "-"), bullet_match.group(2)
        elif number_match:
            bullet, line = number_match.group(1), number_match.group(2)

        pdf.font("", 10.5)
        pdf.set_text_color(30, 30, 35)
        left = pdf.l_margin + indent
        if bullet:
            pdf.set_x(left)
            pdf.cell(7, 5.6, pdf.clean(bullet))
            left += 7
        pdf.set_x(left)
        # fpdf2 markdown: **bold** only; escape other markers it would interpret.
        text = pdf.clean(line).replace("__", "_\u200b_" if pdf.unicode_font else "_ _").replace("--", "- -")
        pdf.multi_cell(pdf.w - pdf.r_margin - left, 5.6, text, markdown=True,
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def _section(pdf: _ReportPDF, title: str, markdown: str) -> None:
    pdf.ln(4)
    pdf.font("B", 13.5)
    pdf.set_text_color(70, 60, 200)
    pdf.cell(0, 8, pdf.clean(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_draw_color(210, 210, 225)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(2.5)
    _write_markdown(pdf, markdown)


def build_pdf_report(
    record: MeetingRecord, include_transcript: bool = True, chat: list[dict[str, Any]] | None = None
) -> bytes:
    """Create a formatted PDF meeting report (optionally with the Q&A chat) and return its bytes."""
    name = record.display_name
    pdf = _ReportPDF(name)
    pdf.set_title(name)
    pdf.set_creator("AI Meeting Assistant")
    pdf.add_page()

    pdf.font("B", 19)
    pdf.set_text_color(20, 20, 30)
    pdf.multi_cell(0, 9, pdf.clean(name), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    if record.title and record.title != name:
        pdf.font("I", 11)
        pdf.set_text_color(90, 90, 110)
        pdf.multi_cell(0, 6, pdf.clean(f"AI title: {record.title}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)
    pdf.font("", 9.5)
    pdf.set_text_color(110, 110, 125)
    meta = []
    if record.processed_at:
        meta.append(f"Processed {format_date(record.processed_at)}")
    if record.language:
        meta.append(f"Language: {record.language.title()}")
    if record.model:
        meta.append(f"Model: {record.model}")
    pdf.multi_cell(0, 5, pdf.clean("   |   ".join(meta)), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    if record.source:
        pdf.multi_cell(0, 5, pdf.clean(f"Source: {record.source}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    _section(pdf, "Meeting Summary", record.summary)
    _section(pdf, "Action Items", record.action_items)
    _section(pdf, "Key Decisions", record.key_decisions)
    _section(pdf, "Open Questions", record.open_questions)

    if chat and any(m.get("role") == "user" for m in chat):
        qa_lines, number = [], 0
        for message in chat:
            if message.get("role") == "user":
                number += 1
                qa_lines += ["", f"**Q{number}. {message.get('content', '').strip()}**"]
            else:
                qa_lines += [_strip_markdown(message.get("content", "")).strip()]
        _section(pdf, "Questions & Answers", "\n".join(qa_lines).strip())

    if include_transcript:
        pdf.add_page()
        pdf.font("B", 13.5)
        pdf.set_text_color(70, 60, 200)
        pdf.cell(0, 8, "Full Transcript", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)
        pdf.font("", 9.5)
        pdf.set_text_color(45, 45, 50)
        for label, text in timestamped_paragraphs(record):
            if label:
                pdf.font("B", 9)
                pdf.set_text_color(90, 90, 140)
                pdf.cell(0, 5, label, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.font("", 9.5)
                pdf.set_text_color(45, 45, 50)
            pdf.multi_cell(0, 5, pdf.clean(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1.5)

    return bytes(pdf.output())


def export_bundle(record: MeetingRecord, chat: list[dict[str, Any]] | None = None) -> bytes:
    """Everything in one ZIP: transcript, each section, Q&A, PDF report and the raw JSON record."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("transcript.txt", transcript_to_txt(record))
        for section in SECTIONS:
            bundle.writestr(f"{section}.txt", section_to_txt(record, section))
        bundle.writestr("summary_all_sections.txt", summary_to_txt(record))
        bundle.writestr("questions_and_answers.txt", chat_to_txt(record, chat or []))
        bundle.writestr("meeting_report.pdf", build_pdf_report(record, chat=chat))
        bundle.writestr("meeting.json", json.dumps(record.to_dict(), ensure_ascii=False, indent=2))
    return buffer.getvalue()
