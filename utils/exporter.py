"""TXT and PDF (fpdf2) exports for processed meetings."""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from pathlib import Path

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
    lines = [record.title, "=" * min(len(record.title), 80)]
    if record.processed_at:
        lines.append(f"Processed: {format_date(record.processed_at)}")
    if record.source:
        lines.append(f"Source: {record.source}")
    if record.language:
        lines.append(f"Language: {record.language.title()}")
    return lines


def transcript_to_txt(record: MeetingRecord) -> str:
    return "\n".join(_header_lines(record) + ["", "FULL TRANSCRIPT", "", record.transcript.strip(), ""])


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


def build_pdf_report(record: MeetingRecord, include_transcript: bool = True) -> bytes:
    """Create a formatted PDF meeting report and return its bytes."""
    pdf = _ReportPDF(record.title)
    pdf.set_title(record.title)
    pdf.set_creator("AI Meeting Assistant")
    pdf.add_page()

    pdf.font("B", 19)
    pdf.set_text_color(20, 20, 30)
    pdf.multi_cell(0, 9, pdf.clean(record.title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
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

    if include_transcript:
        pdf.add_page()
        pdf.font("B", 13.5)
        pdf.set_text_color(70, 60, 200)
        pdf.cell(0, 8, "Full Transcript", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)
        pdf.font("", 9.5)
        pdf.set_text_color(45, 45, 50)
        pdf.multi_cell(0, 5, pdf.clean(record.transcript), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    return bytes(pdf.output())
