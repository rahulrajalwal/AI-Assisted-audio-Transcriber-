"""TXT and PDF exports."""

from __future__ import annotations

import utils.exporter as exporter
from core.meeting_store import MeetingRecord
from utils.exporter import build_pdf_report, safe_filename, summary_to_txt, transcript_to_txt
from tests.conftest import SAMPLE_TRANSCRIPT


def make_record(**overrides) -> MeetingRecord:
    data = dict(
        meeting_id="0123456789abcdef",
        title="Weekly Product Sync — “Launch” ₹ budget",
        transcript=SAMPLE_TRANSCRIPT,
        summary="### Overview\nThe team discussed the **launch**.\n\n### Main Topics Discussed\n- **Launch** — 15 August\n  - nested point",
        action_items="1. **Prepare the press kit**\n   - **Owner:** Priya\n   - **Deadline:** Friday",
        key_decisions="1. **Launch on 15 August** — agreed by the team",
        open_questions="No open questions found.",
        source="https://youtu.be/abc",
        language="english",
        processed_at="2026-11-05T10:00:00+00:00",
        model="gemini-3.8-flash",
        duration_seconds=37.0,
    )
    data.update(overrides)
    return MeetingRecord(**data)


def test_transcript_txt():
    text = transcript_to_txt(make_record())
    assert "FULL TRANSCRIPT" in text and SAMPLE_TRANSCRIPT in text
    assert "Source: https://youtu.be/abc" in text


def test_summary_txt_contains_all_sections_without_markdown():
    text = summary_to_txt(make_record())
    for heading in ("MEETING SUMMARY", "ACTION ITEMS", "KEY DECISIONS", "OPEN QUESTIONS"):
        assert heading in text
    assert "**" not in text and "### " not in text


def test_pdf_report_unicode():
    pdf = build_pdf_report(make_record())
    assert pdf.startswith(b"%PDF") and len(pdf) > 2_000


def test_pdf_report_falls_back_to_core_font(monkeypatch):
    monkeypatch.setattr(exporter, "_FONT_CANDIDATES", ())
    pdf = build_pdf_report(make_record(), include_transcript=False)
    assert pdf.startswith(b"%PDF")


def test_pdf_handles_long_transcript_and_odd_markdown():
    record = make_record(
        transcript=("word " * 20_000) + "https://example.com/" + "x" * 300,
        summary="Plain text with __underscores__ and -- dashes and * stray asterisk",
    )
    assert build_pdf_report(record).startswith(b"%PDF")


def test_safe_filename():
    assert safe_filename('Q3: "Plan" / Review?', "report.pdf") == "Q3_Plan__Review_report.pdf"
