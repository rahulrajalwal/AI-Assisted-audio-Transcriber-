"""TXT and PDF exports."""

from __future__ import annotations

import utils.exporter as exporter
from core.meeting_store import MeetingRecord
from utils.exporter import (
    build_pdf_report,
    chat_to_txt,
    export_bundle,
    safe_filename,
    section_to_txt,
    summary_to_txt,
    timestamped_paragraphs,
    transcript_to_txt,
)
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


CHAT = [
    {"role": "user", "content": "Who prepares the press kit?"},
    {"role": "assistant", "content": "**Priya** prepares it by Friday [Source 1].",
     "sources": [{"number": 1, "title": "Weekly Product Sync", "start_seconds": 65.0}]},
]


def timed_record():
    text = "Hello team, welcome. " * 20 + "Now the budget. " * 20
    half = len(text) // 2
    segments = [
        {"start": 0.0, "end": 50.0, "audio_chunk": 0, "char_start": 0, "char_end": half},
        {"start": 50.0, "end": 100.0, "audio_chunk": 0, "char_start": half + 1, "char_end": len(text)},
    ]
    return make_record(transcript=text, segments=segments)


def test_timestamped_transcript():
    paragraphs = timestamped_paragraphs(timed_record())
    assert [label for label, _ in paragraphs] == ["00:00", "00:50"]
    assert "[00:50]" in transcript_to_txt(timed_record())


def test_transcript_without_timestamps_still_readable():
    paragraphs = timestamped_paragraphs(make_record())
    assert paragraphs and all(label == "" for label, _ in paragraphs)


def test_each_section_downloadable():
    record = make_record()
    assert "ACTION ITEMS" in section_to_txt(record, "action_items") and "Priya" in section_to_txt(record, "action_items")
    assert "KEY DECISIONS" in section_to_txt(record, "key_decisions")
    assert "**" not in section_to_txt(record, "summary")


def test_chat_export_and_pdf_with_qa():
    text = chat_to_txt(make_record(), CHAT)
    assert "Q1: Who prepares the press kit?" in text and "A1: Priya prepares it" in text and "@ 01:05" in text
    assert build_pdf_report(timed_record(), chat=CHAT).startswith(b"%PDF")


def test_bundle_contains_everything():
    import io
    import zipfile

    names = set(zipfile.ZipFile(io.BytesIO(export_bundle(timed_record(), CHAT))).namelist())
    assert {"transcript.txt", "summary.txt", "action_items.txt", "key_decisions.txt", "open_questions.txt",
            "questions_and_answers.txt", "meeting_report.pdf", "meeting.json"} <= names
