"""Map-Reduce summarisation and title generation (offline, fake Gemini)."""

from __future__ import annotations

import pytest

import core.summarizer as summarizer
from core.summarizer import clean_title, collapse_summaries, generate_title, split_transcript, summarize
from tests.conftest import SAMPLE_TRANSCRIPT


def test_short_transcript_single_call(fake_llm):
    llm = fake_llm("### Overview\nA product sync.")
    result = summarize(SAMPLE_TRANSCRIPT)
    assert result.startswith("### Overview")
    assert len(llm.calls) == 1
    assert "full transcript" in llm.system_prompts()[0]
    assert SAMPLE_TRANSCRIPT in llm.last_human()


def test_long_transcript_uses_map_reduce(fake_llm, monkeypatch):
    monkeypatch.setattr(summarizer, "SINGLE_PASS_MAX_CHARS", 1_000)
    monkeypatch.setattr(summarizer, "MAP_CHUNK_CHARS", 800)
    monkeypatch.setattr(summarizer, "MAP_CHUNK_OVERLAP", 50)
    transcript = " ".join(f"Sentence number {i} about topic {i % 7}." for i in range(300))
    chunks = split_transcript(transcript, 800, 50)
    assert len(chunks) > 3

    def respond(messages):
        system = str(messages[0].content)
        return "- partial note" if "one portion" in system else "### Overview\nFinal."

    llm = fake_llm(respond)
    progress = []
    result = summarize(transcript, progress=lambda m, f: progress.append(m))
    assert result == "### Overview\nFinal."
    map_calls = [c for c in llm.calls if "one portion" in str(c[0].content)]
    assert len(map_calls) == len(chunks)
    assert f"Transcript portion 1 of {len(chunks)}" in str(map_calls[0][-1].content)
    assert "chronological notes" in str(llm.calls[-1][0].content)  # final reduce prompt
    assert any("Writing final summary" in m for m in progress)


def test_collapse_reduces_until_it_fits():
    calls = []

    def collapse(text):
        calls.append(text)
        return "x" * 50

    summaries = ["y" * 400 for _ in range(10)]
    result = collapse_summaries(summaries, collapse, max_chars=1_000)
    assert sum(len(s) for s in result) <= 1_000
    assert calls  # at least one collapse happened


def test_collapse_is_bounded():
    result = collapse_summaries(["z" * 600] * 8, lambda text: text, max_chars=100, max_rounds=2)
    assert result  # terminates even when collapsing can't shrink the text


def test_no_collapse_when_small():
    calls = []
    assert collapse_summaries(["a", "b"], lambda t: calls.append(t) or t, max_chars=100) == ["a", "b"]
    assert calls == []


def test_summarize_rejects_empty():
    with pytest.raises(ValueError):
        summarize("   ")


@pytest.mark.parametrize(
    "raw, expected",
    [
        ('"Q3 Launch Planning"', "Q3 Launch Planning"),
        ("Title: Weekly Product Sync.", "Weekly Product Sync"),
        ("**Budget Review**\nextra line", "Budget Review"),
        ("# Meeting Title: Hiring Plan", "Hiring Plan"),
        ("", "Untitled Meeting"),
    ],
)
def test_clean_title(raw, expected):
    assert clean_title(raw) == expected


def test_clean_title_limits_length():
    assert len(clean_title(" ".join(["word"] * 30)).split()) == summarizer.TITLE_MAX_WORDS


def test_generate_title_prefers_summary(fake_llm):
    llm = fake_llm("Mobile App Launch Sync")
    assert generate_title(SAMPLE_TRANSCRIPT, summary="SUMMARY TEXT") == "Mobile App Launch Sync"
    assert llm.last_human() == "SUMMARY TEXT"


def test_generate_title_falls_back_to_transcript(fake_llm):
    llm = fake_llm("Product Sync")
    generate_title(SAMPLE_TRANSCRIPT)
    assert llm.last_human() == SAMPLE_TRANSCRIPT
