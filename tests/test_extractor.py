"""Action-item / decision / open-question extraction (offline, fake Gemini)."""

from __future__ import annotations

import core.extractor as extractor
from core.extractor import (
    NO_ACTION_ITEMS,
    NO_KEY_DECISIONS,
    NO_OPEN_QUESTIONS,
    count_items,
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from tests.conftest import SAMPLE_TRANSCRIPT

ACTIONS = "1. **Prepare the press kit**\n   - **Owner:** Priya\n   - **Deadline:** Friday"


def test_action_items_prompt_and_output(fake_llm):
    llm = fake_llm(ACTIONS)
    assert extract_action_items(SAMPLE_TRANSCRIPT) == ACTIONS
    system = llm.system_prompts()[0]
    assert "ACTION ITEM" in system
    assert "Not specified" in system  # never invent owners/deadlines
    assert "explicitly names" in system
    assert SAMPLE_TRANSCRIPT in llm.last_human()


def test_three_separate_chains(fake_llm):
    llm = fake_llm("1. **Something**")
    extract_action_items(SAMPLE_TRANSCRIPT)
    extract_key_decisions(SAMPLE_TRANSCRIPT)
    extract_questions(SAMPLE_TRANSCRIPT)
    prompts = llm.system_prompts()
    assert len(prompts) == 3
    assert "ACTION ITEM" in prompts[0]
    assert "KEY DECISION" in prompts[1] and "suggestions" in prompts[1]
    assert "OPEN QUESTION" in prompts[2] and "answered" in prompts[2]


def test_none_found_is_normalised(fake_llm):
    fake_llm("no key decisions found")
    assert extract_key_decisions(SAMPLE_TRANSCRIPT) == NO_KEY_DECISIONS


def test_long_transcript_extracts_per_section_and_consolidates(fake_llm, monkeypatch):
    monkeypatch.setattr(extractor, "EXTRACTION_SINGLE_PASS_MAX_CHARS", 500)
    monkeypatch.setattr(extractor, "EXTRACTION_CHUNK_CHARS", 400)
    monkeypatch.setattr(extractor, "EXTRACTION_CHUNK_OVERLAP", 20)
    transcript = SAMPLE_TRANSCRIPT * 4

    def respond(messages):
        if "You merge" in str(messages[0].content):
            return "1. **Merged item**"
        return "1. **Item**"

    llm = fake_llm(respond)
    assert extract_action_items(transcript) == "1. **Merged item**"
    assert "You merge" in llm.system_prompts()[-1]
    assert len(llm.calls) >= 3


def test_long_transcript_all_empty_skips_consolidation(fake_llm, monkeypatch):
    monkeypatch.setattr(extractor, "EXTRACTION_SINGLE_PASS_MAX_CHARS", 500)
    monkeypatch.setattr(extractor, "EXTRACTION_CHUNK_CHARS", 400)
    monkeypatch.setattr(extractor, "EXTRACTION_CHUNK_OVERLAP", 20)
    llm = fake_llm(NO_OPEN_QUESTIONS)
    assert extract_questions(SAMPLE_TRANSCRIPT * 4) == NO_OPEN_QUESTIONS
    assert all("You merge" not in p for p in llm.system_prompts())


def test_count_items():
    assert count_items(ACTIONS + "\n2. **Fix login bug**\n   - **Owner:** Rahul") == 2
    assert count_items(NO_ACTION_ITEMS) == 0
