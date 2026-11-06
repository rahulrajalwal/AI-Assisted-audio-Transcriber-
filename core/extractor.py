"""Action-item, key-decision and open-question extraction (LangChain LCEL).

The chat model comes from ``core.llm`` (active provider: OpenRouter, model from OPENROUTER_MODEL;
Gemini and Mistral are preserved there in an inactive section).

Each extraction keeps its own chain and prompt. A transcript that fits the model's
context budget is processed in one request; longer ones are split, extracted per
section (sequentially) and then consolidated (deduplicated), so no request ever
exceeds the context window and no part of the transcript is skipped.
"""

from __future__ import annotations

import re
from typing import Callable

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from core.llm import build_text_chain, input_char_budget, invoke_with_retry
from core.summarizer import collapse_summaries, split_transcript

# ``None`` = derive from the model's context window (core.llm.input_char_budget).
EXTRACTION_SINGLE_PASS_MAX_CHARS: int | None = None
EXTRACTION_CHUNK_CHARS: int | None = None
EXTRACTION_CHUNK_OVERLAP = 300

NO_ACTION_ITEMS = "No action items found."
NO_KEY_DECISIONS = "No key decisions found."
NO_OPEN_QUESTIONS = "No open questions found."

ProgressFn = Callable[[str, float | None], None]

_RULES = """Rules:
- Use ONLY what is stated in the transcript. Never invent people, owners, deadlines, numbers or outcomes.
- Never guess missing details; follow the format instructions below for how to show them.
- Keep dates and deadlines exactly as said (e.g. "Wednesday", "the fourteenth"); never add months, years or calendar dates.
- Do not repeat the same item twice; merge duplicates.
- Output only the requested list — no introduction or closing remarks."""

ACTION_ITEMS_PROMPT = f"""You are an expert meeting analyst. Extract every ACTION ITEM from the meeting transcript.

An action item is a concrete task that someone committed to, was assigned, or was explicitly asked to do. General ideas, suggestions nobody agreed to act on, and past work that is already finished are NOT action items.
Decisions themselves (e.g. "we decided to launch on 15 August") are NOT action items — they are reported separately. Only list the specific follow-up tasks that carry them out.

{_RULES}
- Owner: include a person/team ONLY if the transcript explicitly names who will do it; otherwise write "Not specified".
- Deadline: include ONLY a deadline or time frame explicitly stated in the transcript; otherwise write "Not specified".

Format as a numbered Markdown list, exactly like this:
1. **<Task, starting with a verb>**
   - **Owner:** <name, or Not specified>
   - **Deadline:** <as stated, or Not specified>

If there are no action items, reply exactly: {NO_ACTION_ITEMS}"""

KEY_DECISIONS_PROMPT = f"""You are an expert meeting analyst. Extract every KEY DECISION from the meeting transcript.

A decision is a choice about direction, scope, timing, budget or approach that the participants explicitly agreed, approved, chose, confirmed or rejected. Proposals, suggestions, options under discussion and personal opinions are NOT decisions unless the transcript shows they were accepted.
Task assignments ("X will do Y by Z") are action items, NOT decisions — do not list them here.

{_RULES}

Format as a numbered Markdown list, exactly like this:
1. **<The decision>** — <brief context or rationale>

Add the " — <context>" part only when the transcript states a reason, condition or who decided; otherwise write just the bold decision. Never write "Not specified" here.

If there are no decisions, reply exactly: {NO_KEY_DECISIONS}"""

OPEN_QUESTIONS_PROMPT = f"""You are an expert meeting analyst. Extract every OPEN QUESTION from the meeting transcript.

An open question is a question, issue, risk or topic that was raised but NOT resolved in the meeting, or that was explicitly deferred for follow-up. Do NOT include questions that were answered or matters that were decided.

{_RULES}

Format as a numbered Markdown list, exactly like this:
1. **<The unresolved question or topic>** — <what is pending or planned next; who raised it only if stated>

Add the " — <context>" part only when the transcript gives such details; never write "Not specified" here.

If there are no open questions, reply exactly: {NO_OPEN_QUESTIONS}"""

CONSOLIDATE_PROMPT = """You merge {kind} that were extracted separately from consecutive sections of ONE meeting.

- Merge duplicates and near-duplicates into one item, keeping the most specific details.
- Keep every distinct item; do not add anything new; keep "Not specified" where details are missing.
- Preserve the exact output format used in the input lists and renumber from 1.
- If every input says none were found, reply exactly: {empty_message}"""


def build_chain(system_prompt: str) -> Runnable:
    """Reusable LCEL chain: transcript text → prompt → Gemini → plain string."""
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "Meeting transcript:\n\n{text}")]
    )
    return (
        RunnablePassthrough()
        | RunnableLambda(lambda text: {"text": text})
        | build_text_chain(prompt)
    )


def _build_consolidate_chain(kind: str, empty_message: str) -> Runnable:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONSOLIDATE_PROMPT),
            ("human", "{text}"),
        ]
    ).partial(kind=kind, empty_message=empty_message)
    return (
        RunnablePassthrough()
        | RunnableLambda(lambda text: {"text": text})
        | build_text_chain(prompt, thinking_level="low")
    )


def _is_empty_result(text: str, empty_message: str) -> bool:
    return text.strip().rstrip(".").lower() == empty_message.rstrip(".").lower()


def _extract(
    transcript: str,
    system_prompt: str,
    empty_message: str,
    kind: str,
    task: str,
    progress: ProgressFn | None = None,
) -> str:
    text = (transcript or "").strip()
    if not text:
        raise ValueError("Cannot extract information from an empty transcript.")

    chain = build_chain(system_prompt)
    budget = input_char_budget(reserve_tokens=150)  # extraction prompts are longer
    if len(text) <= (EXTRACTION_SINGLE_PASS_MAX_CHARS or budget):
        result = invoke_with_retry(chain, text, task=task)
        return empty_message if _is_empty_result(result, empty_message) else result

    sections = split_transcript(text, EXTRACTION_CHUNK_CHARS or int(budget * 0.9), EXTRACTION_CHUNK_OVERLAP)
    partial_results: list[str] = []
    for index, section in enumerate(sections, start=1):
        if progress:
            progress(f"{task}: section {index}/{len(sections)}", None)
        result = invoke_with_retry(chain, section, task=f"{task} (section {index}/{len(sections)})")
        if not _is_empty_result(result, empty_message):
            partial_results.append(result)

    if not partial_results:
        return empty_message
    if len(partial_results) == 1:
        return partial_results[0]
    consolidate_chain = _build_consolidate_chain(kind, empty_message)

    def consolidate(block: str) -> str:
        return invoke_with_retry(consolidate_chain, block, task=f"{task} (consolidation)")

    # Hierarchical: merge groups of lists that fit the context budget, until one remains.
    lists = collapse_summaries(partial_results, consolidate, max_chars=budget)
    if len(lists) > 1:
        lists = [consolidate("\n\n---\n\n".join(
            f"List from section {i}:\n{result}" for i, result in enumerate(lists, start=1)
        ))]
    consolidated = lists[0]
    return empty_message if _is_empty_result(consolidated, empty_message) else consolidated


def extract_action_items(transcript: str, progress: ProgressFn | None = None) -> str:
    """Numbered Markdown list of action items (owner/deadline only when stated)."""
    return _extract(
        transcript, ACTION_ITEMS_PROMPT, NO_ACTION_ITEMS, "action-item lists", "Action-item extraction", progress
    )


def extract_key_decisions(transcript: str, progress: ProgressFn | None = None) -> str:
    """Numbered Markdown list of decisions that were explicitly made."""
    return _extract(
        transcript, KEY_DECISIONS_PROMPT, NO_KEY_DECISIONS, "key-decision lists", "Key-decision extraction", progress
    )


def extract_questions(transcript: str, progress: ProgressFn | None = None) -> str:
    """Numbered Markdown list of unresolved questions / follow-up topics."""
    return _extract(
        transcript, OPEN_QUESTIONS_PROMPT, NO_OPEN_QUESTIONS, "open-question lists", "Open-question extraction", progress
    )


def count_items(markdown_list: str) -> int:
    """Count top-level numbered items in an extraction result (0 for the 'none found' text)."""
    return len(re.findall(r"^\d+\.\s", markdown_list or "", flags=re.M))
