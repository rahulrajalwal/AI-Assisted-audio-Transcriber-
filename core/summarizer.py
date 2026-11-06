"""Map-Reduce meeting summarisation and title generation (LangChain LCEL).

The chat model comes from ``core.llm`` (active provider: OpenRouter, model from OPENROUTER_MODEL;
the Gemini and Mistral integrations are preserved there in an inactive section).

Strategy
--------
* All sizes derive from the model's context window (``core.llm.input_char_budget``),
  so with a small local context (e.g. 4096 tokens) no prompt is ever truncated.
* A transcript that fits the budget is summarised in one call.
* Longer transcripts are split into overlapping chunks (MAP, processed one at a
  time); intermediate summaries are *collapsed* hierarchically in groups until they
  fit the budget (bounded rounds); one REDUCE call writes the final summary.
  Nothing is dropped: every part of the transcript passes through a MAP call.
"""

from __future__ import annotations

import logging
import re
from typing import Callable

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.llm import build_text_chain, input_char_budget, invoke_with_retry

logger = logging.getLogger(__name__)

# Sizes are in characters. ``None`` means "derive from the model's context window"
# (core.llm.input_char_budget); set an int to override (tests do this).
SINGLE_PASS_MAX_CHARS: int | None = None
MAP_CHUNK_CHARS: int | None = None
MAP_CHUNK_OVERLAP = 300
REDUCE_MAX_CHARS: int | None = None
MAX_COLLAPSE_ROUNDS = 6
TITLE_INPUT_MAX_CHARS = 4_000
MAP_SUMMARY_MAX_WORDS = 180  # keeps intermediate summaries small so reductions fit
TITLE_MAX_WORDS = 10

ProgressFn = Callable[[str, float | None], None]

_FAITHFULNESS = (
    "Use ONLY information stated in the provided text. Never invent names, numbers, "
    "dates, decisions or commitments, and never attribute a statement or approval to someone "
    "unless the text says who it was. Keep dates and deadlines exactly as said (e.g. 'Wednesday', "
    "'the fourteenth') - do not add months, years or weekdays. If something is unclear in the "
    "text, say so rather than guessing. Write in clear, professional English."
)

FINAL_SUMMARY_FORMAT = """Use exactly this Markdown structure (omit a section only if the text has nothing for it):

### Overview
2–4 sentences describing the purpose of the meeting and its overall outcome.

### Main Topics Discussed
- **<Topic>** — the key points, facts and viewpoints raised.

### Outcomes & Agreements
- Conclusions or agreements that were explicitly reached.

### Next Steps
- Follow-ups that were explicitly mentioned (owner/date only if stated).

Keep it concise, avoid repetition, and prioritise information that matters to someone who missed the meeting."""


def single_pass_max_chars() -> int:
    return SINGLE_PASS_MAX_CHARS or input_char_budget()


def map_chunk_chars() -> int:
    return MAP_CHUNK_CHARS or int(input_char_budget() * 0.9)


def reduce_max_chars() -> int:
    return REDUCE_MAX_CHARS or input_char_budget()


def split_transcript(
    transcript: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """Split a transcript into overlapping chunks for the MAP step."""
    chunk_size = chunk_size or map_chunk_chars()
    chunk_overlap = MAP_CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )
    return splitter.split_text(transcript)


# ── LCEL chains ─────────────────────────────────────────────────────────────────


def build_map_chain() -> Runnable:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You summarise one portion of a longer meeting transcript. "
                + _FAITHFULNESS
                + "\nCapture every substantive point: topics, facts and figures, opinions, "
                "decisions, action items (with owners/deadlines only if stated) and open "
                f"questions. Use compact bullet points, at most {MAP_SUMMARY_MAX_WORDS} words. "
                "Do not add an introduction.",
            ),
            ("human", "Transcript portion {part} of {total}:\n\n{text}"),
        ]
    )
    # Low thinking level: the MAP step is high-volume and straightforward.
    return build_text_chain(prompt, thinking_level="low")


def build_collapse_chain() -> Runnable:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You merge partial notes from consecutive parts of the same meeting into "
                "one set of notes. "
                + _FAITHFULNESS
                + "\nRemove duplicates, keep every distinct fact, decision, action item and "
                "open question, and keep chronological order. Use compact bullet points.",
            ),
            ("human", "{text}"),
        ]
    )
    return build_text_chain(prompt, thinking_level="low")


def build_final_chain() -> Runnable:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert meeting analyst writing the final summary of a meeting "
                "from {source_description}. "
                + _FAITHFULNESS
                + "\n\n"
                + FINAL_SUMMARY_FORMAT,
            ),
            ("human", "{text}"),
        ]
    )
    return build_text_chain(prompt)


def build_title_chain() -> Runnable:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Write a short, specific, professional title for this meeting "
                f"(at most {TITLE_MAX_WORDS - 2} words) based only on the text. "
                "Return only the title — no quotes, no prefix, no punctuation at the end.",
            ),
            ("human", "{text}"),
        ]
    )
    return (
        RunnablePassthrough()
        | RunnableLambda(lambda text: {"text": text})
        | build_text_chain(prompt, thinking_level="low")
        | RunnableLambda(clean_title)
    )


# ── Helpers ─────────────────────────────────────────────────────────────────────


def clean_title(raw: str) -> str:
    """Normalise a model-generated title to a single short line."""
    lines = [line for line in (raw or "").strip().splitlines() if line.strip()]
    title = lines[0] if lines else ""
    title = re.sub(r"^\s*(#+\s*)?(meeting\s+)?title\s*[:\-]\s*", "", title, flags=re.I)
    title = title.strip().strip("*_`\"'“”‘’ ").rstrip(".")
    words = title.split()
    if len(words) > TITLE_MAX_WORDS:
        title = " ".join(words[:TITLE_MAX_WORDS])
    return title or "Untitled Meeting"


def _group_by_length(texts: list[str], max_chars: int) -> list[list[str]]:
    """Greedily pack texts into groups whose combined length stays under ``max_chars``."""
    groups: list[list[str]] = []
    current: list[str] = []
    current_len = 0
    for text in texts:
        if current and current_len + len(text) > max_chars:
            groups.append(current)
            current, current_len = [], 0
        current.append(text)
        current_len += len(text) + 2
    if current:
        groups.append(current)
    return groups


def collapse_summaries(
    summaries: list[str],
    collapse_fn: Callable[[str], str],
    max_chars: int | None = None,
    max_rounds: int = MAX_COLLAPSE_ROUNDS,
) -> list[str]:
    """Recursively merge intermediate summaries until they fit in ``max_chars``."""
    max_chars = max_chars or reduce_max_chars()
    rounds = 0
    while sum(len(s) for s in summaries) > max_chars and len(summaries) > 1:
        if rounds >= max_rounds:
            logger.warning("Stopped collapsing after %d rounds.", rounds)
            break
        groups = _group_by_length(summaries, max_chars)
        if len(groups) == len(summaries):
            # Every summary is already as large as a group; pair them up to make progress.
            groups = [summaries[i : i + 2] for i in range(0, len(summaries), 2)]
        summaries = [collapse_fn("\n\n---\n\n".join(group)) for group in groups]
        rounds += 1
    return summaries


# ── Public API ──────────────────────────────────────────────────────────────────


def summarize(transcript: str, progress: ProgressFn | None = None) -> str:
    """Return a structured Markdown summary of the transcript (Map-Reduce)."""
    text = (transcript or "").strip()
    if not text:
        raise ValueError("Cannot summarise an empty transcript.")

    def report(message: str, fraction: float | None = None) -> None:
        if progress:
            progress(message, fraction)

    final_chain = build_final_chain()

    if len(text) <= single_pass_max_chars():
        report("Summarising transcript (single pass)", 0.1)
        return invoke_with_retry(
            final_chain,
            {"text": text, "source_description": "its full transcript"},
            task="Meeting summary",
        )

    chunks = split_transcript(text)
    map_chain = build_map_chain()
    partials: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        report(f"Summarising section {index}/{len(chunks)}", (index - 1) / (len(chunks) + 1))
        partials.append(
            invoke_with_retry(
                map_chain,
                {"text": chunk, "part": index, "total": len(chunks)},
                task=f"Summary of section {index}/{len(chunks)}",
            )
        )

    collapse_chain = build_collapse_chain()

    def collapse(text_block: str) -> str:
        report("Condensing intermediate summaries", None)
        return invoke_with_retry(collapse_chain, {"text": text_block}, task="Summary reduction")

    partials = collapse_summaries(partials, collapse)

    report("Writing final summary", len(chunks) / (len(chunks) + 1))
    combined = "\n\n---\n\n".join(
        f"Notes for part {i} of {len(partials)}:\n{summary}"
        for i, summary in enumerate(partials, start=1)
    )
    return invoke_with_retry(
        final_chain,
        {
            "text": combined,
            "source_description": "chronological notes covering consecutive parts of the meeting",
        },
        task="Meeting summary",
    )


def generate_title(transcript: str, summary: str | None = None) -> str:
    """Generate a short meeting title.

    Uses the summary when available (it covers the whole meeting); otherwise an
    excerpt from the start of the transcript.
    """
    source = (summary or "").strip() or (transcript or "").strip()
    if not source:
        raise ValueError("Cannot generate a title from empty text.")
    return invoke_with_retry(
        build_title_chain(), source[:TITLE_INPUT_MAX_CHARS], task="Title generation"
    )
