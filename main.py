"""Command-line entry point for the AI Meeting Assistant.

Usage (PowerShell):
    python main.py                                  # interactive prompts
    python main.py "https://youtu.be/<id>" english
    python main.py "C:\\path\\to\\meeting.mp4" hinglish

The UI is started with:  streamlit run app.py
"""

from __future__ import annotations

import logging
import sys

from core.config import load_environment

load_environment()  # before importing modules that read configuration

from core.pipeline import STAGE_LABELS, PipelineError, ProgressEvent, run_pipeline  # noqa: E402
from core.rag_engine import ask_question, load_rag_chain  # noqa: E402
from core.llm import LLMError  # noqa: E402


def print_progress(event: ProgressEvent) -> None:
    label = STAGE_LABELS.get(event.stage, event.stage)
    symbol = {"running": "…", "done": "✔", "error": "✖"}.get(event.status, "·")
    detail = f" — {event.message}" if event.message else ""
    print(f"  {symbol} {label}{detail}", flush=True)


def main(argv: list[str]) -> int:
    # Windows consoles default to cp1252; avoid UnicodeEncodeError on symbols/LLM output.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if len(argv) >= 2:
        source = argv[1]
        language = argv[2] if len(argv) >= 3 else "english"
    else:
        source = input("Enter YouTube URL or local file path: ").strip().strip('"')
        language = input("Language (english/hinglish) [english]: ").strip() or "english"

    print("\nStarting AI Meeting Assistant pipeline\n")
    try:
        record = run_pipeline(source, language, progress=print_progress)
    except PipelineError as exc:
        print(f"\n✖ Failed during '{exc.stage_label}': {exc}")
        return 1

    print("\n" + "=" * 70)
    print(f"Title: {record.title}")
    print(f"\nSummary:\n{record.summary}")
    print(f"\nAction Items:\n{record.action_items}")
    print(f"\nKey Decisions:\n{record.key_decisions}")
    print(f"\nOpen Questions:\n{record.open_questions}")
    print("=" * 70)

    print("\nChat with your meeting (type 'exit' to quit)\n")
    rag_chain = load_rag_chain(meeting_id=record.meeting_id)
    history: list[dict[str, str]] = []
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            return 0
        if not question:
            continue
        try:
            result = ask_question(rag_chain, question, history)
        except LLMError as exc:
            print(f"\nError: {exc}\n")
            continue
        print(f"\nAssistant: {result['answer']}\n")
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": result["answer"]}]


if __name__ == "__main__":
    sys.exit(main(sys.argv))
