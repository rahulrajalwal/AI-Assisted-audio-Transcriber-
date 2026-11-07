"""Retrieval-Augmented Q&A over stored meeting transcripts (LangChain LCEL).

Flow: question (+ recent chat history) → ChromaDB MMR search over the stored
transcript chunks (relevant *and* spread across the recording, so long meetings are
covered) → excerpts in chronological order with meeting/date/time labels (trimmed to
the model's context budget) → ChatPromptTemplate → chat model (OpenRouter,
OPENROUTER_MODEL — see core/llm.py) → StrOutputParser → answer + sources.

* Scope: one meeting (``meeting_id``, the default in the app) or — only when asked —
  all stored meetings. One-meeting answers also get a short overview of that meeting
  (its stored summary and extracted lists) for whole-meeting questions.
* In all-meetings scope, time references in the question ("previous meeting",
  "yesterday", "this week", "last week", "this month", "last month") are resolved
  against stored meeting dates and turned into a metadata filter. If no meeting
  matches, the chain says so instead of answering from unrelated meetings.
* Follow-ups: the previous user question is added to the retrieval query and the
  last turns are passed to the prompt (no extra LLM call for rewriting).
* Nothing relevant retrieved → "not found" answer without calling the LLM.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any, Iterator, Sequence

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough

from core.llm import (
    build_stream_chain,
    build_text_chain,
    estimate_tokens,
    input_char_budget,
    invoke_with_retry,
    stream_with_retry,
)
from core.meeting_store import load_meeting
from core.vector_store import (
    build_vector_store,
    compute_meeting_id,
    get_retriever,  # noqa: F401  (re-exported for callers that build plain retrievers)
    list_indexed_meetings,
    load_vector_store,
    meeting_filter,
)

NOT_FOUND_MESSAGE = "I could not find this information in the meeting transcript."
DEFAULT_K = 6  # excerpts per answer (~800 characters each)
FETCH_K = 24  # MMR candidates; the k most relevant-yet-different are kept
MMR_LAMBDA = 0.7  # 1 = pure relevance, 0 = pure diversity
OVERVIEW_MAX_CHARS = 6_000
MAX_HISTORY_MESSAGES = 4  # last 2 question/answer pairs (small context window)
HISTORY_MESSAGE_MAX_CHARS = 500
CATALOGUE_MAX_MEETINGS = 8

RAG_SYSTEM_PROMPT = """You are a meeting assistant answering questions about recorded meetings.

Answer ONLY from the transcript excerpts in the context below.
- If the context does not contain the answer, reply exactly: "{not_found}"
- Never invent names, owners, deadlines, numbers or decisions. If the excerpts only partly answer, say what is missing.
- The "Date" in a source label is the date of that MEETING, not a deadline. Quote deadlines exactly as spoken (e.g. "Wednesday") and never convert them to calendar dates.
- Distinguish clearly between what was decided and what was only suggested or discussed.
- When excerpts come from different meetings, say which meeting (title and date) each point comes from.
- Cite the supporting excerpts inline as [Source N] and add the time from that source's label when it has one (e.g. [Source 2, ~12:30]). Never state a time that is not in a source label.
- The meeting overview (if present) was generated from the full transcript: use it for questions about the whole meeting (topics, all decisions, all action items), cite it as [Overview], and prefer the transcript excerpts for details and quotes. When a point also appears in an excerpt, cite that excerpt (with its time) rather than [Overview].
- Be concise; use bullet points for lists.

Use the conversation history only to understand what a follow-up question refers to — not as a source of facts.
{scope_note}
{overview}Context (transcript excerpts, in chronological order):
{context}"""


# ── Formatting ──────────────────────────────────────────────────────────────────


def format_timestamp(seconds: Any) -> str:
    try:
        total = int(float(seconds))
    except (TypeError, ValueError):
        return ""
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _meeting_name(meta: dict) -> str:
    """Original YouTube title / filename; older chunks only have the AI title."""
    return meta.get("source_title") or meta.get("title") or "Untitled meeting"


def _source_label(meta: dict) -> str:
    parts = [f"Meeting: {_meeting_name(meta)}"]
    when = meta.get("meeting_date") or str(meta.get("processed_at", ""))[:10]
    if when:
        parts.append(f"Date: {when}")
    if meta.get("start_seconds") is not None:
        parts.append(f"Time: ~{format_timestamp(meta['start_seconds'])}–{format_timestamp(meta.get('end_seconds'))}")
    elif meta.get("chunk_index") is not None:
        parts.append(f"Segment {int(meta['chunk_index']) + 1}")
    return " | ".join(parts)


def format_docs(docs: Sequence[Document]) -> str:
    """Render retrieved chunks with source labels the model can cite."""
    return "\n\n".join(
        f"[Source {number}] {_source_label(doc.metadata or {})}\n{doc.page_content.strip()}"
        for number, doc in enumerate(docs, start=1)
    )


def docs_to_sources(docs: Sequence[Document]) -> list[dict[str, Any]]:
    sources = []
    for number, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        sources.append(
            {
                "number": number,
                "meeting_id": meta.get("meeting_id", ""),
                "title": _meeting_name(meta),
                "meeting_date": meta.get("meeting_date") or str(meta.get("processed_at", ""))[:10],
                "chunk_index": meta.get("chunk_index"),
                "start_seconds": meta.get("start_seconds"),
                "end_seconds": meta.get("end_seconds"),
                "excerpt": doc.page_content.strip(),
            }
        )
    return sources


# ── Time-aware scoping for "all meetings" questions ─────────────────────────────

_PREVIOUS_MEETING = re.compile(r"\b(previous|last|prior|earlier|preceding)\s+(meeting|sync|call|session|review|standup)\b", re.I)


def _meeting_day(meeting: dict) -> date | None:
    value = meeting.get("meeting_date") or str(meeting.get("processed_at", ""))[:10]
    try:
        return date.fromisoformat(value[:10])
    except (TypeError, ValueError):
        return None


def resolve_time_scope(
    question: str,
    meetings: list[dict],
    current_meeting_id: str | None = None,
    today: date | None = None,
) -> tuple[list[str] | None, str]:
    """Map time references in a question to meeting IDs.

    Returns ``(None, "")`` when the question has no time reference, otherwise
    ``(meeting_ids, description)`` — ``meeting_ids`` may be empty if nothing matches.
    """
    today = today or date.today()
    q = question.lower()
    dated = [(m, _meeting_day(m)) for m in meetings if m.get("meeting_id") and m["meeting_id"] != "legacy"]
    dated = [(m, d) for m, d in dated if d is not None]

    if _PREVIOUS_MEETING.search(q):
        ordered = sorted(dated, key=lambda md: (md[1], md[0].get("processed_at", "")), reverse=True)
        if current_meeting_id:
            current = next((md for md in ordered if md[0]["meeting_id"] == current_meeting_id), None)
            ordered = [
                md for md in ordered
                if md[0]["meeting_id"] != current_meeting_id
                and (current is None or (md[1], md[0].get("processed_at", "")) <= (current[1], current[0].get("processed_at", "")))
            ]
        return ([ordered[0][0]["meeting_id"]] if ordered else []), "the previous meeting"

    windows: list[tuple[str, date, date]] = []
    monday = today - timedelta(days=today.weekday())
    first_of_month = today.replace(day=1)
    last_month_end = first_of_month - timedelta(days=1)
    if re.search(r"\byesterday\b", q):
        windows.append(("yesterday", today - timedelta(days=1), today - timedelta(days=1)))
    if re.search(r"\btoday\b", q):
        windows.append(("today", today, today))
    if re.search(r"\bthis week\b", q):
        windows.append(("this week", monday, today))
    if re.search(r"\b(last|previous|past) week\b", q):
        windows.append(("last week", monday - timedelta(days=7), monday - timedelta(days=1)))
    if re.search(r"\bthis month\b", q):
        windows.append(("this month", first_of_month, today))
    if re.search(r"\b(last|previous|past) month\b", q):
        windows.append(("last month", last_month_end.replace(day=1), last_month_end))
    if not windows:
        return None, ""
    ids = [m["meeting_id"] for m, d in dated if any(start <= d <= end for _, start, end in windows)]
    label = " / ".join(f"{name} ({start:%d %b}–{end:%d %b %Y})" for name, start, end in windows)
    return ids, label


def _catalogue(meetings: list[dict], ids: list[str] | None) -> str:
    chosen = [m for m in meetings if m.get("meeting_id") != "legacy" and (ids is None or m["meeting_id"] in ids)]
    lines = [
        f"- {m.get('title') or 'Untitled meeting'} ({m.get('meeting_date') or str(m.get('processed_at', ''))[:10]})"
        for m in chosen[:CATALOGUE_MAX_MEETINGS]
    ]
    return "\n".join(lines)


# ── Chain ───────────────────────────────────────────────────────────────────────


def _to_messages(history: Sequence[Any] | None) -> list[BaseMessage]:
    """Accept [{'role','content'}] dicts, (role, content) tuples or messages; keep the last turns."""
    messages: list[BaseMessage] = []
    for item in history or []:
        if isinstance(item, BaseMessage):
            messages.append(item)
            continue
        if isinstance(item, dict):
            role, content = item.get("role"), item.get("content", "")
        else:
            role, content = item
        if not content:
            continue
        content = str(content)[:HISTORY_MESSAGE_MAX_CHARS]
        messages.append(HumanMessage(content) if role == "user" else AIMessage(content))
    messages = messages[-MAX_HISTORY_MESSAGES:]
    while messages and not isinstance(messages[0], HumanMessage):  # must start with a user turn
        messages.pop(0)
    return messages


def _normalize_input(inputs: Any) -> dict[str, Any]:
    if isinstance(inputs, str):
        return {"question": inputs, "chat_history": [], "current_meeting_id": None}
    return {
        "question": str(inputs.get("question", "")).strip(),
        "chat_history": _to_messages(inputs.get("chat_history")),
        "current_meeting_id": inputs.get("current_meeting_id"),
    }


def _retrieval_query(inputs: dict[str, Any]) -> str:
    """Add the previous user question so short follow-ups retrieve the right context."""
    previous = [m.content for m in inputs["chat_history"] if isinstance(m, HumanMessage)]
    return f"{previous[-1]}\n{inputs['question']}" if previous else inputs["question"]


def _build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [("system", RAG_SYSTEM_PROMPT), MessagesPlaceholder("chat_history"), ("human", "{question}")]
    ).partial(not_found=NOT_FOUND_MESSAGE)


def meeting_overview(meeting_id: str) -> str:
    """The stored summary and extracted lists of one meeting (empty if not saved)."""
    record = load_meeting(meeting_id)
    if record is None or not record.summary:
        return ""
    parts = [
        f"Meeting: {record.display_name}" + (f" (AI title: {record.title})" if record.title != record.display_name else ""),
        f"Summary:\n{record.summary}",
        f"Action items:\n{record.action_items}",
        f"Key decisions:\n{record.key_decisions}",
        f"Open questions:\n{record.open_questions}",
    ]
    text = "\n\n".join(part.strip() for part in parts)
    return f"Meeting overview [Overview]:\n{text[:OVERVIEW_MAX_CHARS]}\n\n"


def _chronological(doc: Document) -> tuple:
    meta = doc.metadata or {}
    return (meta.get("meeting_ts") or 0, meta.get("meeting_id", ""), meta.get("chunk_index") or 0)


def retrieve_context(vector_store, meeting_id: str | None, inputs: dict[str, Any], k: int = DEFAULT_K) -> dict[str, Any]:
    """MMR-search ChromaDB for the question and keep the chunks that fit the context budget.

    Restricted to ``meeting_id`` by a metadata filter (no other meeting can leak in).
    Shared by the LCEL chain and the streaming chat so both retrieve identically.
    """
    scope_note, scope_label, ids = "", "", [meeting_id] if meeting_id else None
    if meeting_id is None:
        meetings = list_indexed_meetings(vector_store)
        resolved, label = resolve_time_scope(inputs["question"], meetings, inputs.get("current_meeting_id"))
        if resolved is not None:
            ids, scope_label = resolved, label
            scope_note = f"Question scope: {label}. Meetings in scope:\n{_catalogue(meetings, ids) or '- none'}\n"
        else:
            catalogue = _catalogue(meetings, None)
            scope_note = f"Stored meetings (newest first):\n{catalogue}\n" if catalogue else ""
    empty = {"docs": [], "overview": "", "scope_note": scope_note, "scope_label": scope_label}
    if ids == []:
        return {**inputs, **empty, "no_meetings_in_scope": True}

    docs = vector_store.max_marginal_relevance_search(
        _retrieval_query(inputs), k=k, fetch_k=max(FETCH_K, k), lambda_mult=MMR_LAMBDA, filter=meeting_filter(ids)
    )
    overview = meeting_overview(meeting_id) if meeting_id and docs else ""
    # Keep the retrieved context inside the model's context window (most relevant first).
    history_tokens = sum(estimate_tokens(str(m.content)) for m in inputs["chat_history"])
    budget = input_char_budget(
        reserve_tokens=history_tokens + estimate_tokens(scope_note + overview + inputs["question"])
    )
    kept, used = [], 0
    for doc in docs:
        if kept and used + len(doc.page_content) > budget:
            break
        kept.append(doc)
        used += len(doc.page_content)
    kept.sort(key=_chronological)  # read in meeting order; [Source N] numbers follow it
    return {**inputs, **empty, "docs": kept, "overview": overview, "no_meetings_in_scope": False}


def _prompt_inputs(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "context": format_docs(state["docs"]),
        "question": state["question"],
        "chat_history": state["chat_history"],
        "scope_note": state["scope_note"],
        "overview": state.get("overview", ""),
    }


def build_rag_chain_from_store(vector_store, *, meeting_id: str | None = None, k: int = DEFAULT_K) -> Runnable:
    """LCEL RAG chain over the persistent vector store.

    Input: ``str`` or ``{"question", "chat_history", "current_meeting_id"}``.
    Output: ``{"answer": str, "sources": [...], "scope_note": str}``.
    """
    answer_chain = RunnableLambda(_prompt_inputs) | build_text_chain(_build_prompt())
    generate = RunnableBranch(
        (
            lambda x: x["no_meetings_in_scope"],
            RunnableLambda(lambda x: f"No stored meetings match that time period: {x['scope_label']}."),
        ),
        (lambda x: not x["docs"], RunnableLambda(lambda _: NOT_FOUND_MESSAGE)),
        answer_chain,
    )
    return (
        RunnableLambda(_normalize_input)
        | RunnableLambda(lambda inputs: retrieve_context(vector_store, meeting_id, inputs, k))
        | RunnablePassthrough.assign(answer=generate)
        | RunnableLambda(
            lambda x: {"answer": x["answer"], "sources": docs_to_sources(x["docs"]), "scope_note": x["scope_note"]}
        )
    )


def build_rag_chain(transcript: str, **metadata: Any) -> Runnable:
    """Index a transcript (skipped if already indexed) and return a chain scoped to it."""
    meeting_id = metadata.pop("meeting_id", None) or compute_meeting_id(transcript)
    vector_store = build_vector_store(transcript, meeting_id=meeting_id, **metadata)
    return build_rag_chain_from_store(vector_store, meeting_id=meeting_id)


def load_rag_chain(meeting_id: str | None = None, persist_directory: str | None = None) -> Runnable:
    """RAG chain over the persisted store — one meeting, or all meetings if ``meeting_id`` is None."""
    return build_rag_chain_from_store(load_vector_store(persist_directory), meeting_id=meeting_id)


def ask_question(
    rag_chain: Runnable,
    question: str,
    chat_history: Sequence[Any] | None = None,
    current_meeting_id: str | None = None,
) -> dict[str, Any]:
    """Ask a question; returns ``{"answer", "sources", "scope_note"}``."""
    question = (question or "").strip()
    if not question:
        raise ValueError("Please enter a question.")
    return invoke_with_retry(
        rag_chain,
        {"question": question, "chat_history": list(chat_history or []), "current_meeting_id": current_meeting_id},
        task="Answering your question",
    )


def stream_answer(
    question: str,
    *,
    meeting_id: str | None = None,
    chat_history: Sequence[Any] | None = None,
    current_meeting_id: str | None = None,
    vector_store=None,
    k: int = DEFAULT_K,
) -> tuple[dict[str, Any], Iterator[str]]:
    """Retrieve context, then stream the answer word by word (for the chat UI).

    Returns ``(info, chunks)``: ``info`` has ``sources`` and ``scope_note`` (known before
    the answer starts), ``chunks`` yields the answer text. Same retrieval, prompt and
    no-hallucination rules as :func:`ask_question`.
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("Please enter a question.")
    store = vector_store or load_vector_store()
    state = retrieve_context(
        store,
        meeting_id,
        _normalize_input({"question": question, "chat_history": list(chat_history or []),
                          "current_meeting_id": current_meeting_id}),
        k,
    )
    info = {"sources": docs_to_sources(state["docs"]), "scope_note": state["scope_note"]}
    if state["no_meetings_in_scope"]:
        return info, iter([f"No stored meetings match that time period: {state['scope_label']}."])
    if not state["docs"]:
        return info, iter([NOT_FOUND_MESSAGE])
    chain = build_stream_chain(_build_prompt())
    return info, stream_with_retry(chain, _prompt_inputs(state), task="Answering your question")
