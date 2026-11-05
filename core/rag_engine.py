"""Retrieval-Augmented Q&A over meeting transcripts (LangChain LCEL + Gemini).

Flow: question (+ recent chat history) → retriever (ChromaDB) → formatted
context with source labels → grounded prompt → Gemini → answer + sources.

* Retrieval can be scoped to one meeting (``meeting_id``) or all meetings.
* Follow-up questions: the previous user question is added to the retrieval
  query and the last few turns are passed to the prompt, without spending an
  extra LLM call on question rewriting.
* If nothing relevant is retrieved, the chain answers "not found" without
  calling Gemini at all.
"""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)

from core.llm import build_text_chain, invoke_with_retry
from core.vector_store import build_vector_store, get_retriever, load_vector_store

NOT_FOUND_MESSAGE = "I could not find this information in the meeting transcript."
DEFAULT_K = 5
MAX_HISTORY_MESSAGES = 6  # last 3 question/answer pairs

RAG_SYSTEM_PROMPT = """You are a meeting assistant answering questions about recorded meetings.

Answer ONLY from the transcript excerpts in the context below.
- If the context does not contain the answer, reply exactly: "{not_found}"
- Never invent names, owners, deadlines, numbers or decisions. If the excerpts are only partially relevant, answer what they support and say what is missing.
- Distinguish clearly between what was decided and what was only suggested or discussed.
- When the excerpts come from more than one meeting, say which meeting each point comes from.
- Cite the supporting excerpts inline as [Source N].
- Be concise; use bullet points for lists.

Use the conversation history only to understand what a follow-up question refers to — not as a source of facts.

Context (transcript excerpts):
{context}"""


def format_docs(docs: Sequence[Document]) -> str:
    """Render retrieved chunks with source labels the model can cite."""
    blocks = []
    for number, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        header = f"[Source {number}] Meeting: {meta.get('title') or 'Untitled meeting'}"
        if meta.get("processed_at"):
            header += f" | Processed: {str(meta['processed_at'])[:10]}"
        if meta.get("chunk_index") is not None:
            header += f" | Segment {int(meta['chunk_index']) + 1}"
        blocks.append(f"{header}\n{doc.page_content.strip()}")
    return "\n\n".join(blocks)


def docs_to_sources(docs: Sequence[Document]) -> list[dict[str, Any]]:
    return [
        {
            "number": number,
            "meeting_id": (doc.metadata or {}).get("meeting_id", ""),
            "title": (doc.metadata or {}).get("title", "Untitled meeting"),
            "chunk_index": (doc.metadata or {}).get("chunk_index"),
            "excerpt": doc.page_content.strip(),
        }
        for number, doc in enumerate(docs, start=1)
    ]


def _to_messages(history: Sequence[Any] | None) -> list[BaseMessage]:
    """Accept [{'role','content'}] dicts, (role, content) tuples or messages."""
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
        messages.append(HumanMessage(content) if role == "user" else AIMessage(content))
    messages = messages[-MAX_HISTORY_MESSAGES:]
    # Gemini requires the conversation to start with a user turn.
    while messages and not isinstance(messages[0], HumanMessage):
        messages.pop(0)
    return messages


def _normalize_input(inputs: Any) -> dict[str, Any]:
    if isinstance(inputs, str):
        return {"question": inputs, "chat_history": []}
    question = str(inputs.get("question", "")).strip()
    return {"question": question, "chat_history": _to_messages(inputs.get("chat_history"))}


def _retrieval_query(inputs: dict[str, Any]) -> str:
    """Add the previous user question so short follow-ups retrieve the right context."""
    previous = [m.content for m in inputs["chat_history"] if isinstance(m, HumanMessage)]
    if previous:
        return f"{previous[-1]}\n{inputs['question']}"
    return inputs["question"]


def _build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    ).partial(not_found=NOT_FOUND_MESSAGE)


def build_rag_chain_from_store(vector_store, *, meeting_id: str | None = None, k: int = DEFAULT_K) -> Runnable:
    """LCEL RAG chain over an existing vector store.

    Input: ``str`` or ``{"question": str, "chat_history": [...]}``.
    Output: ``{"answer": str, "sources": [...]}``.
    """
    retriever = get_retriever(vector_store, k=k, meeting_id=meeting_id)
    answer_chain = (
        {
            "context": lambda x: format_docs(x["docs"]),
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
        }
        | build_text_chain(_build_prompt())
    )
    generate = RunnableBranch(
        (lambda x: not x["docs"], RunnableLambda(lambda _: NOT_FOUND_MESSAGE)),
        answer_chain,
    )
    return (
        RunnableLambda(_normalize_input)
        | RunnablePassthrough.assign(docs=RunnableLambda(_retrieval_query) | retriever)
        | RunnablePassthrough.assign(answer=generate)
        | RunnableLambda(lambda x: {"answer": x["answer"], "sources": docs_to_sources(x["docs"])})
    )


def build_rag_chain(transcript: str, **metadata: Any) -> Runnable:
    """Index a transcript (skipped if already indexed) and return a chain scoped to it."""
    from core.vector_store import compute_meeting_id

    meeting_id = metadata.pop("meeting_id", None) or compute_meeting_id(transcript)
    vector_store = build_vector_store(transcript, meeting_id=meeting_id, **metadata)
    return build_rag_chain_from_store(vector_store, meeting_id=meeting_id)


def load_rag_chain(meeting_id: str | None = None, persist_directory: str | None = None) -> Runnable:
    """RAG chain over the persisted store — one meeting, or all meetings if ``meeting_id`` is None."""
    vector_store = load_vector_store(persist_directory)
    return build_rag_chain_from_store(vector_store, meeting_id=meeting_id)


def ask_question(rag_chain: Runnable, question: str, chat_history: Sequence[Any] | None = None) -> dict[str, Any]:
    """Ask a question; returns ``{"answer": str, "sources": [...]}``."""
    question = (question or "").strip()
    if not question:
        raise ValueError("Please enter a question.")
    return invoke_with_retry(
        rag_chain,
        {"question": question, "chat_history": list(chat_history or [])},
        task="Answering your question",
    )
