"""ChromaDB vector store and LCEL RAG chain (isolated temporary collection)."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

import core.vector_store as vs
from core.rag_engine import (
    NOT_FOUND_MESSAGE,
    ask_question,
    build_rag_chain,
    build_rag_chain_from_store,
    format_docs,
    load_rag_chain,
)
from core.vector_store import (
    build_vector_store,
    compute_meeting_id,
    get_retriever,
    list_indexed_meetings,
    load_vector_store,
)
from tests.conftest import SAMPLE_TRANSCRIPT

OTHER_TRANSCRIPT = (
    "Hiring sync. We agreed to hire two backend engineers in Berlin. "
    "Maria owns the job descriptions and will publish them by Monday."
)


def _count(store) -> int:
    return len(store.get(include=[])["ids"])


class TestVectorStore:
    def test_index_with_metadata(self, fake_embeddings):
        store = build_vector_store(
            SAMPLE_TRANSCRIPT, title="Product Sync", source="meeting.mp4", language="english"
        )
        data = store.get(include=["metadatas"])
        meeting_id = compute_meeting_id(SAMPLE_TRANSCRIPT)
        assert data["ids"][0] == f"{meeting_id}-00000"
        meta = data["metadatas"][0]
        assert meta["meeting_id"] == meeting_id
        assert meta["title"] == "Product Sync"
        assert meta["source"] == "meeting.mp4"
        assert meta["language"] == "english"
        assert meta["processed_at"]
        assert {m["chunk_index"] for m in data["metadatas"]} == set(range(len(data["ids"])))

    def test_reprocessing_does_not_duplicate(self, fake_embeddings):
        store = build_vector_store(SAMPLE_TRANSCRIPT)
        first = _count(store)
        build_vector_store(SAMPLE_TRANSCRIPT)
        build_vector_store("  " + SAMPLE_TRANSCRIPT.upper() + "  ")  # same content, normalised id
        assert _count(store) == first

    def test_persists_and_reloads(self, fake_embeddings):
        count = _count(build_vector_store(SAMPLE_TRANSCRIPT))
        vs._open_store.cache_clear()
        assert _count(load_vector_store()) == count

    def test_existing_legacy_data_is_kept(self, fake_embeddings):
        store = load_vector_store()
        store.add_documents([Document(page_content="old chunk", metadata={"chunk_index": 0})], ids=["legacy-1"])
        build_vector_store(SAMPLE_TRANSCRIPT, title="New")
        meetings = {m["meeting_id"]: m for m in list_indexed_meetings(store)}
        assert vs.LEGACY_MEETING_ID in meetings
        assert compute_meeting_id(SAMPLE_TRANSCRIPT) in meetings
        assert "legacy-1" in store.get(include=[])["ids"]

    def test_retriever_scoped_to_meeting(self, fake_embeddings):
        store = build_vector_store(SAMPLE_TRANSCRIPT, title="Product Sync")
        build_vector_store(OTHER_TRANSCRIPT, title="Hiring Sync")
        other_id = compute_meeting_id(OTHER_TRANSCRIPT)
        docs = get_retriever(store, k=10, meeting_id=other_id).invoke("hire engineers")
        assert docs and all(d.metadata["meeting_id"] == other_id for d in docs)
        all_docs = get_retriever(store, k=50).invoke("anything")
        assert {d.metadata["meeting_id"] for d in all_docs} == {other_id, compute_meeting_id(SAMPLE_TRANSCRIPT)}

    def test_rejects_empty_transcript(self, fake_embeddings):
        with pytest.raises(ValueError):
            build_vector_store("   ")


class TestRagChain:
    def test_answer_with_sources_and_grounded_prompt(self, fake_embeddings, fake_llm):
        llm = fake_llm("Priya will prepare the press kit by Friday [Source 1].")
        chain = build_rag_chain(SAMPLE_TRANSCRIPT, title="Product Sync")
        result = ask_question(chain, "Who prepares the press kit?")
        assert result["answer"].startswith("Priya")
        assert result["sources"] and result["sources"][0]["title"] == "Product Sync"
        system = str(llm.calls[0][0].content)
        assert "[Source 1] Meeting: Product Sync" in system
        assert NOT_FOUND_MESSAGE in system
        assert str(llm.calls[0][-1].content) == "Who prepares the press kit?"

    def test_follow_up_uses_history(self, fake_embeddings, fake_llm):
        llm = fake_llm("Rahul.")
        chain = build_rag_chain(SAMPLE_TRANSCRIPT)
        history = [
            {"role": "user", "content": "Who prepares the press kit?"},
            {"role": "assistant", "content": "Priya."},
        ]
        ask_question(chain, "And the login bug?", history)
        roles = [type(m).__name__ for m in llm.calls[0]]
        assert roles == ["SystemMessage", "HumanMessage", "AIMessage", "HumanMessage"]
        assert str(llm.calls[0][-1].content) == "And the login bug?"

    def test_no_documents_returns_not_found_without_llm_call(self, fake_embeddings, fake_llm):
        llm = fake_llm("should not be used")
        chain = build_rag_chain_from_store(load_vector_store(), meeting_id="0" * 16)
        result = ask_question(chain, "What was decided?")
        assert result == {"answer": NOT_FOUND_MESSAGE, "sources": []}
        assert llm.calls == []

    def test_load_rag_chain_regression(self, fake_embeddings, fake_llm):
        """The old load_rag_chain() called get_retriever() without a vector store."""
        fake_llm("answer")
        build_vector_store(SAMPLE_TRANSCRIPT)
        for meeting_id in (compute_meeting_id(SAMPLE_TRANSCRIPT), None):
            result = ask_question(load_rag_chain(meeting_id), "launch date?")
            assert result["answer"] == "answer"

    def test_plain_string_input_still_supported(self, fake_embeddings, fake_llm):
        fake_llm("ok")
        chain = build_rag_chain(SAMPLE_TRANSCRIPT)
        assert chain.invoke("launch date?")["answer"] == "ok"

    def test_empty_question_rejected(self, fake_embeddings, fake_llm):
        fake_llm("x")
        with pytest.raises(ValueError):
            ask_question(build_rag_chain(SAMPLE_TRANSCRIPT), "  ")

    def test_format_docs_labels(self):
        text = format_docs(
            [Document(page_content="hello", metadata={"title": "T", "chunk_index": 2, "processed_at": "2026-11-05T10:00:00"})]
        )
        assert text.startswith("[Source 1] Meeting: T | Processed: 2026-11-05 | Segment 3")


@pytest.mark.slow
def test_real_minilm_embeddings_semantic_search():
    """Uses the real all-MiniLM-L6-v2 model (downloaded/cached by sentence-transformers)."""
    vs._open_store.cache_clear()
    store = build_vector_store(SAMPLE_TRANSCRIPT + " " + OTHER_TRANSCRIPT, title="Mixed")
    docs = get_retriever(store, k=1).invoke("Who is recruiting engineers in Germany?")
    assert "Berlin" in docs[0].page_content
    vs._open_store.cache_clear()
