"""ChromaDB storage workflow and the LCEL RAG chain (isolated temporary collection)."""

from __future__ import annotations

from datetime import date

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
    resolve_time_scope,
    stream_answer,
)
from core.vector_store import (
    backup_vector_store,
    build_vector_store,
    compute_meeting_id,
    get_retriever,
    list_indexed_meetings,
    load_vector_store,
    split_for_indexing,
)
from tests.conftest import FIXTURES, SAMPLE_TRANSCRIPT

PRODUCT = (FIXTURES / "meeting_product_sync.txt").read_text(encoding="utf-8")
BUDGET = (FIXTURES / "meeting_budget_review.txt").read_text(encoding="utf-8")
OTHER_TRANSCRIPT = (
    "Hiring sync. We agreed to hire two backend engineers in Berlin. "
    "Maria owns the job descriptions and will publish them by Monday."
)


def _ids(store, meeting_id=None) -> list[str]:
    where = {"meeting_id": meeting_id} if meeting_id else None
    return store.get(where=where, include=[])["ids"]


class TestTextChunking:
    def test_chunks_are_bounded_and_overlap(self):
        chunks = split_for_indexing(PRODUCT)
        assert len(chunks) > 5
        assert all(len(c.page_content) <= vs.CHUNK_SIZE for c in chunks)
        starts = [c.metadata["start_index"] for c in chunks]
        assert starts == sorted(starts)
        for chunk in chunks:  # offsets point at the real text
            start = chunk.metadata["start_index"]
            assert PRODUCT[start : start + len(chunk.page_content)] == chunk.page_content

    def test_chunks_prefer_sentence_boundaries(self):
        chunks = split_for_indexing(PRODUCT)
        ends_on_sentence = sum(c.page_content.rstrip().endswith((".", "?", "!")) for c in chunks[:-1])
        assert ends_on_sentence / (len(chunks) - 1) >= 0.8

    def test_audio_chunks_and_text_chunks_are_independent(self):
        """Text chunk size is set here; audio chunking (10-minute WAVs) is elsewhere."""
        from utils.audio_processor import DEFAULT_CHUNK_MINUTES

        assert DEFAULT_CHUNK_MINUTES == 10 and vs.CHUNK_SIZE == 800


class TestVectorStore:
    def test_index_with_full_metadata(self, fake_embeddings):
        segments = [
            {"start": 0.0, "end": 30.0, "audio_chunk": 0, "char_start": 0, "char_end": 600},
            {"start": 30.0, "end": 65.5, "audio_chunk": 0, "char_start": 601, "char_end": 1400},
            {"start": 600.0, "end": 640.0, "audio_chunk": 1, "char_start": 1401, "char_end": len(PRODUCT)},
        ]
        store = build_vector_store(
            PRODUCT, title="Product Sync", source="sync.mp4", source_type="upload",
            language="english", meeting_date="2026-03-10", segments=segments,
        )
        meeting_id = compute_meeting_id(PRODUCT)
        data = store.get(include=["metadatas"])
        assert data["ids"][0] == f"{meeting_id}-00000"
        first = min(data["metadatas"], key=lambda m: m["chunk_index"])
        assert first["meeting_id"] == meeting_id and first["chunk_id"] == f"{meeting_id}-00000"
        assert first["title"] == "Product Sync" and first["source"] == "sync.mp4"
        assert first["source_type"] == "upload" and first["language"] == "english"
        assert first["meeting_date"] == "2026-03-10" and first["meeting_ts"] > 0
        assert first["start_seconds"] == 0.0 and first["audio_chunk"] == 0
        assert first["embedding_model"] == "all-MiniLM-L6-v2" and first["schema_version"] == 2
        last = max(data["metadatas"], key=lambda m: m["chunk_index"])
        assert last["audio_chunk"] == 1 and last["end_seconds"] == 640.0
        assert sorted(m["chunk_index"] for m in data["metadatas"]) == list(range(len(data["ids"])))

    def test_reprocessing_does_not_duplicate(self, fake_embeddings):
        store = build_vector_store(PRODUCT)
        first = len(_ids(store))
        build_vector_store(PRODUCT)
        build_vector_store("  " + PRODUCT.upper() + "  ")  # same content → same meeting id
        assert len(_ids(store)) == first

    def test_replace_updates_only_that_meeting(self, fake_embeddings):
        store = build_vector_store(PRODUCT, title="Old title")
        build_vector_store(BUDGET, title="Budget")
        product_id, budget_id = compute_meeting_id(PRODUCT), compute_meeting_id(BUDGET)
        budget_ids = _ids(store, budget_id)

        # Simulate a stale extra chunk from an older indexing run of the same meeting.
        store.add_documents([Document(page_content="stale", metadata={"meeting_id": product_id})], ids=[f"{product_id}-99999"])
        build_vector_store(PRODUCT, title="New title", replace=True)

        product_meta = store.get(where={"meeting_id": product_id}, include=["metadatas"])
        assert f"{product_id}-99999" not in product_meta["ids"]
        assert {m["title"] for m in product_meta["metadatas"]} == {"New title"}
        assert _ids(store, budget_id) == budget_ids  # other meeting untouched

    def test_persists_across_restart(self, fake_embeddings):
        count = len(_ids(build_vector_store(PRODUCT)))
        build_vector_store(BUDGET)
        vs._open_store.cache_clear()  # simulate an application restart
        reopened = load_vector_store()
        assert len(_ids(reopened, compute_meeting_id(PRODUCT))) == count
        assert {m["meeting_id"] for m in list_indexed_meetings(reopened)} == {
            compute_meeting_id(PRODUCT), compute_meeting_id(BUDGET)
        }

    def test_existing_legacy_data_is_kept(self, fake_embeddings):
        store = load_vector_store()
        store.add_documents([Document(page_content="old chunk", metadata={"chunk_index": 0})], ids=["legacy-1"])
        build_vector_store(PRODUCT, title="New")
        meetings = {m["meeting_id"] for m in list_indexed_meetings(store)}
        assert vs.LEGACY_MEETING_ID in meetings and "legacy-1" in _ids(store)

    def test_retriever_scoped_to_meeting(self, fake_embeddings):
        store = build_vector_store(PRODUCT, title="Product Sync")
        build_vector_store(OTHER_TRANSCRIPT, title="Hiring Sync")
        other_id = compute_meeting_id(OTHER_TRANSCRIPT)
        docs = get_retriever(store, k=10, meeting_id=other_id).invoke("hire engineers")
        assert docs and all(d.metadata["meeting_id"] == other_id for d in docs)

    def test_backup_copies_database(self, fake_embeddings, tmp_path, monkeypatch):
        monkeypatch.setattr(vs, "PROJECT_ROOT", tmp_path)
        build_vector_store(PRODUCT)
        backup = backup_vector_store()
        assert backup is not None and (backup / "chroma.sqlite3").exists()

    def test_rejects_empty_transcript(self, fake_embeddings):
        with pytest.raises(ValueError):
            build_vector_store("   ")


class TestTimeScope:
    MEETINGS = [
        {"meeting_id": "c" * 16, "meeting_date": "2026-03-10", "title": "Product sync"},
        {"meeting_id": "b" * 16, "meeting_date": "2026-03-03", "title": "Budget review"},
        {"meeting_id": "a" * 16, "meeting_date": "2026-02-10", "title": "Kickoff"},
        {"meeting_id": "legacy", "title": "Earlier meeting"},
    ]
    TODAY = date(2026, 3, 11)  # a Wednesday

    def test_no_time_reference(self):
        assert resolve_time_scope("What was decided about search?", self.MEETINGS, today=self.TODAY) == (None, "")

    def test_previous_meeting_relative_to_current(self):
        ids, _ = resolve_time_scope("What did we discuss in the previous meeting?", self.MEETINGS, "c" * 16, self.TODAY)
        assert ids == ["b" * 16]

    def test_previous_meeting_without_current(self):
        ids, _ = resolve_time_scope("Summarise the last meeting", self.MEETINGS, today=self.TODAY)
        assert ids == ["c" * 16]

    def test_last_week(self):
        ids, label = resolve_time_scope("What decisions were made last week?", self.MEETINGS, today=self.TODAY)
        assert ids == ["b" * 16] and "last week" in label

    def test_this_month_and_empty_period(self):
        assert set(resolve_time_scope("this month's action items", self.MEETINGS, today=self.TODAY)[0]) == {"c" * 16, "b" * 16}
        assert resolve_time_scope("anything from yesterday?", self.MEETINGS, today=self.TODAY)[0] == ["c" * 16]
        assert resolve_time_scope("what happened today?", self.MEETINGS, today=self.TODAY)[0] == []


class TestRagChain:
    def test_answer_with_sources_and_grounded_prompt(self, fake_embeddings, fake_llm):
        llm = fake_llm("Arjun owns the image cache fix [Source 1].")
        chain = build_rag_chain(PRODUCT, title="Product Sync", meeting_date="2026-03-10")
        result = ask_question(chain, "Who owns the image cache fix?")
        assert result["answer"].startswith("Arjun")
        assert result["sources"] and result["sources"][0]["title"] == "Product Sync"
        assert result["sources"][0]["meeting_date"] == "2026-03-10"
        system = str(llm.calls[0][0].content)
        assert "[Source 1] Meeting: Product Sync | Date: 2026-03-10" in system
        assert NOT_FOUND_MESSAGE in system
        assert str(llm.calls[0][-1].content) == "Who owns the image cache fix?"

    def test_retrieval_uses_stored_chunks_not_memory(self, fake_embeddings, fake_llm):
        """The chain is built from the persisted store after a restart, with no transcript in memory."""
        fake_llm("ok")
        build_vector_store(PRODUCT, title="Product Sync")
        vs._open_store.cache_clear()
        result = ask_question(load_rag_chain(compute_meeting_id(PRODUCT)), "image cache fix")
        assert result["sources"] and all(s["meeting_id"] == compute_meeting_id(PRODUCT) for s in result["sources"])

    def test_context_stays_within_budget(self, fake_embeddings, fake_llm, monkeypatch):
        import core.llm

        monkeypatch.setattr(core.llm, "MAX_INPUT_TOKENS", 2048)
        llm = fake_llm("ok")
        chain = build_rag_chain_from_store(build_vector_store(PRODUCT), meeting_id=compute_meeting_id(PRODUCT), k=10)
        ask_question(chain, "What was discussed?")
        system = str(llm.calls[0][0].content)
        from core.llm import input_char_budget

        assert len(system) < input_char_budget() + 2_000

    def test_follow_up_uses_history(self, fake_embeddings, fake_llm):
        llm = fake_llm("By Wednesday.")
        chain = build_rag_chain(PRODUCT)
        history = [
            {"role": "user", "content": "Who owns the image cache fix?"},
            {"role": "assistant", "content": "Arjun."},
        ]
        ask_question(chain, "And by when?", history)
        roles = [type(m).__name__ for m in llm.calls[0]]
        assert roles == ["SystemMessage", "HumanMessage", "AIMessage", "HumanMessage"]

    def test_all_meetings_previous_meeting_is_filtered(self, fake_embeddings, fake_llm):
        llm = fake_llm("answer")
        build_vector_store(BUDGET, title="Budget review", meeting_date="2026-03-03")
        build_vector_store(PRODUCT, title="Product sync", meeting_date="2026-03-10")
        result = ask_question(
            load_rag_chain(None), "What was decided in the previous meeting?", current_meeting_id=compute_meeting_id(PRODUCT)
        )
        assert result["sources"] and {s["meeting_id"] for s in result["sources"]} == {compute_meeting_id(BUDGET)}
        assert "Budget review" in str(llm.calls[0][0].content)

    def test_time_period_without_meetings_answers_without_llm(self, fake_embeddings, fake_llm):
        llm = fake_llm("should not be used")
        build_vector_store(PRODUCT, title="Product sync", meeting_date="2020-01-01")
        result = ask_question(load_rag_chain(None), "What did we decide yesterday?")
        assert result["answer"].startswith("No stored meetings match") and llm.calls == []

    def test_no_documents_returns_not_found_without_llm_call(self, fake_embeddings, fake_llm):
        llm = fake_llm("should not be used")
        chain = build_rag_chain_from_store(load_vector_store(), meeting_id="0" * 16)
        result = ask_question(chain, "What was decided?")
        assert result["answer"] == NOT_FOUND_MESSAGE and result["sources"] == []
        assert llm.calls == []

    def test_plain_string_input_still_supported(self, fake_embeddings, fake_llm):
        fake_llm("ok")
        assert build_rag_chain(SAMPLE_TRANSCRIPT).invoke("launch date?")["answer"] == "ok"

    def test_empty_question_rejected(self, fake_embeddings, fake_llm):
        fake_llm("x")
        with pytest.raises(ValueError):
            ask_question(build_rag_chain(SAMPLE_TRANSCRIPT), "  ")

    def test_format_docs_labels(self):
        text = format_docs(
            [Document(page_content="hello", metadata={"title": "T", "meeting_date": "2026-11-05", "start_seconds": 65, "end_seconds": 130})]
        )
        assert text.startswith("[Source 1] Meeting: T | Date: 2026-11-05 | Time: ~01:05–02:10")


@pytest.mark.slow
def test_real_minilm_embeddings_semantic_search():
    """Real all-MiniLM-L6-v2: a paraphrased question finds the right meeting and chunk."""
    vs._open_store.cache_clear()
    store = build_vector_store(PRODUCT, title="Product sync")
    build_vector_store(BUDGET, title="Budget review")
    docs = store.similarity_search("Why did the cloud bill go up?", k=1)
    assert docs[0].metadata["meeting_id"] == compute_meeting_id(BUDGET)
    assert "search" in docs[0].page_content.lower()
    vs._open_store.cache_clear()


class TestStreaming:
    def test_stream_answer_uses_retrieval_and_streams(self, fake_embeddings, fake_llm):
        llm = fake_llm("Arjun owns the image cache fix [Source 1].")
        build_vector_store(PRODUCT, title="Product Sync")
        info, chunks = stream_answer("Who owns the image cache fix?", meeting_id=compute_meeting_id(PRODUCT))
        assert info["sources"] and info["sources"][0]["title"] == "Product Sync"
        assert "".join(chunks).startswith("Arjun")
        assert "[Source 1] Meeting: Product Sync" in str(llm.calls[0][0].content)

    def test_stream_not_found_without_llm(self, fake_embeddings, fake_llm):
        llm = fake_llm("unused")
        info, chunks = stream_answer("Anything?", meeting_id="0" * 16)
        assert "".join(chunks) == NOT_FOUND_MESSAGE and info["sources"] == [] and llm.calls == []

    def test_stream_rejects_empty_question(self, fake_embeddings, fake_llm):
        fake_llm("x")
        with pytest.raises(ValueError):
            stream_answer("   ")
