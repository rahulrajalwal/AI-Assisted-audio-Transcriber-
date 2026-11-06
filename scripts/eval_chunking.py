"""Compare transcript text-chunking settings for retrieval quality (all-MiniLM-L6-v2).

For each (chunk_size, overlap) setting, both fixture meetings are chunked and embedded
in a throw-away in-memory Chroma collection, then a set of questions with known
answer phrases is asked. A hit means a retrieved chunk contains the answer phrase.

Usage:  python scripts/eval_chunking.py
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

QUESTIONS = [  # (question, phrase that must appear in a retrieved chunk)
    ("When will version 2.4 be released?", "twenty-seventh of March"),
    ("What is the crash-free rate on Android 10?", "ninety-four percent"),
    ("Who is fixing the image cache problem?", "you own the image cache fix"),
    ("Where should the location permission prompt appear?", "first opens the map"),
    ("How did users do with the new onboarding flow?", "nine out of twelve"),
    ("Is account creation mandatory for the premium trial?", "mandatory for the premium trial"),
    ("How much is the paid social campaign?", "fifteen thousand dollars"),
    ("Why do we need a support contractor?", "thirty percent"),
    ("Why did the cloud bill increase?", "never turned on autoscaling"),
    ("How much does the Insightly contract cost?", "twenty-four thousand dollars"),
    ("Was the mobile engineer hire approved?", "mobile engineer hire is approved"),
    ("Who presents at the partner webinar?", "who presents at the webinar"),
]
SETTINGS = [(500, 50), (800, 120), (1000, 150)]


def main() -> int:
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from core.vector_store import CHUNK_SEPARATORS, get_embeddings

    embeddings = get_embeddings()
    fixtures = ROOT / "tests" / "fixtures"
    meetings = {
        "product": (fixtures / "meeting_product_sync.txt").read_text(encoding="utf-8"),
        "budget": (fixtures / "meeting_budget_review.txt").read_text(encoding="utf-8"),
    }
    print(f"{'setting':>12} {'chunks':>6} {'hit@1':>6} {'hit@3':>6}  avg chars")
    for size, overlap in SETTINGS:
        old_style = (size, overlap) == (500, 50)  # the previous default splitter (no sentence separators)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            **({} if old_style else {"separators": CHUNK_SEPARATORS, "keep_separator": "end"}),
        )
        docs = [
            Document(page_content=chunk, metadata={"meeting": name})
            for name, text in meetings.items()
            for chunk in splitter.split_text(text)
        ]
        store = Chroma(collection_name=f"eval_{uuid.uuid4().hex[:8]}", embedding_function=embeddings)
        store.add_documents(docs)
        hit1 = hit3 = 0
        for question, phrase in QUESTIONS:
            results = [d.page_content.lower() for d in store.similarity_search(question, k=3)]
            hit1 += phrase.lower() in results[0]
            hit3 += any(phrase.lower() in r for r in results)
        avg = sum(len(d.page_content) for d in docs) // len(docs)
        label = f"{size}/{overlap}" + (" (old)" if old_style else "")
        print(f"{label:>12} {len(docs):>6} {hit1:>3}/{len(QUESTIONS)} {hit3:>3}/{len(QUESTIONS)}  {avg}")
        store.delete_collection()
    return 0


if __name__ == "__main__":
    sys.exit(main())
