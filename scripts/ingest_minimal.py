# scripts/ingest_minimal.py
from pathlib import Path

# --- project imports (works when run as: python -m scripts.ingest_minimal) ---
from rag.embedder import Embedder
from rag.index import VectorIndex
from rag.retriever import Retriever
from rag.chunker import chunk_markdown  # your headings-aware chunker
from tools.doc_search import doc_search
from tools.answer_with_citations import synthesize_answer


def build_retriever() -> Retriever:
    data_dir = Path("data/policies")
    file = data_dir / "leave_policy.md"
    text = file.read_text(encoding="utf-8")

    # Tune these if you want more/fewer chunks
    chunks = chunk_markdown(
        text,
        chunk_size=60,
        overlap=20,
        source=file.name,
    )
    print(f"num_chunks: {len(chunks)}")

    embedder = Embedder()
    index = VectorIndex()

    ids = [c["id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metas = [c["meta"] for c in chunks]

    vecs = embedder.encode(texts)
    index.add(ids, vecs, metas, texts)

    retriever = Retriever(index, embedder)
    return retriever


def demo_queries(retriever: Retriever) -> None:
    queries = [
        "How many PTO days in Year 1?",
        "How to request leave?",
    ]

    for q in queries:
        # Raw hits (for debugging)
        hits = retriever.retrieve(q, top_k=5)
        print(f"\nQ: {q}")
        for h in hits[:3]:
            print(f"- {h['score']:.3f} :: {h['meta']} :: {h['text'][:90]}…")

        # Tool output (what your agent would see)
        tool_out = doc_search(q, retriever, top_k=5)
        # Synthesized, cited answer (what you'd show users)
        answer = synthesize_answer(q, hits)  # or use tool_out to drive your LLM later
        print("\nAnswer:", answer)
        print("Citations:", tool_out["citations"][:3])


if __name__ == "__main__":
    retriever = build_retriever()
    demo_queries(retriever)
