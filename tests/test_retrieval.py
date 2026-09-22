"""Retrieval correctness tests against the real (local, free) embedding
model and a real Chroma store — no LLM/API key involved. CI runs the ingest
script before pytest (see .github/workflows/ci.yml); locally, run
`python scripts/ingest_langchain.py` first or these are skipped."""
import pytest
from pathlib import Path

from config import CHROMA_DIR

pytestmark = pytest.mark.skipif(
    not Path(CHROMA_DIR).exists(),
    reason="vectorstore not built — run scripts/ingest_langchain.py first",
)


def test_dense_retrieval_finds_pto_accrual_chunk():
    from rag.vectorstore import get_retriever
    docs = get_retriever(k=3).invoke("How many PTO days do I get in Year 1?")
    assert any("15 days" in d.page_content for d in docs)


def test_hybrid_retrieval_finds_pto_accrual_chunk():
    from rag.hybrid_retriever import get_hybrid_retriever
    docs = get_hybrid_retriever(k=3).invoke("How many PTO days do I get in Year 1?")
    assert any("15 days" in d.page_content for d in docs)


def test_hybrid_finds_exact_acronym_query():
    # A case dense embeddings alone can miss: a bare acronym with little
    # semantic context for the embedding model to latch onto.
    from rag.hybrid_retriever import get_hybrid_retriever
    docs = get_hybrid_retriever(k=3).invoke("FMLA")
    assert any("FMLA" in d.page_content for d in docs)


def test_configured_retriever_respects_retrieval_mode(monkeypatch):
    import config
    import rag.retrieval as retrieval

    monkeypatch.setattr(config, "RETRIEVAL_MODE", "hybrid")
    monkeypatch.setattr(retrieval, "RETRIEVAL_MODE", "hybrid")
    from rag.hybrid_retriever import HybridRetriever
    r = retrieval.get_configured_retriever(k=2)
    assert isinstance(r, HybridRetriever)
