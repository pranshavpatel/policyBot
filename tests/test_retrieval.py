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


def test_configured_retriever_respects_rerank_toggle(monkeypatch):
    import config
    import rag.retrieval as retrieval
    from rag.reranker import RerankingRetriever

    monkeypatch.setattr(config, "RERANK_ENABLED", True)
    monkeypatch.setattr(retrieval, "RERANK_ENABLED", True)
    r = retrieval.get_configured_retriever(k=2)
    assert isinstance(r, RerankingRetriever)


def test_reranking_finds_pto_accrual_chunk_with_real_model():
    # End-to-end with the real cross-encoder (downloads on first use if not
    # already cached) — confirms the actual wiring, not just the sorting
    # logic tests/test_reranker.py already covers with a fake model.
    from rag.vectorstore import get_retriever
    from rag.reranker import RerankingRetriever
    r = RerankingRetriever(get_retriever(k=10), top_k=3, fetch_k=10)
    docs = r.invoke("How many PTO days do I get in Year 1?")
    assert any("15 days" in d.page_content for d in docs)


@pytest.mark.parametrize("query", [
    "How do I reset a forgotten VPN password?",
    "Who do I contact if my VPN reset still isn't working?",
])
def test_reranking_ranks_vpn_reset_first_over_its_own_cross_reference(query):
    # Regression test for a real miss found via scripts/eval_retrieval.py
    # on the expanded (26-doc) corpus: dense retrieval alone ranked
    # remote_access_policy.md's one-line cross-reference ("For a forgotten
    # VPN password, follow the steps in VPN Reset") ABOVE vpn_reset.md's
    # actual numbered steps, because that cross-reference sentence
    # lexically echoes the query almost word-for-word — a bi-encoder
    # comparing independently-computed embeddings has no way to notice
    # it's a pointer, not an answer. The cross-encoder reranker (scoring
    # the (query, chunk) pair jointly) correctly promotes vpn_reset.md to
    # rank 1; this asserts that guarantee holds, not dense's specific
    # wrong behavior (which is a softer, more change-tolerant regression
    # signal — see README's "Retrieval quality" section for the full
    # measured comparison).
    from rag.vectorstore import get_retriever
    from rag.reranker import RerankingRetriever
    r = RerankingRetriever(get_retriever(k=10), top_k=1, fetch_k=10)
    docs = r.invoke(query)
    assert docs[0].metadata.get("source") == "vpn_reset.md"


def test_multi_source_question_matches_either_labeled_doc():
    # eval/qa.jsonl labels some questions with more than one correct source
    # doc — genuine overlap, not an eval-authoring mistake (e.g. Leave
    # Policy's parental-leave summary and the dedicated Parental Leave
    # Policy both correctly answer this). Confirm retrieval finds at least
    # one of them, and that scripts/eval_retrieval.py's rank helper (which
    # the real eval run relies on) handles a list source correctly against
    # real retrieval output, not just the stub retriever in
    # tests/test_eval_retrieval_metrics.py.
    from rag.hybrid_retriever import get_hybrid_retriever
    from scripts.eval_retrieval import _rank_of_first_relevant

    docs = get_hybrid_retriever(k=5).invoke("How many weeks of paid parental leave do eligible employees get?")
    rank = _rank_of_first_relevant(docs, ["leave_policy.md", "parental_leave_policy.md"])
    assert rank is not None
