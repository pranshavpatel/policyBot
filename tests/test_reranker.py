"""Unit tests for rag/reranker.py's ranking logic, using a fake cross-encoder
model (no ~90MB download, no real inference) so these run everywhere the
rest of the suite does. tests/test_retrieval.py separately covers the real
model wired through rag/retrieval.py, skipped if the vectorstore isn't
built."""
from types import SimpleNamespace
import rag.reranker as reranker


def _doc(text, source="x.md"):
    return SimpleNamespace(page_content=text, metadata={"source": source})


class FakeCrossEncoder:
    """Scores each (query, text) pair by how many of the query's words
    appear in the text — deterministic and good enough to prove sorting."""
    def predict(self, pairs):
        scores = []
        for query, text in pairs:
            q_words = set(query.lower().split())
            t_words = set(text.lower().split())
            scores.append(len(q_words & t_words))
        return scores


def test_rerank_sorts_by_score_descending(monkeypatch):
    monkeypatch.setattr(reranker, "_get_model", lambda: FakeCrossEncoder())
    docs = [
        _doc("completely unrelated text about lunch"),
        _doc("PTO days accrue in Year 1 of employment"),
        _doc("PTO"),
    ]
    out = reranker.rerank("PTO days Year 1", docs, top_k=3)
    assert out[0].page_content == "PTO days accrue in Year 1 of employment"
    assert out[-1].page_content == "completely unrelated text about lunch"


def test_rerank_respects_top_k(monkeypatch):
    monkeypatch.setattr(reranker, "_get_model", lambda: FakeCrossEncoder())
    docs = [_doc(f"PTO PTO PTO doc {i}") for i in range(5)]
    out = reranker.rerank("PTO", docs, top_k=2)
    assert len(out) == 2


def test_rerank_empty_docs_returns_empty(monkeypatch):
    monkeypatch.setattr(reranker, "_get_model", lambda: FakeCrossEncoder())
    assert reranker.rerank("anything", [], top_k=5) == []


def test_reranking_retriever_wraps_base(monkeypatch):
    monkeypatch.setattr(reranker, "_get_model", lambda: FakeCrossEncoder())

    class FakeBaseRetriever:
        def invoke(self, query):
            return [
                _doc("irrelevant filler text here"),
                _doc("PTO Year 1 accrual policy details"),
            ]

    r = reranker.RerankingRetriever(FakeBaseRetriever(), top_k=1, fetch_k=2)
    out = r.invoke("PTO Year 1")
    assert len(out) == 1
    assert out[0].page_content == "PTO Year 1 accrual policy details"
