"""Unit tests for scripts/eval_retrieval.py's metric math (Recall@k,
Precision@k, MRR) using a stub retriever — no vectorstore/embedding model
needed, so these run everywhere the rest of the suite does."""
from types import SimpleNamespace
from scripts.eval_retrieval import _rank_of_first_relevant, evaluate


def _doc(source):
    return SimpleNamespace(metadata={"source": source})


class StubRetriever:
    """Returns a fixed, pre-scripted ranking per query text."""
    def __init__(self, rankings: dict):
        self.rankings = rankings

    def invoke(self, query):
        return [_doc(s) for s in self.rankings[query]]


def test_rank_of_first_relevant_finds_first_match():
    docs = [_doc("a.md"), _doc("b.md"), _doc("c.md")]
    assert _rank_of_first_relevant(docs, "b.md") == 2


def test_rank_of_first_relevant_none_when_absent():
    docs = [_doc("a.md"), _doc("b.md")]
    assert _rank_of_first_relevant(docs, "z.md") is None


def test_mrr_rewards_higher_rank():
    # q1: correct doc ranked 1st -> RR=1.0 ; q2: ranked 3rd -> RR=1/3
    rows = [{"q": "q1", "source": "a.md"}, {"q": "q2", "source": "a.md"}]
    retriever = StubRetriever({
        "q1": ["a.md", "b.md", "c.md"],
        "q2": ["b.md", "c.md", "a.md"],
    })
    result = evaluate(retriever, rows, ks=[3])
    assert result["mrr"] == round((1.0 + 1 / 3) / 2, 3)


def test_recall_and_precision_at_k():
    rows = [{"q": "q1", "source": "a.md"}, {"q": "q2", "source": "z.md"}]
    retriever = StubRetriever({
        "q1": ["a.md", "b.md", "c.md"],   # hit at k=1
        "q2": ["b.md", "c.md", "d.md"],   # miss entirely
    })
    result = evaluate(retriever, rows, ks=[1, 3])
    assert result["per_k"][1]["recall_at_k"] == 50.0     # 1/2 queries
    assert result["per_k"][1]["precision_at_k"] == 50.0  # 1 hit / (2 queries * k=1)
    assert result["per_k"][3]["recall_at_k"] == 50.0      # still just q1
    assert result["per_k"][3]["precision_at_k"] == round(1 / (2 * 3) * 100, 1)
    assert result["per_k"][3]["misses"] == ["q2"]


def test_never_found_gets_zero_reciprocal_rank():
    rows = [{"q": "q1", "source": "z.md"}]
    retriever = StubRetriever({"q1": ["a.md", "b.md"]})
    result = evaluate(retriever, rows, ks=[3])
    assert result["mrr"] == 0.0
