"""Cross-encoder reranking.

A dense/hybrid retriever scores (query, chunk) pairs independently — the
query embedding and each chunk embedding never actually see each other,
they're just compared by cosine similarity after the fact. A cross-encoder
scores the (query, chunk) pair *jointly* in one forward pass, so it can
pick up on interactions a bi-encoder's independent embeddings structurally
can't — which is why cross-encoders consistently score higher on
retrieval-ranking benchmarks than bi-encoder similarity alone. The
trade-off is cost: it's one model forward pass per candidate instead of a
single batched similarity computation, so it only runs on a small
candidate set (RERANK_CANDIDATES) pulled from the cheap first-stage
retriever, never the whole corpus — the standard retrieve-then-rerank
two-stage pattern.

cross-encoder/ms-marco-MiniLM-L-6-v2 is used here: a small (~90MB), CPU-
friendly model trained specifically for passage reranking (MS MARCO),
widely used as the default choice for exactly this job — not a novel
pick, a boring, well-understood one, which is the point for a second-stage
component that runs on every query.
"""
from __future__ import annotations
from typing import List

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from config import RERANK_MODEL, RERANK_CANDIDATES

_model_singleton: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model_singleton
    if _model_singleton is None:
        _model_singleton = CrossEncoder(RERANK_MODEL)
    return _model_singleton


def rerank(query: str, docs: List[Document], top_k: int) -> List[Document]:
    """Scores each doc against the query with the cross-encoder and returns
    the top_k, highest-score-first. Safe to call with fewer docs than
    top_k (just returns them all, reranked)."""
    if not docs:
        return []
    pairs = [(query, d.page_content) for d in docs]
    scores = _get_model().predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
    return [d for d, _ in ranked[:top_k]]


class RerankingRetriever:
    """Wraps any retriever exposing .invoke(query) -> List[Document] with a
    cross-encoder reranking stage. fetch_k candidates come from the base
    retriever; the cross-encoder reorders them and only the top_k survive."""

    def __init__(self, base_retriever, top_k: int, fetch_k: int = RERANK_CANDIDATES):
        self._base = base_retriever
        self.top_k = top_k
        self.fetch_k = max(fetch_k, top_k)

    def invoke(self, query: str) -> List[Document]:
        candidates = self._base.invoke(query)[: self.fetch_k]
        return rerank(query, candidates, self.top_k)
