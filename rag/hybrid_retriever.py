"""Hybrid (BM25 + dense) retrieval via Reciprocal Rank Fusion.

Dense (embedding) retrieval is good at paraphrase/semantic matches but can
miss queries that hinge on an exact term (a policy name, a specific number,
an acronym like "MFA" or "FMLA"). BM25 (lexical/keyword) retrieval is the
opposite: great on exact terms, poor on paraphrase. Combining their rankings
with Reciprocal Rank Fusion (RRF) — rather than trying to calibrate and add
their raw scores, which live on different scales — is a standard,
well-studied way to get both without extra training:

    score(doc) = sum over rankers of  1 / (k + rank_in_that_ranker)

See config.RETRIEVAL_MODE ("dense" | "hybrid") to toggle this at the
tools/doc_search.py and tools/qa_chain.py call sites; rag/hybrid_eval.py
measures the actual retrieval-quality delta between the two.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from rag.splitter import split_markdown
from rag.vectorstore import get_retriever as get_dense_retriever

POLICY_DIR = Path(__file__).resolve().parent.parent / "data" / "policies"


def load_corpus(policy_dir: Path = POLICY_DIR) -> List[Document]:
    """Rebuild the same chunk corpus the ingest script writes to Chroma,
    directly from the source markdown files. BM25 needs the raw corpus
    in memory (it's not an index Chroma persists), so this keeps the two
    retrievers looking at identical chunks without a second data store."""
    docs: List[Document] = []
    for path in sorted(policy_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        docs.extend(split_markdown(text, source=path.name))
    return docs


_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def _tokenize(text: str) -> List[str]:
    """BM25Retriever's default preprocess_func is just str.split() — no
    lowercasing or punctuation stripping, so "(FMLA)." never matches a
    query of "FMLA". Word-boundary tokenization + lowercasing fixes that."""
    return _TOKEN_RE.findall(text.lower())


def _doc_key(d: Document) -> str:
    return f"{d.metadata.get('source', '')}::{d.page_content[:120]}"


def reciprocal_rank_fusion(ranked_lists: List[List[Document]], k: int = 60) -> List[Document]:
    scores: dict = {}
    doc_by_key: dict = {}
    for docs in ranked_lists:
        for rank, d in enumerate(docs):
            key = _doc_key(d)
            doc_by_key[key] = d
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
    ranked_keys = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [doc_by_key[key] for key, _ in ranked_keys]


class HybridRetriever:
    def __init__(self, dense_retriever, bm25_retriever, top_k: int = 5):
        self._dense = dense_retriever
        self._bm25 = bm25_retriever
        self.top_k = top_k

    def invoke(self, query: str) -> List[Document]:
        dense_hits = self._dense.invoke(query)
        bm25_hits = self._bm25.invoke(query)
        fused = reciprocal_rank_fusion([dense_hits, bm25_hits])
        return fused[: self.top_k]


_bm25_singleton: BM25Retriever | None = None


def _get_bm25(k: int) -> BM25Retriever:
    global _bm25_singleton
    if _bm25_singleton is None:
        _bm25_singleton = BM25Retriever.from_documents(load_corpus(), preprocess_func=_tokenize)
    _bm25_singleton.k = k
    return _bm25_singleton


def get_hybrid_retriever(k: int = 5, fetch_k: int = 10) -> HybridRetriever:
    """fetch_k: how many candidates each ranker contributes before fusion
    trims to top_k — wider than k so RRF has something to actually fuse."""
    dense = get_dense_retriever(k=fetch_k)
    bm25 = _get_bm25(k=fetch_k)
    return HybridRetriever(dense, bm25, top_k=k)
