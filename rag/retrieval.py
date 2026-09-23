"""Single entrypoint the rest of the app uses to get a retriever, so the
dense-vs-hybrid choice (config.RETRIEVAL_MODE) and whether reranking is on
(config.RERANK_ENABLED) each live in one place instead of being duplicated
across tools/doc_search.py and tools/qa_chain.py."""
from config import RETRIEVAL_MODE, RERANK_ENABLED, RERANK_CANDIDATES
from rag.vectorstore import get_retriever as _get_dense_retriever
from rag.hybrid_retriever import get_hybrid_retriever as _get_hybrid_retriever
from rag.reranker import RerankingRetriever


def get_configured_retriever(k: int = 5):
    fetch_k = max(RERANK_CANDIDATES, k) if RERANK_ENABLED else k

    if RETRIEVAL_MODE == "hybrid":
        base = _get_hybrid_retriever(k=fetch_k)
    else:
        base = _get_dense_retriever(k=fetch_k)

    if RERANK_ENABLED:
        return RerankingRetriever(base, top_k=k, fetch_k=fetch_k)
    return base
