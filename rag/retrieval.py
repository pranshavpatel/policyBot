"""Single entrypoint the rest of the app uses to get a retriever, so the
dense-vs-hybrid choice (config.RETRIEVAL_MODE) lives in one place instead of
being duplicated across tools/doc_search.py and tools/qa_chain.py."""
from config import RETRIEVAL_MODE
from rag.vectorstore import get_retriever as _get_dense_retriever
from rag.hybrid_retriever import get_hybrid_retriever as _get_hybrid_retriever


def get_configured_retriever(k: int = 5):
    if RETRIEVAL_MODE == "hybrid":
        return _get_hybrid_retriever(k=k)
    return _get_dense_retriever(k=k)
