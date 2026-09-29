"""Embeddings via fastembed (ONNX runtime), not sentence-transformers/torch.

Found live deploying to Render's free tier: this app OOM'd at 512MB before
serving a single request. Profiled it — `import sentence_transformers`
alone (via langchain_huggingface.HuggingFaceEmbeddings, the previous
implementation) jumps process memory from ~213MB to ~803MB, before even
loading model weights; the full app was landing around ~900MB just to
start. fastembed uses ONNX runtime instead of PyTorch for inference and
has no torch dependency: importing it costs ~80MB, and loading this exact
same model (same weights, same 384-dim output — sentence-transformers/
all-MiniLM-L6-v2 is available as a standard ONNX export) plus embedding a
query lands around 367MB total. That's the fix, not a smaller/different
model — verified numerically compatible by re-running the full ingest and
retrieval eval against the new embeddings (see README).

rag/reranker.py's cross-encoder still uses sentence-transformers/torch —
that's fine, because it's optional (RERANK_ENABLED=false by default) and
only imported when actually enabled (see rag/retrieval.py), so it doesn't
cost anything in the default deploy configuration this fix targets.
"""
from typing import List

from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings

from config import EMBED_MODEL, EMBED_THREADS

_model_singleton: TextEmbedding | None = None


def _get_model() -> TextEmbedding:
    global _model_singleton
    if _model_singleton is None:
        # ONNX Runtime (fastembed's backend) sizes its thread pool to the
        # visible CPU count by default — 8 in this project's own dev
        # container — and each thread carries its own working buffers.
        # That's real memory, not just CPU scheduling, and it's what
        # turned a "fits at 512MB" measurement into an actual OOM under
        # request load on a real container (verified: idle memory scaled
        # UP with the container's memory *limit*, 440MB at 512MB -> 629MB
        # at 768MB, the signature of a pool auto-sizing to what looks
        # available rather than a fixed requirement). A single request at
        # a time doesn't benefit from 8-way intra-op parallelism on one
        # short embedding call anyway — capped via EMBED_THREADS
        # (config.py), default 1.
        _model_singleton = TextEmbedding(model_name=EMBED_MODEL, threads=EMBED_THREADS)
    return _model_singleton


class FastEmbedEmbeddings(Embeddings):
    """Minimal langchain_core.embeddings.Embeddings implementation over
    fastembed, so Chroma(embedding_function=...) keeps working unchanged —
    only this module's internals differ from the previous
    HuggingFaceEmbeddings-based one."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [vec.tolist() for vec in _get_model().embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def get_embeddings() -> Embeddings:
    return FastEmbedEmbeddings()
