"""Unit tests for rag/embeddings.py's fastembed-based Embeddings
implementation. Real model load (fastembed downloads/caches
sentence-transformers/all-MiniLM-L6-v2 as an ONNX export on first use, no
GROQ_API_KEY or network beyond that needed) — this is the actual migration
from sentence-transformers/torch (see the module's docstring for why:
908MB -> ~330MB app memory footprint, verified live against a real 512MB-
constrained Docker container, not just measured in isolation)."""
from rag.embeddings import get_embeddings, FastEmbedEmbeddings


def test_get_embeddings_returns_fastembed_implementation():
    emb = get_embeddings()
    assert isinstance(emb, FastEmbedEmbeddings)


def test_embed_query_returns_correct_dimension():
    emb = get_embeddings()
    vec = emb.embed_query("How many PTO days in Year 1?")
    assert len(vec) == 384  # all-MiniLM-L6-v2's output dimension
    assert all(isinstance(x, float) for x in vec)


def test_embed_documents_handles_a_batch():
    emb = get_embeddings()
    vecs = emb.embed_documents(["PTO policy", "VPN reset instructions"])
    assert len(vecs) == 2
    assert all(len(v) == 384 for v in vecs)


def test_similar_texts_embed_closer_than_dissimilar_ones():
    # Sanity check that the ONNX model produces semantically meaningful
    # vectors, not just correctly-shaped ones — same kind of real check as
    # scripts/eval_retrieval.py runs at the corpus level.
    import math

    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb)

    emb = get_embeddings()
    pto_a = emb.embed_query("How many vacation days do I get?")
    pto_b = emb.embed_query("What is my PTO allowance?")
    unrelated = emb.embed_query("How do I reset my VPN password?")

    assert cosine(pto_a, pto_b) > cosine(pto_a, unrelated)
