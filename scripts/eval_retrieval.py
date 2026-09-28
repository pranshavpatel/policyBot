"""Retrieval-only evaluation: for each labeled question in eval/qa.jsonl,
measure standard IR ranking metrics — Recall@k, Precision@k, and MRR — for
dense vs. hybrid retrieval. No LLM call, no API key needed — just the local
embedding model and/or BM25, so this is cheap enough to run in CI or on
every PR.

This measures retrieval quality in isolation from generation quality; it's
what scripts/eval_via_api.py (end-to-end answer quality, requires a live
server + GROQ_API_KEY) can't tell you on its own — a wrong final answer
could be a retrieval miss OR a generation error, and only this script
isolates the first half.

Most questions in eval/qa.jsonl label exactly one correct source doc, but
a "source" can also be a list — for questions that are genuinely
ambiguous because two policies legitimately cover the same fact (e.g.
Leave Policy's parental-leave summary and the dedicated Parental Leave
Policy both correctly answer "how many weeks of parental leave"), any doc
in that list counts as a hit.

Metrics, and why these three specifically:
- Recall@k: was a correct source doc anywhere in the top-k? With exactly
  one relevant doc per query this is binary and identical to "hit rate";
  the multi-source questions are still binary per query (hit if *any*
  labeled source appears), just with more than one way to score a hit.
- Precision@k: what fraction of the top-k were actually relevant? This is
  where single- vs multi-source questions genuinely differ — hit/k for a
  single-source question, but a multi-source question could in principle
  have more than one relevant chunk in the same top-k, so precision isn't
  just an algebraic function of recall the way it is for single-source
  questions. It's what penalizes a bloated k (retrieving 10 chunks to find
  1 relevant one scores worse than retrieving 3) — Recall@k alone can't
  see that trade-off.
- MRR (Mean Reciprocal Rank): rewards ranking a correct doc 1st over 3rd,
  which Recall@k/hit-rate genuinely cannot distinguish — MRR is the fix
  for that blind spot.
NDCG is intentionally not here: it needs graded relevance labels (0/1/2/3),
and even the multi-source questions here are binary-relevant (a doc either
answers the question or it doesn't, no partial credit), so NDCG would
collapse to the same information MRR already gives.

Usage:
    python -m scripts.ingest_langchain          # build the vectorstore first
    python -m scripts.eval_retrieval [eval/qa.jsonl] [--k 1 --k 3 --k 5] [--rerank]

--rerank additionally reports dense+cross-encoder and hybrid+cross-encoder
(rag/reranker.py), so you can see whether reranking earns its extra
latency on top of whichever first-stage retriever you're already running.
"""
import argparse
import json
import sys
import time

from rag.vectorstore import get_retriever
from rag.hybrid_retriever import get_hybrid_retriever
from rag.reranker import RerankingRetriever

MRR_CUTOFF = 10  # how deep to look for the correct doc when computing MRR


def load_labeled_questions(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            if ex.get("source"):  # skip out-of-scope questions (source: null)
                rows.append(ex)
    return rows


def _rank_of_first_relevant(docs, sources):
    """1-indexed rank of the first chunk whose source is a relevant doc, or
    None. `sources` may be a single doc filename (the common case: one
    genuinely correct source) or a list (a deliberately ambiguous question —
    two policies legitimately cover the same fact, e.g. Leave Policy's
    parental-leave summary and the dedicated Parental Leave Policy — either
    is a correct retrieval, so eval/qa.jsonl labels both)."""
    relevant = {sources} if isinstance(sources, str) else set(sources)
    for i, d in enumerate(docs):
        if d.metadata.get("source") in relevant:
            return i + 1
    return None


def evaluate(retriever, rows, ks):
    """Retrieves MRR_CUTOFF candidates once per query, then computes every
    requested k's Recall/Precision from that single retrieval — cheaper
    than re-querying per k, and guarantees the metrics are all measuring
    the exact same ranking."""
    max_k = max(max(ks), MRR_CUTOFF)
    ranks = []           # rank of first relevant doc per query, or None
    latencies = []
    misses_by_k = {k: [] for k in ks}

    for ex in rows:
        t0 = time.time()
        docs = retriever.invoke(ex["q"])[:max_k]
        latencies.append((time.time() - t0) * 1000)
        rank = _rank_of_first_relevant(docs, ex["source"])
        ranks.append(rank)
        for k in ks:
            if rank is None or rank > k:
                misses_by_k[k].append(ex["q"])

    n = len(rows) or 1
    mrr = sum((1.0 / r) for r in ranks if r is not None and r <= MRR_CUTOFF) / n
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    per_k = {}
    for k in ks:
        hits = sum(1 for r in ranks if r is not None and r <= k)
        per_k[k] = {
            "recall_at_k": round(hits / n * 100, 1),
            "precision_at_k": round(hits / (n * k) * 100, 1),
            "misses": misses_by_k[k],
        }

    return {"mrr": round(mrr, 3), "avg_latency_ms": round(avg_latency, 1), "per_k": per_k, "n": len(rows)}


def _print_report(name, result, ks):
    print(f"{name}: MRR={result['mrr']}  (avg {result['avg_latency_ms']} ms/query, n={result['n']})")
    for k in ks:
        pk = result["per_k"][k]
        print(f"  @k={k}: Recall={pk['recall_at_k']}%  Precision={pk['precision_at_k']}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="eval/qa.jsonl")
    parser.add_argument("--k", type=int, action="append", dest="ks",
                         help="report Recall@k/Precision@k for this k (repeatable; default 1,3,5)")
    parser.add_argument("--show-misses", action="store_true", help="list queries missed at the largest k")
    parser.add_argument("--rerank", action="store_true",
                         help="also report dense/hybrid + cross-encoder reranking (rag/reranker.py)")
    args = parser.parse_args()
    ks = sorted(set(args.ks)) if args.ks else [1, 3, 5]
    fetch_k = max(max(ks), MRR_CUTOFF)

    rows = load_labeled_questions(args.path)
    print(f"Loaded {len(rows)} source-labeled questions from {args.path}\n")

    dense = get_retriever(k=fetch_k)
    dense_result = evaluate(dense, rows, ks)
    _print_report("Dense           ", dense_result, ks)

    print()
    hybrid = get_hybrid_retriever(k=fetch_k, fetch_k=max(10, max(ks) * 3))
    hybrid_result = evaluate(hybrid, rows, ks)
    _print_report("Hybrid          ", hybrid_result, ks)

    if args.rerank:
        # Cross-encoder model load + first-inference warmup is a one-time
        # ~1s CPU cost (see rag/reranker.py's docstring) that has nothing
        # to do with per-query retrieval latency — warm it here, the way
        # you'd warm a model before serving real traffic, so the timed
        # runs below report steady-state cost instead of the average
        # being skewed by whichever retriever happens to run first.
        from rag.reranker import rerank as _warm_rerank
        _warm_rerank("warmup", dense.invoke("warmup")[:1], 1)

        print()
        dense_rerank = RerankingRetriever(get_retriever(k=fetch_k), top_k=fetch_k, fetch_k=fetch_k)
        dense_rerank_result = evaluate(dense_rerank, rows, ks)
        _print_report("Dense + rerank  ", dense_rerank_result, ks)

        print()
        hybrid_rerank = RerankingRetriever(
            get_hybrid_retriever(k=fetch_k, fetch_k=max(10, max(ks) * 3)), top_k=fetch_k, fetch_k=fetch_k)
        hybrid_rerank_result = evaluate(hybrid_rerank, rows, ks)
        _print_report("Hybrid + rerank ", hybrid_rerank_result, ks)

    if args.show_misses:
        largest_k = max(ks)
        for name, result in (("Dense", dense_result), ("Hybrid", hybrid_result)):
            misses = result["per_k"][largest_k]["misses"]
            if misses:
                print(f"\n{name} misses at k={largest_k}:")
                for q in misses:
                    print("  -", q)


if __name__ == "__main__":
    sys.exit(main())
