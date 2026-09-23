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

Metrics, and why these three specifically:
- Recall@k: was the correct source doc anywhere in the top-k? Binary per
  query here because eval/qa.jsonl labels exactly one relevant doc per
  question — with only one relevant item, Recall@k and "hit rate" are the
  same number, this just uses the standard IR name for it.
- Precision@k: what fraction of the top-k were actually relevant? With one
  relevant doc per query this is just hit/k, but it's what actually
  penalizes a bloated k (retrieving 10 chunks to find 1 relevant one scores
  worse on precision than retrieving 3, even though recall is identical) —
  Recall@k alone can't tell you that trade-off.
- MRR (Mean Reciprocal Rank): rewards ranking the correct doc 1st over 3rd,
  which Recall@k/hit-rate genuinely cannot distinguish — on this corpus
  Recall@k was maxed out at 100% and told us nothing about ranking quality
  within that top-k; MRR is the fix for that blind spot.
NDCG is intentionally not here: it needs graded relevance labels (0/1/2/3),
and eval/qa.jsonl only has a single binary-relevant doc per question, so
NDCG would collapse to the same information MRR already gives.

Usage:
    python -m scripts.ingest_langchain          # build the vectorstore first
    python -m scripts.eval_retrieval [eval/qa.jsonl] [--k 1 --k 3 --k 5]
"""
import argparse
import json
import sys
import time

from rag.vectorstore import get_retriever
from rag.hybrid_retriever import get_hybrid_retriever

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


def _rank_of_first_relevant(docs, source: str):
    """1-indexed rank of the first chunk whose source matches, or None."""
    for i, d in enumerate(docs):
        if d.metadata.get("source") == source:
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
    args = parser.parse_args()
    ks = sorted(set(args.ks)) if args.ks else [1, 3, 5]

    rows = load_labeled_questions(args.path)
    print(f"Loaded {len(rows)} source-labeled questions from {args.path}\n")

    dense = get_retriever(k=max(max(ks), MRR_CUTOFF))
    dense_result = evaluate(dense, rows, ks)
    _print_report("Dense ", dense_result, ks)

    print()
    hybrid = get_hybrid_retriever(k=max(max(ks), MRR_CUTOFF), fetch_k=max(10, max(ks) * 3))
    hybrid_result = evaluate(hybrid, rows, ks)
    _print_report("Hybrid", hybrid_result, ks)

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
