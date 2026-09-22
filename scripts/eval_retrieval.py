"""Retrieval-only evaluation: for each labeled question in eval/qa.jsonl,
check whether the correct source document appears in the top-k retrieved
chunks. No LLM call, no API key needed — just the local embedding model
and/or BM25, so this is cheap enough to run in CI or on every PR.

This measures retrieval quality in isolation from generation quality; it's
what scripts/eval_via_api.py (end-to-end answer quality, requires a live
server + GROQ_API_KEY) can't tell you on its own — a wrong final answer
could be a retrieval miss OR a generation error, and only this script
isolates the first half.

Usage:
    python scripts/ingest_langchain.py   # build the vectorstore first
    python scripts/eval_retrieval.py [eval/qa.jsonl] [--k 3]
"""
import argparse
import json
import sys
import time

from rag.vectorstore import get_retriever
from rag.hybrid_retriever import get_hybrid_retriever


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


def hit_rate(retriever, rows, k: int):
    hits = 0
    latencies = []
    misses = []
    for ex in rows:
        t0 = time.time()
        docs = retriever.invoke(ex["q"])
        latencies.append((time.time() - t0) * 1000)
        sources = {d.metadata.get("source") for d in docs[:k]}
        if ex["source"] in sources:
            hits += 1
        else:
            misses.append(ex["q"])
    n = len(rows)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    return {
        "hits": hits,
        "total": n,
        "hit_rate": round(hits / n * 100, 1) if n else 0.0,
        "avg_latency_ms": round(avg_latency, 1),
        "misses": misses,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="eval/qa.jsonl")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    rows = load_labeled_questions(args.path)
    print(f"Loaded {len(rows)} source-labeled questions from {args.path} (top-{args.k})\n")

    dense = get_retriever(k=args.k)
    dense_result = hit_rate(dense, rows, args.k)
    print(f"Dense  : {dense_result['hits']}/{dense_result['total']} = "
          f"{dense_result['hit_rate']}%  (avg {dense_result['avg_latency_ms']} ms/query)")

    hybrid = get_hybrid_retriever(k=args.k, fetch_k=max(10, args.k * 3))
    hybrid_result = hit_rate(hybrid, rows, args.k)
    print(f"Hybrid : {hybrid_result['hits']}/{hybrid_result['total']} = "
          f"{hybrid_result['hit_rate']}%  (avg {hybrid_result['avg_latency_ms']} ms/query)")

    if dense_result["misses"]:
        print("\nDense misses:")
        for q in dense_result["misses"]:
            print("  -", q)
    if hybrid_result["misses"]:
        print("\nHybrid misses:")
        for q in hybrid_result["misses"]:
            print("  -", q)


if __name__ == "__main__":
    sys.exit(main())
