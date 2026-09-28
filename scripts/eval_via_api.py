"""End-to-end answer-quality eval against a live /agent endpoint.

Requires: a running API server (uvicorn api.app:app) with a real LLM
behind it — a GROQ_API_KEY, or LLM_PROVIDER=local pointed at a running
Ollama/vLLM server. This makes real LLM calls, unlike
scripts/eval_retrieval.py, which is why it isn't part of CI.

Usage:
    uvicorn api.app:app --port 8000 &
    python -m scripts.eval_via_api eval/qa.jsonl [results.json]

Prints one progress line per question as it finishes, then accuracy and a
latency summary (mean/p50/p95/max). If a second argument is given, the
per-question rows and the summary are also written there as JSON.

EVAL_TIMEOUT (seconds, default 300) is the per-question HTTP timeout. It
should stay above LOCAL_LLM_TIMEOUT_SECONDS: if the client gives up first,
it moves on while the server is still busy with the abandoned question, and
the next question queues behind it — which is how one stuck generation
turned a whole local-model run into a cascade of timeouts.
"""
import json, os, statistics, sys, time, unicodedata, requests

BASE = os.getenv("API_BASE", "http://localhost:8000")
EVAL_USERNAME = os.getenv("EVAL_USERNAME", "alice")
EVAL_PASSWORD = os.getenv("EVAL_PASSWORD", "alice123")
EVAL_TIMEOUT = float(os.getenv("EVAL_TIMEOUT", "300"))


def _norm(text: str) -> str:
    # LLM output routinely uses "typographic" Unicode punctuation — narrow
    # no-break spaces (U+202F) between a number and its unit, non-breaking
    # hyphens, en/em dashes — that a naive substring check on must_contain
    # like "june 30" will silently miss even though the answer is correct.
    # NFKC folds these to their ASCII-ish compatibility form.
    return unicodedata.normalize("NFKC", text).lower()


def contains_all(text, needles):
    t = _norm(text); return all(_norm(n) in t for n in needles)
def contains_none(text, needles):
    t = _norm(text); return all(_norm(n) not in t for n in needles)


def _login() -> str:
    r = requests.post(f"{BASE}/auth/login", data={"username": EVAL_USERNAME, "password": EVAL_PASSWORD}, timeout=15)
    r.raise_for_status()
    return r.json()["access_token"]


# Retries specifically on 429 (api/app.py's GroqRateLimitError handler
# returns this distinctly from a real 500 — see api/app.py's exception
# handler docstring for why that distinction matters here). A real 70-
# question eval run hit Groq's rate limit partway through and every
# subsequent request failed until the process was restarted; this is the
# fix, not a hypothetical.
MAX_RETRIES = 5
BASE_BACKOFF_SECONDS = 10


def _post_with_retry(url, **kwargs):
    for attempt in range(MAX_RETRIES + 1):
        r = requests.post(url, **kwargs)
        if r.status_code != 429:
            return r
        if attempt == MAX_RETRIES:
            return r
        wait = int(r.headers.get("Retry-After", BASE_BACKOFF_SECONDS * (2 ** attempt)))
        print(f"   … rate limited, retrying in {wait}s (attempt {attempt + 1}/{MAX_RETRIES})")
        time.sleep(wait)
    return r  # unreachable, satisfies linters


def _percentile(sorted_vals, pct):
    # Nearest-rank percentile: no interpolation, so every reported number
    # is a latency that actually happened in this run.
    if not sorted_vals:
        return 0.0
    rank = max(1, -(-pct * len(sorted_vals) // 100))  # ceil(pct/100 * n)
    return sorted_vals[int(rank) - 1]


def latency_summary(ms_values):
    vals = sorted(ms_values)
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean_ms": round(statistics.fmean(vals), 1),
        "p50_ms": round(_percentile(vals, 50), 1),
        "p95_ms": round(_percentile(vals, 95), 1),
        "max_ms": round(vals[-1], 1),
    }


def main(path="eval/qa.jsonl", out_path=None, trace=False):
    # On Windows, piped stdout (e.g. into Tee-Object) is cp1252, which can't
    # encode characters LLM answers routinely contain (U+202F etc.) —
    # replace them instead of crashing the run mid-way.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    token = _login()
    headers = {"Authorization": f"Bearer {token}"}

    with open(path, "r", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]

    total=len(examples); correct=0; rows=[]
    for i, ex in enumerate(examples, 1):
        q=ex["q"]
        t0=time.time()
        error=None
        try:
            r=_post_with_retry(f"{BASE}/agent", json={"message": q, "trace": trace}, headers=headers, timeout=EVAL_TIMEOUT)
            r.raise_for_status()
            ans=r.json().get("answer","")
        except Exception as e:
            error=str(e)
            ans=f"__ERROR__ {e}"
        ms=(time.time()-t0)*1000
        ok = error is None and contains_all(ans, ex.get("must_contain", [])) and contains_none(ans, ex.get("must_not", []))
        if ok: correct+=1
        rows.append({"q":q,"ok":ok,"error":error is not None,"ms":round(ms,1),"ans":ans[:240]})
        # flush: stdout is block-buffered when piped (e.g. into Tee-Object),
        # which otherwise makes a healthy run look frozen until it finishes.
        print(f"[{i}/{total}] {'PASS' if ok else 'FAIL'} {q} ({ms/1000:.1f}s)", flush=True)

    acc=(correct/total*100) if total else 0.0
    errors=sum(r["error"] for r in rows)
    # Latency over answered questions only: an error's time is just however
    # long it took to fail (usually the timeout), not a response time.
    lat=latency_summary([r["ms"] for r in rows if not r["error"]])
    print()
    print(f"API eval: {correct}/{total} = {acc:.1f}%  (errors: {errors})")
    if lat["n"]:
        print(f"Latency over {lat['n']} answered: mean {lat['mean_ms']/1000:.1f}s, "
              f"p50 {lat['p50_ms']/1000:.1f}s, p95 {lat['p95_ms']/1000:.1f}s, max {lat['max_ms']/1000:.1f}s")
    for r in rows:
        if not r["ok"]:
            print("FAIL", r["q"], f"({r['ms']} ms)")
            print("   ->", r["ans"])

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"eval_set": path, "correct": correct, "total": total,
                       "accuracy_pct": round(acc, 1), "errors": errors,
                       "latency": lat, "rows": rows}, f, ensure_ascii=False, indent=2)
        print(f"Wrote {out_path}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv)>1 else "eval/qa.jsonl"
    out_path = sys.argv[2] if len(sys.argv)>2 else None
    main(path, out_path, trace=False)
