"""End-to-end answer-quality eval against a live /agent endpoint.

Requires: a running API server (uvicorn api.app:app) with a real
GROQ_API_KEY configured — this makes real LLM calls, unlike
scripts/eval_retrieval.py, which is why it isn't part of CI.

Usage:
    uvicorn api.app:app --port 8000 &
    python -m scripts.eval_via_api eval/qa.jsonl
"""
import json, os, sys, time, unicodedata, requests

BASE = os.getenv("API_BASE", "http://localhost:8000")
EVAL_USERNAME = os.getenv("EVAL_USERNAME", "alice")
EVAL_PASSWORD = os.getenv("EVAL_PASSWORD", "alice123")


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


def main(path="eval/qa.jsonl", trace=False):
    token = _login()
    headers = {"Authorization": f"Bearer {token}"}

    total=0; correct=0; rows=[]
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            ex=json.loads(line)
            q=ex["q"]; total+=1
            t0=time.time()
            try:
                r=requests.post(f"{BASE}/agent", json={"message": q, "trace": trace}, headers=headers, timeout=60)
                r.raise_for_status()
                ans=r.json().get("answer","")
            except Exception as e:
                ans=f"__ERROR__ {e}"
            ms=(time.time()-t0)*1000
            ok = contains_all(ans, ex.get("must_contain", [])) and contains_none(ans, ex.get("must_not", []))
            if ok: correct+=1
            rows.append({"q":q,"ok":ok,"ms":round(ms,1),"ans":ans[:240]})
    acc=(correct/total*100) if total else 0.0
    print(f"API eval: {correct}/{total} = {acc:.1f}%")
    for r in rows:
        print(("✅" if r["ok"] else "❌"), r["q"], f"({r['ms']} ms)")
        if not r["ok"]:
            print("   ↳", r["ans"])

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv)>1 else "eval/qa.jsonl"
    main(path, trace=False)
