import json, sys, time, requests

BASE = "http://localhost:8000"

def contains_all(text, needles): 
    t = text.lower(); return all(n.lower() in t for n in needles)
def contains_none(text, needles):
    t = text.lower(); return all(n.lower() not in t for n in needles)

def main(path="eval/qa.jsonl", trace=False):
    total=0; correct=0; rows=[]
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            ex=json.loads(line)
            q=ex["q"]; total+=1
            t0=time.time()
            try:
                r=requests.post(f"{BASE}/agent", json={"message": q, "trace": trace}, timeout=60)
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
