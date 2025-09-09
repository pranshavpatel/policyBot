import sys, requests, json
BASE="http://localhost:8000"
msg=" ".join(sys.argv[1:]) or "How many PTO days in Year 1?"
res=requests.post(f"{BASE}/agent", json={"message": msg, "trace": False}, timeout=60).json()
print(json.dumps(res, indent=2, ensure_ascii=False))
