# PolicyBot – HR Policy Q&A + Leave Management

[![CI](https://github.com/pranshavpatel/policyBot/actions/workflows/ci.yml/badge.svg)](https://github.com/pranshavpatel/policyBot/actions/workflows/ci.yml)

PolicyBot is an HR assistant that answers policy questions over a RAG pipeline and manages leave requests through a ReAct-style chat agent and a REST API, both behind JWT auth with role-based access control and an audit trail.

- **FastAPI backend**, JWT auth + RBAC, structured JSON logging
- **Hybrid (BM25 + dense) retrieval** over 6 HR/IT policy docs, with a lexical groundedness guardrail on generated answers
- **ReAct-style chat agent** (Groq / Llama) that can also create/approve/reject/cancel leave requests as tool calls
- **React + Tailwind frontend** for the chat UI
- **Slack integration** via the Events API
- **45-test pytest suite + GitHub Actions CI**

---

## Architecture

```
                     ┌─────────────┐
  React UI  ───────▶ │  FastAPI    │ ◀─── Slack Events API
  (frontend/)        │  api/app.py │
                     └──────┬──────┘
                            │  JWT (auth/)  →  RBAC (auth/authz.py)  →  audit log (auth/audit.py)
                ┌───────────┼────────────────┐
                ▼                            ▼
        tools/qa_chain.py            agent/react_agent.py  (ReAct loop)
        (RAG: retrieve→prompt→LLM)          │
                │                     agent/tools.py (create/approve/reject/
                ▼                     cancel leave, doc_search, holidays…)
        rag/retrieval.py                    │
        (dense | hybrid, config.py)         ▼
                │                     tools/leave_request.py → db/ (SQLite)
                ▼
        rag/groundedness.py (numeric/date claims checked against context)
```

Every write path — direct REST call *or* chat-agent tool call — goes through the same `auth/authz.py` rules and writes to the same audit log; see [Auth & authorization](#auth--authorization).

---

## Setup

### 1. Clone & install
```bash
git clone https://github.com/pranshavpatel/policyBot.git
cd policyBot
python -m venv .venv && source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env   # fill in GROQ_API_KEY at minimum
```

### 2. Initialize the database and build the vectorstore
```bash
python -m scripts.db_init          # creates data/policybot.db (leave_requests, audit_log)
python -m scripts.ingest_langchain # chunks data/policies/*.md into Chroma (local embeddings, no API key needed)
```

### 3. Run the API
```bash
uvicorn api.app:app --reload --port 8000
```

### 4. Run the frontend
```bash
cd frontend && npm install && npm run dev   # http://localhost:5173
```

### 5. Docker
```bash
docker compose up --build
```

### 6. Slack (optional)
Create a Slack App → Event Subscriptions → point the Request URL at `https://<your-tunnel>/slack/events`.

---

## Auth & authorization

Every leave-request and chat/agent endpoint requires a bearer token. Three seeded demo users (see `auth/users.py`):

| username | password | role |
|---|---|---|
| `alice` | `alice123` | employee |
| `bob` | `bob123` | employee |
| `manager1` | `manager123` | manager |

```bash
TOKEN=$(curl -s -X POST localhost:8000/auth/login -d "username=alice&password=alice123" | jq -r .access_token)
curl -H "Authorization: Bearer $TOKEN" localhost:8000/leave-requests
```

Rules, centralized in `auth/authz.py` and enforced identically by the REST API (`api/app.py`) **and** the chat agent (`agent/tools.py`, so "ask the bot to approve my own leave" is rejected the same way a direct API call is):

- An employee can only create, view, and cancel **their own** leave requests — `?user=someone-else` is silently ignored and scoped back to the caller.
- Only a **manager** can approve/reject, and never their own request.
- Every login/create/approve/reject/cancel is written to an append-only audit log, readable at `GET /audit-log` (manager-only).

This was verified end-to-end against a live server, not just unit-tested — see the commit history for the full request/response trace (login → create → self-approve-rejected-403 → manager-approves → manager-self-approve-rejected-403 → scoped-list → audit-log-403-then-200).

---

## Retrieval quality: dense vs. hybrid

`RETRIEVAL_MODE=dense|hybrid` in `.env` toggles between pure embedding search and BM25+dense fused with Reciprocal Rank Fusion (`rag/hybrid_retriever.py`). Measure it yourself:

```bash
python -m scripts.ingest_langchain
python -m scripts.eval_retrieval eval/qa.jsonl --k 3
```

Measured on this repo's actual corpus (6 docs / 21 chunks, 42 source-labeled questions in `eval/qa.jsonl`):

| Mode | Hit-rate @k=1 | Hit-rate @k=3 |
|---|---|---|
| Dense | 42/42 = 100% | 42/42 = 100% |
| Hybrid | 42/42 = 100% | 42/42 = 100% |

**Honest reading:** on this corpus, dense retrieval alone already nails it — each policy topic maps almost 1:1 to a single source document, so there's no ambiguity for embeddings to get wrong. That's an expected result, not a null finding for hybrid: the value of lexical (BM25) retrieval shows up on **exact-term queries** — bare acronyms, policy numbers, IDs — where embedding similarity can be inconsistent even when it isn't here. `eval/qa.jsonl` includes bare acronym queries (`FMLA`, `MDM`) specifically to probe this; both retrievers currently handle them correctly on this small corpus, but hybrid is the one with a mechanism (exact lexical match) guaranteeing it rather than a happy accident of the corpus being small and clean. At a larger scale, or with more topically-overlapping documents, expect dense-only to show a real gap.

One real bug the hybrid path surfaced along the way: `BM25Retriever`'s default tokenizer is plain `str.split()` — no lowercasing, no punctuation stripping — so a query for `FMLA` never matched `(FMLA).` in the corpus. Fixed with a proper word-boundary tokenizer (`rag/hybrid_retriever.py::_tokenize`); regression-tested in `tests/test_retrieval.py`.

---

## Groundedness guardrail

`rag/groundedness.py` extracts numeric/date tokens from a generated answer and checks each one actually appears in the retrieved context, before the answer is returned. This catches the most common and most costly failure mode for a policy bot — a confidently wrong number (wrong PTO day count, wrong dollar amount) — for zero extra LLM calls and zero extra latency.

```python
>>> check_groundedness("You get 25 sick days.", "Employees receive 10 paid sick days annually.")
GroundednessResult(grounded=False, coverage=0.0, unsupported_claims=['25'])
```

When an answer fails the check, it's swapped for an explicit low-confidence fallback rather than shown to the user. Unit-tested in `tests/test_groundedness.py`.

---

## End-to-end answer-quality eval

`scripts/eval_via_api.py` runs `eval/qa.jsonl` against a live `/agent` endpoint and checks `must_contain`/`must_not` on the final answer — this is the one eval that needs a real `GROQ_API_KEY` (it makes live LLM calls), so it isn't part of CI:

```bash
uvicorn api.app:app --port 8000 &
python -m scripts.eval_via_api eval/qa.jsonl
```

*(Numbers not included here — this development environment didn't have a `GROQ_API_KEY` configured. Run the command above with your own key to populate this; everything else in this README — retrieval hit-rate, the RBAC/audit trace, the 45-test suite — was actually executed, not estimated.)*

---

## Observability

`observability.py` emits one structured JSON line per HTTP request (method, path, status, latency, request id) and per agent step (LLM call latency + token usage where the model reports it, tool-call latency, per-run summary). Sample, captured from a real run:

```json
{"ts": "2026-09-22T18:42:37", "level": "INFO", "logger": "policybot.http", "message": "request", "request_id": "293b5c34", "method": "POST", "path": "/auth/login", "status_code": 200, "latency_ms": 176.9}
```

---

## Testing

```bash
pytest              # 45 tests: leave-request logic, holiday logic, authz rules,
                     # groundedness, actor-scoped agent tools, full API RBAC flows,
                     # retrieval correctness (skipped if the vectorstore isn't built)
```

CI (`.github/workflows/ci.yml`) builds the vectorstore with local embeddings (no secret required — `GROQ_API_KEY` only needs to be *present*, since `ChatGroq`'s constructor validates that and nothing in the suite calls `.invoke()`), runs the suite, and separately lints + builds the frontend.

---

## Project structure
```
policyBot/
├── agent/        # ReAct planning loop (react_agent.py) + actor-scoped tools (tools.py)
├── api/          # FastAPI app: auth, leave requests, holidays, chat/agent, audit log
├── auth/         # JWT, seeded users, RBAC rules, audit log
├── db/           # SQLAlchemy models + session (SQLite)
├── frontend/     # React + Tailwind chat UI
├── observability.py
├── rag/          # splitter, embeddings, dense + hybrid retrieval, groundedness
├── scripts/      # ingest, db_init, eval_retrieval, eval_via_api
├── slack/        # Slack Events API integration
├── tools/        # doc_search, qa_chain, leave_request, holiday_check
├── tests/        # pytest suite
└── eval/qa.jsonl # labeled Q&A set used by both eval scripts
```

## Example queries
- *How many PTO days in Year 1?*
- *What is the PTO carryover limit?*
- *Do I accrue PTO during unpaid leave?*
- *Request PTO from 2025-10-02 to 2025-10-04*
- *List my leave requests*

## Demo
![PolicyBot Demo](demo/one.png)
![PolicyBot Demo](demo/two.png)

---

## Author
Pranshav Patel
North Carolina State University
Master's in Computer Science
