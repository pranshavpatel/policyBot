# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Setup (run once per environment)
```bash
pip install -r requirements.txt
cp .env.example .env               # fill in GROQ_API_KEY at minimum
python -m scripts.db_init          # creates data/policybot.db (leave_requests, audit_log tables)
python -m scripts.ingest_langchain # chunks data/policies/*.md into Chroma (local embeddings, no API key needed)
```
Always invoke scripts as `python -m scripts.<name>`, not `python scripts/<name>.py` — the latter has no package root on `sys.path` and fails with `ModuleNotFoundError: No module named 'rag'` (or `db`, `agent`, etc.).

### Run
```bash
uvicorn api.app:app --reload --port 8000          # backend
cd frontend && npm install && npm run dev          # frontend, http://localhost:5173
docker compose up --build                          # both, containerized
```
Demo logins (seeded in `auth/users.py`): `alice`/`alice123` and `bob`/`bob123` (employee), `manager1`/`manager123` (manager).

### Tests
```bash
pytest                              # full suite (pytest.ini sets pythonpath=. — no env setup needed)
pytest tests/test_leave_request.py  # a single file
pytest tests/test_api.py::test_manager_can_approve_others_but_not_own  # a single test
```
- `tests/conftest.py` points `DATABASE_URL` at a throwaway temp sqlite file — it never touches `data/policybot.db`.
- `GROQ_API_KEY`/`JWT_SECRET` just need to be *present* (any dummy value) for tests to run — `ChatGroq`'s constructor validates a key exists but nothing in the suite calls `.invoke()`, so no real Groq credential or network call happens in the test suite.
- Tests in `tests/test_retrieval.py` and the real-model tests in `tests/test_reranker.py`/`test_retrieval.py` auto-skip if the vectorstore hasn't been built (`python -m scripts.ingest_langchain` first).

### Lint / build (frontend)
```bash
cd frontend && npm run lint && npm run build
```

### Evals (not part of `pytest`)
```bash
python -m scripts.eval_retrieval eval/qa.jsonl --k 1 --k 3 --k 5 --rerank --show-misses
# ^ retrieval-only: MRR/Recall@k/Precision@k for dense vs hybrid vs +reranking. No API key, cheap.

uvicorn api.app:app --port 8000 &
python -m scripts.eval_via_api eval/qa.jsonl
# ^ end-to-end: real LLM calls against a live server, needs GROQ_API_KEY. Not in CI for that reason.
```

### CI
`.github/workflows/ci.yml` — one job builds the vectorstore (local embeddings only) and runs `pytest`; a second job lints + builds the frontend. Both run on every PR into `main`.

## Architecture

**Two entry points into the same authorization and business logic.** The REST API (`api/app.py`) and the chat agent (`agent/react_agent.py` → `agent/tools.py`) are not independent — both funnel every leave-request state change through the same rules in `auth/authz.py` and write to the same audit log (`auth/audit.py`). The agent receives the authenticated caller as `actor: {"username", "role"}` (injected by `run_agent`, never trusted from the LLM's own plan JSON) and threads it into `agent/tools.py`'s `ACTOR_SCOPED_TOOLS` via an `args["_actor"]` convention, mirroring what `api/app.py` does with FastAPI's `Depends(get_current_user)`. When changing an authorization rule, change `auth/authz.py` once — both call sites pick it up.

**Retrieval is a single configurable pipeline**, not three separate code paths. `rag/retrieval.py::get_configured_retriever(k)` is the only place `tools/doc_search.py` and `tools/qa_chain.py` should get a retriever from — it composes `config.RETRIEVAL_MODE` (`dense` | `hybrid`, `rag/hybrid_retriever.py`'s Reciprocal-Rank-Fusion of BM25 + embeddings) with `config.RERANK_ENABLED` (cross-encoder second-stage reranking, `rag/reranker.py`) rather than either being a special case. Adding a new retrieval strategy means extending this entrypoint, not adding a new import elsewhere.

**Groundedness is checked at two different points, deliberately.** `tools/qa_chain.py`'s `QAChain.invoke()` checks the RAG answer against its own retrieved context (`rag/groundedness.py`) before returning from a single `rag_answer` tool call. Separately, `agent/react_agent.py::_ground_final_answer()` re-checks *whatever text is about to be the final agent response* — the planner's own `"final"` action or the post-tool-loop `synthesize()` step — against the pooled context from every `rag_answer` call made during that run. The second check exists because a later planning step can restate/elaborate on a properly-grounded tool answer in new words that reintroduce a fabrication, or skip retrieval entirely and invent an answer from the model's own "knowledge" (this happened in production testing — see git history around the VPN-support hallucination fix). If you add a new way for the agent to produce a final answer, route it through `_ground_final_answer()`.

**langchain-core version incompatibilities are real in this repo's resolved dependency set** — `langchain.chains.RetrievalQA` and `langchain.schema.Document` are both broken here (`langchain.chains.base.Chain` imports `langchain_core.memory.BaseMemory`, which doesn't exist in the resolved `langchain-core`). `tools/qa_chain.py::QAChain` is a small hand-rolled retrieve→prompt→LLM chain for exactly this reason — don't reach for `RetrievalQA` or `langchain.schema` imports; use `langchain_core.documents.Document`, `langchain_core.prompts.PromptTemplate`, and `langchain_chroma.Chroma` (not `langchain_community.vectorstores.Chroma`, which is deprecated and triggers a warning on every call).

**`config.GROQ_MODEL` needs to be a currently-available Groq model** — the original default (`llama-3.1-8b-instant`) was retired from Groq's catalog mid-project and every `/agent` call 500'd until it was caught by actually running the eval against a live server. If agent calls start failing with a `groq.NotFoundError: model_not_found`, check `GET https://api.groq.com/openai/v1/models` with a live key before assuming anything else is broken.

**The ReAct loop (`agent/react_agent.py::run_agent`) can be slow (15-40s+) and can re-plan 2-3 steps for one question** — visible in the structured logs' `agent_run.steps` field. This is a known, not-yet-fixed characteristic (see `PLANNER_PROMPT`'s rules and `max_steps`), not a regression — don't assume a slow agent response means something you touched broke.

**Observability**: `observability.py` provides `get_logger`/`log_event` (structured JSON, one line per HTTP request or agent step) and `RequestLoggingMiddleware`. Use these rather than ad-hoc `print`/`logging` calls when adding new request- or agent-level instrumentation, so log lines stay parseable as JSON.
