import os
from dotenv import load_dotenv

load_dotenv()

# LLM: "groq" (default, hosted) or "local" (any OpenAI-compatible server —
# Ollama, vLLM, llama.cpp's server — reached via LOCAL_LLM_BASE_URL). See
# agent/llm.py for the factory both agent/react_agent.py and
# tools/qa_chain.py go through, so this is a single config flip, not a
# code change, the same pattern as RETRIEVAL_MODE below.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# llama-3.1-8b-instant was retired from Groq's catalog; verified against
# https://api.groq.com/openai/v1/models with a live key (2026-09-22).
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# Local LLM (LLM_PROVIDER=local): any server exposing an OpenAI-compatible
# /v1/chat/completions endpoint. Defaults match Ollama running locally;
# point LOCAL_LLM_BASE_URL at a GPU workstation's reachable address (LAN
# IP, Tailscale, SSH tunnel — this project doesn't set up that network
# path) to run inference there instead. LOCAL_LLM_API_KEY is typically
# unchecked by these servers but most OpenAI-compatible clients require a
# non-empty string.
LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "qwen3:8b")
LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "ollama")
# Bounds on a single local LLM call. Without them, one call can hang the
# whole server: Ollama's default is unlimited generation (num_predict=-1,
# with context shifting), and qwen3 at temperature 0 can fall into an
# endless repetition loop — this happened on a real eval run, stuck on the
# same question twice. The HTTP client had no timeout either, so the call
# never returned, and every later request queued behind it in Ollama until
# uvicorn's threadpool was exhausted and even /health stopped answering.
# The token cap covers thinking tokens too (Ollama counts them against
# num_predict), so leave headroom for reasoning models.
LOCAL_LLM_MAX_TOKENS = int(os.getenv("LOCAL_LLM_MAX_TOKENS", "4096"))
LOCAL_LLM_TIMEOUT_SECONDS = float(os.getenv("LOCAL_LLM_TIMEOUT_SECONDS", "180"))

# Embeddings
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Database
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/policybot.db")

# Chroma settings (local persistent store)
CHROMA_DIR = os.getenv("CHROMA_DIR", "vectorstore/chroma_policybot")

# Retrieval: "dense" (pure vector search) or "hybrid" (BM25 + dense, ensembled)
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "dense")

# Agent control flow: "loop" (the original hand-rolled loop,
# agent/react_agent.py::run_agent) or "graph" (LangGraph StateGraph,
# agent/graph.py::run_agent_graph) — deterministic routing after a
# successful rag_answer instead of relying on the planner prompt to decide
# to stop, plus per-user conversation memory via a checkpointer. Default
# stays "loop" so nothing changes for anyone not opting in.
AGENT_ENGINE = os.getenv("AGENT_ENGINE", "loop")

# Cross-encoder reranking: retrieve RERANK_CANDIDATES candidates from
# whichever RETRIEVAL_MODE is configured, score each (query, chunk) pair
# with a cross-encoder, and keep only the top k after reranking. Off by
# default — it's a real CPU/latency cost (a forward pass per candidate,
# not a single batched embedding lookup) and a ~90MB model download on
# first use, so it shouldn't turn on silently for everyone running the
# default config. See rag/reranker.py for why a cross-encoder over just
# using the dense retriever's own similarity score.
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "false").lower() == "true"
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "15"))

# Auth (JWT)
JWT_SECRET = os.getenv("JWT_SECRET", "dev-only-insecure-secret-change-me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))

# Groundedness guardrail: minimum fraction of numeric/date tokens in an
# answer that must also appear in the retrieved context for the answer to
# be trusted as-is (see rag/groundedness.py)
GROUNDEDNESS_MIN_COVERAGE = float(os.getenv("GROUNDEDNESS_MIN_COVERAGE", "1.0"))
