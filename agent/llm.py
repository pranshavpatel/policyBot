"""Single LLM provider entrypoint — the same "config flip, not a code
change" pattern as rag/retrieval.py's RETRIEVAL_MODE. Both
agent/react_agent.py and tools/qa_chain.py should get their chat model
from here rather than constructing ChatGroq directly, so switching
LLM_PROVIDER (config.py) covers both call sites at once.

Why "local" matters here specifically, not just in the abstract: Groq's
free-tier rate limit was hit live during eval work on this project (see
README's "End-to-end answer-quality eval" section and
api/app.py's groq.RateLimitError handler) — a local model has no such
limit. It's also a real, defensible privacy story for an HR bot
specifically: policy and leave-request content never has to leave your
own infrastructure.
"""
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from config import (
    LLM_PROVIDER, GROQ_API_KEY, GROQ_MODEL,
    LOCAL_LLM_BASE_URL, LOCAL_LLM_MODEL, LOCAL_LLM_API_KEY,
)


def get_llm(temperature: float = 0):
    if LLM_PROVIDER == "local":
        # Any OpenAI-compatible server — Ollama, vLLM, llama.cpp's server —
        # works here unchanged; only LOCAL_LLM_BASE_URL needs to point at
        # it. Ollama's OpenAI-compatible endpoint is at /v1 (the default).
        return ChatOpenAI(
            base_url=LOCAL_LLM_BASE_URL,
            api_key=LOCAL_LLM_API_KEY,
            model=LOCAL_LLM_MODEL,
            temperature=temperature,
        )
    return ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=temperature)
