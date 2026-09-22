import os
from dotenv import load_dotenv

load_dotenv()

# LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Embeddings
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Database
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/policybot.db")

# Chroma settings (local persistent store)
CHROMA_DIR = os.getenv("CHROMA_DIR", "vectorstore/chroma_policybot")

# Retrieval: "dense" (pure vector search) or "hybrid" (BM25 + dense, ensembled)
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "dense")

# Auth (JWT)
JWT_SECRET = os.getenv("JWT_SECRET", "dev-only-insecure-secret-change-me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))

# Groundedness guardrail: minimum fraction of numeric/date tokens in an
# answer that must also appear in the retrieved context for the answer to
# be trusted as-is (see rag/groundedness.py)
GROUNDEDNESS_MIN_COVERAGE = float(os.getenv("GROUNDEDNESS_MIN_COVERAGE", "1.0"))
