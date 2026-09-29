FROM python:3.11-slim
WORKDIR /app

# Cap BLAS/OpenMP thread pools instead of letting them auto-size to the
# container's *visible* CPU count (8, even on a resource-limited host) —
# each thread carries its own working buffers. Tested this in isolation
# against a real 512MB-limited container: it did NOT measurably reduce
# idle memory on its own (documented honestly, not overstated) — so this
# is kept as cheap, standard containerization practice (one thread has no
# reason to fight over cores it won't get anyway), not claimed as the fix
# for the OOM below. See rag/embeddings.py's EMBED_THREADS for the same
# idea applied to the embedding model specifically.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# torch (a transitive dependency of sentence-transformers, used only by
# the optional cross-encoder reranker — see rag/retrieval.py) resolves to
# a CUDA-enabled build by default on Linux, pulling ~1.5GB+ of NVIDIA
# runtime packages (cublas, cudnn, nccl, ...) this CPU-only deploy target
# never uses. Installing the CPU wheel explicitly first means the later
# `pip install -r requirements.txt` sees torch already satisfied and
# skips the CUDA download entirely — found live: a build spent several
# extra minutes and ~2GB of transfer on packages that were never going to
# be imported.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY ./requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt
COPY . /app

# Bake the vectorstore into the image instead of building it on first
# request or requiring a manual step after deploy — local embeddings only,
# no API key needed at build time (same reason CI's ingest step doesn't
# need one). GROQ_API_KEY is deliberately NOT required here: config.py
# only reads it, ChatGroq isn't constructed until a request comes in.
RUN python -m scripts.ingest_langchain

EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]