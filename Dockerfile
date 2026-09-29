FROM python:3.11-slim
WORKDIR /app
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