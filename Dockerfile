# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates cmake \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) deps separate pt caching
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 2) codul + UI
COPY index.html /index.html
COPY app /app/app

# 3) Baza Chroma preconstruită (baked-in) – copiaz-o în repo înainte de build
COPY app/chroma_db_bge_m3 /app/chroma_db_bge_m3

# 4) Pre-warm modele la build (bge-m3 + cross-encoder)
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer("BAAI/bge-m3")
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("OK: models cached")
PY

# 5) defaults
ENV PORT=8080 \
    HOST=0.0.0.0 \
    CHROMA_PATH=app/chroma_db_bge_m3 \
    TOP_K=4 \
    DISTANCE_MAX=1.2 \
    MAX_WORDS=100 \
    GEN_MODEL=gpt-4o-mini

# 6) server – SSE friendly
CMD exec uvicorn app.main:app --host ${HOST} --port ${PORT} --workers 1 --proxy-headers --forwarded-allow-ips="*"
