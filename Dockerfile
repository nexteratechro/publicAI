# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# --- instalare pachete de bază ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git curl ca-certificates cmake && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- instalare requirements ---
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# --- copiere cod aplicație ---
COPY index.html /index.html
COPY app /app/app

# --- copiere baza Chroma preconstruită (opțional, dar recomandat) ---
COPY app/chroma_db_bge_m3 /app/chroma_db_bge_m3

# --- pre-încălzire modele Hugging Face ---
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer("BAAI/bge-m3")
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("✅ Modelele au fost pre-încărcate cu succes.")
PY

# --- configurare implicită ---
ENV PORT=8080 \
    HOST=0.0.0.0 \
    CHROMA_PATH=app/chroma_db_bge_m3 \
    TOP_K=4 \
    DISTANCE_MAX=1.2 \
    MAX_WORDS=100 \
    GEN_MODEL=gpt-4o-mini

# --- comandă lansare server ---
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips=*"]
