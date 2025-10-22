# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS base

# Env
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        ca-certificates \
        cmake \
        zstd \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalează dependențele înainte de cod (cache mai bun)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copiază aplicația și UI
COPY index.html /app/index.html
COPY app /app/app

# Copiază baza Chroma baked-in (folderul trebuie să existe în repo)
COPY app/chroma_db_bge_m3 /app/chroma_db_bge_m3

# Pre-încălzire modele fără heredoc (evită parse error la FROM)
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-m3'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
print('OK: models cached')"

# Cleanup cache (diminuează imaginea)
RUN rm -rf /root/.cache/pip \
    && rm -rf /root/.cache/torch/* \
    && rm -rf /tmp/*

# Runtime env
ENV PORT=8080 \
    HOST=0.0.0.0 \
    CHROMA_PATH=app/chroma_db_bge_m3 \
    TOP_K=4 \
    DISTANCE_MAX=1.2 \
    MAX_WORDS=100 \
    GEN_MODEL=gpt-4o-mini

# Start server (exec form; bun pentru Cloud Run, SSE ok)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips=*"]

