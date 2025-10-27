# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS base

# ----------------------------------------------------
# --- CONFIGURARE VARIABILE DE MEDIU DE BAZĂ (STRAT STABIL) ---
# ----------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# ----------------------------------------------------
# --- DEPENDENȚE DE SISTEM (STRAT STABIL) ---
# ----------------------------------------------------
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

# ----------------------------------------------------
# --- DEPENDENȚE PYTHON (STRAT STABIL - CACHING) ---
# Se re-execută DOAR dacă requirements.txt se schimbă.
# ----------------------------------------------------
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ----------------------------------------------------
# --- PRE-ÎNCĂLZIRE MODELE ȘI CLEANUP (STRAT STABIL - CACHING) ---
# Folosește cache-ul, chiar dacă codul aplicației s-a schimbat.
# ----------------------------------------------------
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-m3'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
print('OK: models cached')"

# Cleanup cache (reduce dimensiunea imaginii finale)
RUN rm -rf /root/.cache/pip \
    && rm -rf /root/.cache/torch/* \
    && rm -rf /tmp/*

# ----------------------------------------------------
# --- BAZA CHROMA (STRAT CU SCHIMBĂRI RARE) ---
# Copiază baza de date vectorială (embedding-urile)
# ----------------------------------------------------
COPY app/chroma_db_bge_m3 /app/chroma_db_bge_m3

# ----------------------------------------------------
# --- FIȘIERE APLICAȚIE (STRAT CU SCHIMBĂRI FRECVENTE) ---
# ----------------------------------------------------
COPY index.html /app/index.html
COPY app /app/app


# ----------------------------------------------------
# --- DEFAULT-URI RULARE (RUNTIME ENV) ---
# ----------------------------------------------------
ENV PORT=8080 \
    HOST=0.0.0.0 \
    CHROMA_PATH=app/chroma_db_bge_m3 \
    TOP_K=4 \
    DISTANCE_MAX=1.2 \
    MAX_WORDS=100 \
    GEN_MODEL=gpt-4o-mini # Poate fi înlocuit cu un model Gemini Vertex AI

# ----------------------------------------------------
# --- START SERVER ---
# ----------------------------------------------------

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips=*"]
