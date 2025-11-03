# app/main.py
import os, re, json, urllib.parse, time, threading
from typing import List, Dict, Any, Optional
from collections import deque

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator

from openai import AsyncOpenAI

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


import asyncio

async def rewrite_query_async(user_query: str, client) -> str:
    """
    Rescrie întrebarea în română clară, cu diacritice și termeni oficiali.
    Corectează greșeli, dar nu adaugă informații noi.
    """
    try:
        system = (
            "Rescrie întrebarea în limba română literară, cu diacritice, "
            "clară și potrivită pentru căutare semantică. "
            "Corectează greșelile ortografice și extinde termenii informali "
            "la forma lor oficială (ex: 'buletin' -> 'carte de identitate'). "
            "Nu adăuga informații noi și nu răspunde la întrebare."
        )
        completion = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_query},
            ],
            temperature=0.1,
            max_tokens=80,
        )
        rewritten = completion.choices[0].message.content.strip()
        return rewritten
    except Exception as e:
        print(f"[rewrite_query_async] Warning: {e}")
        return user_query  # fallback dacă GPT e indisponibil

# =========================
# Config
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
INDEX_PATH = os.path.join(ROOT_DIR, "index.html")

CHROMA_PATH    = os.getenv("CHROMA_PATH", "app/chroma_db_bge_m3")
COLLECTION     = os.getenv("COLLECTION_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME     = os.getenv("GEN_MODEL", "gpt-4o-mini")
TOP_K          = int(os.getenv("TOP_K", "6"))
DISTANCE_MAX   = float(os.getenv("DISTANCE_MAX", "1.0"))
MAX_WORDS      = int(os.getenv("MAX_WORDS", "120"))

NO_ANS = "Nu am găsit documente relevante în baza locală…PublicAI răspunde doar cu informații publice ale Primăriei Sector 2"

# =========================
# Clients
# =========================
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

embedding_fn = SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-m3")
chroma = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))

if COLLECTION:
    collection = chroma.get_collection(COLLECTION, embedding_function=embedding_fn)
else:
    cols = chroma.list_collections()
    if not cols:
        raise RuntimeError("Nu există colecții în Chroma la CHROMA_PATH.")
    collection = chroma.get_collection(cols[0].name, embedding_function=embedding_fn)

# CrossEncoder re-ranker


# =========================
# Helpers — memorie pe prompt + fallback LRU în RAM
# =========================
# LRU per session_id: max 10 mesaje (user/assistant), TTL 60 min
_MEM: Dict[str, Dict[str, Any]] = {}
_MEM_LOCK = threading.Lock()
_MEM_TTL_SEC = 3600
_MEM_MAX_MSGS = 10

def _now() -> float:
    return time.time()

def _cleanup_mem():
    now = _now()
    stale = [sid for sid, rec in _MEM.items() if now - rec.get("t", 0) > _MEM_TTL_SEC]
    for sid in stale:
        _MEM.pop(sid, None)

def _get_mem_hist(sid: str) -> List[Dict[str, str]]:
    if not sid:
        return []
    with _MEM_LOCK:
        rec = _MEM.get(sid)
        if not rec:
            return []
        rec["t"] = _now()
        return list(rec["hist"])

def _save_mem_msg(sid: str, role: str, content: str):
    if not sid:
        return
    with _MEM_LOCK:
        _cleanup_mem()
        rec = _MEM.setdefault(sid, {"hist": deque(maxlen=_MEM_MAX_MSGS), "t": _now()})
        rec["hist"].append({"role": role, "content": content})
        rec["t"] = _now()

def summarize_url(url: str) -> str:
    try:
        u = urllib.parse.urlparse(url)
        segs = [s for s in u.path.split("/") if s]
        short_path = "/".join(segs[:2]) if segs else ""
        base = u.netloc.replace("www.", "")
        return f"{base}/{short_path}" if short_path else base
    except:
        return url

def system_prompt() -> str:
    # Păstrăm aceleași reguli de răspuns
    return (
        "Ești PublicAI, asistent al unei instituții publice.\n"
        "- Folosește EXCLUSIV fragmentele furnizate (RAG). Nu inventa.\n"
        "- Ton politicos, clar. Max 100 de cuvinte.\n"
        "- Structură: 1) răspuns direct în 1–2 fraze; 2) dacă e util, 1–3 puncte cheie.\n"
        "- Dacă informația nu există în fragmente, răspunde EXACT:\n"
        "  «Nu am găsit documente relevante în baza locală…PublicAI răspunde doar cu informații publice ale Primăriei Sector 2»\n"
       
    )

def simple_sent_split(text: str) -> List[str]:
    """
    Împarte textul în propoziții, dar protejează abrevierile frecvente
    (ex: Str., Nr., Bd., Bl., Dl., Dna., Dr.) pentru a evita tăierea incorectă.
    """
    if not text:
        return []

    # Protejăm abrevierile înlocuind punctul cu un simbol temporar
    ABBR = r"\b(?:Str|Strada|Bd|Bld|Bl|Nr|nr|Dl|Dna|Dr)\."
    t = re.sub(ABBR, lambda m: m.group(0).replace(".", "⟂"), text.strip())

    # Split după ., !, ? urmate de spațiu (dar nu în abrevieri)
    parts = re.split(r"(?<=[.!?])\s+", t)

    # Restaurăm simbolul ⟂ înapoi în .
    parts = [p.replace("⟂", ".").strip() for p in parts if p.strip()]
    return parts


def support_sents(doc: str, question: str, limit_words=90) -> str:
    sents = simple_sent_split(doc)
    q_terms = [w.lower() for w in re.findall(r"\w+", question) if len(w) > 2]
    picked = []
    for s in sents:
        low = s.lower()
        if any(t in low for t in q_terms):
            picked.append(s.strip())
        if len(" ".join(picked).split()) >= limit_words:
            break
    if not picked:
        picked = sents[:2]
    return " ".join(picked)

def build_user_prompt(question: str, ctxs: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, c in enumerate(ctxs, 1):
        snippet = support_sents(c['doc'], question, limit_words=90)
        blocks.append(f"[Fragment {i}]\n{snippet}\n(Sursă: {c['src']})")
    fragments_text = "\n\n".join(blocks) if blocks else "(niciun fragment)"
    return (
        f"Întrebare: {question}\n\n"
        f"Fragmente (folosește DOAR acestea):\n{fragments_text}\n\n"
        f"Instrucțiuni: răspuns direct + 1–3 puncte dacă e util; maxim 100 cuvinte; nu inventa."
    )

def retrieve(q: str) -> List[Dict[str, Any]]:
    res = collection.query(
        query_texts=[q],
        n_results=max(TOP_K * 12, 60),          # un pic mai mare, ca să avem recall bun
        include=["documents", "metadatas", "distances"],
    )
    docs = res["documents"][0] if res["documents"] else []
    metas = res["metadatas"][0] if res["metadatas"] else []
    dists = res["distances"][0] if res["distances"] else []

    candidates = []
    for d, m, dist in zip(docs, metas, dists):
        if not d:
            continue
        src = (m or {}).get("source") or (m or {}).get("url") or (m or {}).get("href") or ""
        candidates.append({"doc": d.strip(), "src": src, "dist": float(dist)})

    if not candidates:
        return []

    # FĂRĂ re-rank: filtrăm ușor după distanță și ordonăm crescător (mai mic = mai relevant)
    filtered = [c for c in candidates if c["dist"] <= DISTANCE_MAX]
    ranked = sorted(filtered or candidates, key=lambda x: x["dist"])

    picked, seen = [], set()
    for c in ranked:
        key = (c["src"], c["doc"][:160])
        if key in seen:
            continue
        seen.add(key)
        picked.append(c)
        if len(picked) >= TOP_K:
            break
    return picked


def trim_words(text: str, n: int = MAX_WORDS) -> str:
    words = re.findall(r"\S+", text or "")
    return " ".join(words[:n]) + ("…" if len(words) > n else "")

# =========================
# API models
# =========================
class Msg(BaseModel):
    role: str = Field(..., description="user sau assistant")
    content: str

    @validator("role")
    def _role_ok(cls, v):
        v = v.strip()
        if v not in ("user", "assistant"):
            raise ValueError("role trebuie să fie 'user' sau 'assistant'")
        return v

class ChatIn(BaseModel):
    session_id: Optional[str] = "web"
    question: str
    # memorie pe prompt – istoric opțional trimis de UI
    history: Optional[List[Msg]] = None

# =========================
# FastAPI app
# =========================
app = FastAPI(title="PublicAI — RAG + FastAPI (memorie pe prompt, no-redis)")

static_dir = os.path.join(ROOT_DIR, "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def root():
    return FileResponse(INDEX_PATH)

@app.get("/health")
def health():
    return {"status": "ok"}

# =========================
# Utilitare istoric efectiv folosit
# =========================
def _clip_history(msgs: List[Dict[str, str]], n_pairs: int = 3) -> List[Dict[str, str]]:
    """Ultimele n_pairs (user+assistant => 2*n_pairs mesaje)."""
    return msgs[-(2 * n_pairs):] if msgs else []

def _merge_prompt_history(body: ChatIn) -> List[Dict[str, str]]:
    """
    Ia prioritar history din UI; dacă lipsește, folosește fallback LRU per session_id.
    """
    if body.history:
        hist = [{"role": m.role, "content": m.content} for m in body.history]
    else:
        hist = _get_mem_hist(body.session_id or "web")
    return _clip_history(hist, n_pairs=3)

# =========================
# /chat (sync)
# =========================
@app.post("/chat")
async def chat(body: ChatIn):
    sid = body.session_id or "web"
    q = (body.question or "").strip()
    if not q:
        return JSONResponse({"answer": "Te rog formulează o întrebare.", "citations": ""})

    # Rescriere rapidă cu GPT (corectare + sinonime)
    rewritten_q = await rewrite_query_async(q, client)
    ctxs = retrieve(rewritten_q)
    _save_mem_msg(sid, "user", rewritten_q)

    if not ctxs:
        _save_mem_msg(sid, "assistant", NO_ANS)
        return JSONResponse({"answer": NO_ANS, "citations": ""})

    history = _merge_prompt_history(body)
    messages = [{"role": "system", "content": system_prompt()}]
    messages.extend(history)
    messages.append({"role": "user", "content": build_user_prompt(rewritten_q, ctxs)})

    comp = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.1,
        presence_penalty=0.0,
        frequency_penalty=0.2,
    )
    answer = comp.choices[0].message.content or ""
    answer = trim_words(answer, MAX_WORDS).strip()
    _save_mem_msg(sid, "assistant", answer)

    main_src = ctxs[0]["src"]
    citations = f"[{summarize_url(main_src)}]({main_src})" if main_src else ""
    return JSONResponse({"answer": answer, "citations": citations})

# =========================
# /chat/stream (SSE)
# =========================
# =========================
# /chat/stream (SSE)
# =========================
@app.post("/chat/stream")
async def chat_stream(body: ChatIn):
    sid = body.session_id or "web"
    q = (body.question or "").strip()
    if not q:
        async def empty_gen():
            yield "data: Te rog formulează o întrebare.\n\n"
            yield "event: done\ndata: ok\n\n"
        return StreamingResponse(
            empty_gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # 1) Rescriere cu GPT (asincron)
    rewritten_q = await rewrite_query_async(q, client)

    # 2) Retrieve + memorie
    ctxs = retrieve(rewritten_q)
    _save_mem_msg(sid, "user", rewritten_q)

    # 3) Generatorul SSE (asincron) + iterare async a streamului OpenAI
    async def gen():
        yield "event: ping\ndata: 1\n\n"

        if not ctxs:
            yield f"data: {NO_ANS}\n\n"
            yield "event: done\ndata: ok\n\n"
            return

        history = _merge_prompt_history(body)
        messages = [{"role": "system", "content": system_prompt()}]
        messages.extend(history)
        messages.append({"role": "user", "content": build_user_prompt(rewritten_q, ctxs)})

        def count_words(s: str) -> int:
            return len(re.findall(r"\b\w+\b", s, flags=re.UNICODE))

        accum = ""
        last_ping = time.time()

        stream = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            stream=True,
            temperature=0.1,
            presence_penalty=0.0,
            frequency_penalty=0.2,
        )

        async for ev in stream:
            delta = getattr(getattr(ev.choices[0], "delta", None), "content", None)
            if not delta:
                if time.time() - last_ping > 15:
                    yield "event: ping\ndata: 1\n\n"
                    last_ping = time.time()
                continue

            tentative = (accum + delta)
            if count_words(tentative) <= MAX_WORDS:
                accum = tentative
                yield f"data: {delta}\n\n"
            else:
                # taie strict la MAX_WORDS
                cur = accum
                stop_idx = 0
                for i, ch in enumerate(delta):
                    cur += ch
                    if count_words(cur) >= MAX_WORDS:
                        stop_idx = i + 1
                        break
                if stop_idx > 0:
                    slice_ = delta[:stop_idx]
                    accum += slice_
                    yield f"data: {slice_}\n\n"
                break

            if time.time() - last_ping > 15:
                yield "event: ping\ndata: 1\n\n"
                last_ping = time.time()

        ans = accum.strip()
        _save_mem_msg(sid, "assistant", ans)

        main_src = ctxs[0]["src"]
        src_label = summarize_url(main_src) if main_src else "necunoscută"
        yield f"data: Sursă: [{src_label}]({main_src})\n\n"
        yield "event: done\ndata: ok\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
