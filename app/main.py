# app/main.py
import os, re, json, urllib.parse, time, threading
from typing import List, Dict, Any, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import redis
from openai import OpenAI

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import CrossEncoder

# =========================
# Config
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
INDEX_PATH = os.path.join(ROOT_DIR, "index.html")

CHROMA_PATH    = os.getenv("CHROMA_PATH", "app/chroma_db_bge_m3")
COLLECTION     = os.getenv("COLLECTION_NAME")
REDIS_URL      = os.getenv("REDIS_URL", "redis://localhost:6379/0")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME     = os.getenv("GEN_MODEL", "gpt-4o-mini")
TOP_K          = int(os.getenv("TOP_K", "4"))
DISTANCE_MAX   = float(os.getenv("DISTANCE_MAX", "1.2"))
MAX_WORDS      = int(os.getenv("MAX_WORDS", "100"))

NO_ANS = "Nu am găsit documente relevante în baza locală…PublicAI răspunde doar cu informații publice ale Primăriei Sector 2"

# =========================
# Clients
# =========================
client = OpenAI(api_key=OPENAI_API_KEY)
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

embedding_fn = SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-m3")
chroma = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))

if COLLECTION:
    collection = chroma.get_collection(COLLECTION, embedding_function=embedding_fn)
else:
    cols = chroma.list_collections()
    if not cols:
        raise RuntimeError("Nu există colecții în Chroma la CHROMA_PATH.")
    collection = chroma.get_collection(cols[0].name, embedding_function=embedding_fn)

# CrossEncoder re-ranker (calitate mai bună)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# =========================
# Helpers (session memory cu fallback)
# =========================
MEM_FALLBACK: Dict[str, List[Dict[str, str]]] = {}
MEM_LOCK = threading.Lock()

def _redis_ok() -> bool:
    try:
        redis_client.ping()
        return True
    except Exception:
        return False

def s_key(sid: str) -> str:
    return f"chat:session:{sid}"

def get_hist(sid: str) -> List[Dict[str, str]]:
    if not sid:
        return []
    if _redis_ok():
        try:
            raw = redis_client.get(s_key(sid))
            return json.loads(raw) if raw else []
        except Exception:
            pass
    with MEM_LOCK:
        return MEM_FALLBACK.get(sid, [])

def save_msg(sid: str, role: str, content: str):
    if not sid:
        return
    if _redis_ok():
        try:
            hist = get_hist(sid)
            hist.append({"role": role, "content": content})
            if len(hist) > 10:
                hist = hist[-10:]
            redis_client.set(s_key(sid), json.dumps(hist), ex=3600)
            return
        except Exception:
            pass
    with MEM_LOCK:
        MEM_FALLBACK.setdefault(sid, [])
        MEM_FALLBACK[sid].append({"role": role, "content": content})
        if len(MEM_FALLBACK[sid]) > 10:
            MEM_FALLBACK[sid] = MEM_FALLBACK[sid][-10:]

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
    return (
        "Ești PublicAI, asistent al unei instituții publice.\n"
        "- Folosește EXCLUSIV fragmentele furnizate (RAG). Nu inventa.\n"
        "- Ton politicos, clar. Max 100 de cuvinte.\n"
        "- Structură: 1) răspuns direct în 1–2 fraze; 2) dacă e util, 1–3 puncte cheie.\n"
        "- Dacă informația nu există în fragmente, răspunde EXACT:\n"
        "  «Nu am găsit documente relevante în baza locală…PublicAI răspunde doar cu informații publice ale Primăriei Sector 2»\n"
        "- La final voi adăuga eu: «Sursă: {link} (URL)». Respectă contextul sesiunii."
    )

def simple_sent_split(text: str) -> List[str]:
    return re.split(r"(?<=[\.\!\?])\s+", (text or "").strip())

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
    # candidati mai mulți pentru re-rank
    res = collection.query(
        query_texts=[q],
        n_results=max(TOP_K * 5, 20),
        include=["documents", "metadatas", "distances"],
    )
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    candidates = []
    for d, m, dist in zip(docs, metas, dists):
        if not d:
            continue
        src = (m or {}).get("source") or (m or {}).get("url") or (m or {}).get("href") or ""
        candidates.append({"doc": d.strip(), "src": src, "dist": float(dist)})

    if not candidates:
        return []

    # re-rank
    pairs = [(q, c["doc"]) for c in candidates]
    scores = reranker.predict(pairs)
    for c, s in zip(candidates, scores):
        c["score"] = float(s)

    filtered = [c for c in candidates if c["dist"] <= DISTANCE_MAX * 1.5]
    ranked = sorted(filtered or candidates, key=lambda x: x["score"], reverse=True)

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
# API
# =========================
class ChatIn(BaseModel):
    session_id: Optional[str] = "web"
    question: str

app = FastAPI(title="PublicAI — RAG + FastAPI + Redis (clean stream)")

static_dir = os.path.join(ROOT_DIR, "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def root():
    return FileResponse(INDEX_PATH)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(body: ChatIn):
    sid = body.session_id or "web"
    q = body.question.strip()
    ctxs = retrieve(q)
    save_msg(sid, "user", q)

    if not ctxs:
        save_msg(sid, "assistant", NO_ANS)
        return JSONResponse({"answer": NO_ANS, "citations": ""})

    history = get_hist(sid)[-6:]
    messages = [{"role": "system", "content": system_prompt()}]
    messages.extend(history)
    messages.append({"role": "user", "content": build_user_prompt(q, ctxs)})

    comp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.1,
        presence_penalty=0.0,
        frequency_penalty=0.2,
    )
    answer = comp.choices[0].message.content or ""
    answer = trim_words(answer, MAX_WORDS).strip()
    save_msg(sid, "assistant", answer)

    main_src = ctxs[0]["src"]
    citations = f"[{summarize_url(main_src)}]({main_src})" if main_src else ""
    return JSONResponse({"answer": answer, "citations": citations})

@app.post("/chat/stream")
def chat_stream(body: ChatIn):
    sid = body.session_id or "web"
    q = body.question.strip()
    ctxs = retrieve(q)
    save_msg(sid, "user", q)

    def gen():
        yield "event: ping\ndata: 1\n\n"

        if not ctxs:
            yield f"data: {NO_ANS}\n\n"
            yield "event: done\ndata: ok\n\n"
            return

        history = get_hist(sid)[-6:]
        messages = [{"role": "system", "content": system_prompt()}]
        messages.extend(history)
        messages.append({"role": "user", "content": build_user_prompt(q, ctxs)})

        def count_words(s: str) -> int:
            return len(re.findall(r"\b\w+\b", s, flags=re.UNICODE))

        wc = 0
        accum = ""
        last_ping = time.time()

        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            stream=True,
            temperature=0.1,
            presence_penalty=0.0,
            frequency_penalty=0.2,
        )

        for ev in stream:
            delta = getattr(getattr(ev.choices[0], "delta", None), "content", None)
            if not delta:
                if time.time() - last_ping > 15:
                    yield "event: ping\ndata: 1\n\n"
                    last_ping = time.time()
                continue

            tentative = accum + delta
            words_now = count_words(tentative)

            if words_now <= MAX_WORDS:
                accum = tentative
                yield f"data: {delta}\n\n"
            else:
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
        save_msg(sid, "assistant", ans)

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

