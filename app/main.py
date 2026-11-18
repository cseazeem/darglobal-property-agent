"""FastAPI entrypoint: serves the chat UI and the agent API."""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config
from .agent import run_agent
from .retriever import get_retriever
from .tools import meta

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

app = FastAPI(title=config.APP_TITLE)
STATIC = config.ROOT / "app" / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")

# Simple per-IP rate limit so a public demo URL cannot burn the free-model quota.
_hits: dict[str, deque] = defaultdict(deque)
RATE_LIMIT, WINDOW = 12, 60.0


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[Message] = []


@app.on_event("startup")
def warm_up():
    t0 = time.time()
    r = get_retriever()
    r.search("warm up", k=1)  # loads the embedding model once
    log.info("Retriever ready: %d chunks in %.1fs; models=%s",
             len(r.chunks), time.time() - t0, config.OPENROUTER_MODELS)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "llm_configured": bool(config.OPENROUTER_API_KEY)}


@app.get("/api/meta")
def get_meta():
    m = meta()
    return {"snapshot": m["snapshot"], "projects": m["n_projects"], "listings": m["n_listings"],
            "chunks": m["n_chunks"], "cities": m["wasalt_cities"], "models": config.OPENROUTER_MODELS}


@app.post("/api/chat")
def chat(req: ChatRequest, request: Request):
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "?")
    ip = ip.split(",")[0].strip()
    q, now = _hits[ip], time.time()
    while q and now - q[0] > WINDOW:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        raise HTTPException(429, "Too many requests — please wait a minute.")
    q.append(now)
    try:
        return run_agent(req.message, [m.model_dump() for m in req.history], meta())
    except Exception as e:
        log.exception("agent failed")
        raise HTTPException(502, f"The language model is unavailable right now ({e}). Please retry.")
