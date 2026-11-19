"""FastAPI entrypoint: serves the chat UI and the agent API."""
from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, Field

from . import config
from .agent import run_agent, run_agent_stream
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


class Lead(BaseModel):
    project: str = Field(min_length=1, max_length=200)
    project_url: str | None = None
    name: str = Field(min_length=1, max_length=120)
    email: EmailStr | None = None
    phone: str | None = Field(default=None, max_length=40)
    budget: str | None = Field(default=None, max_length=120)
    preferred_contact: str | None = Field(default=None, max_length=60)
    notes: str | None = Field(default=None, max_length=1000)


def _db() -> sqlite3.Connection:
    con = sqlite3.connect(config.LEADS_DB)
    con.execute("""CREATE TABLE IF NOT EXISTS leads (
        ref TEXT PRIMARY KEY, created REAL, project TEXT, project_url TEXT, name TEXT,
        email TEXT, phone TEXT, budget TEXT, preferred_contact TEXT, notes TEXT)""")
    return con


def _client_ip(request: Request) -> str:
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "?")
    return ip.split(",")[0].strip()


def _rate_limit(request: Request) -> None:
    q, now = _hits[_client_ip(request)], time.time()
    while q and now - q[0] > WINDOW:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        raise HTTPException(429, "Too many requests — please wait a minute.")
    q.append(now)


@app.on_event("startup")
def warm_up():
    t0 = time.time()
    r = get_retriever()
    r.search("warm up", k=1)  # loads the embedding model once
    _db().close()
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
    _rate_limit(request)
    try:
        return run_agent(req.message, [m.model_dump() for m in req.history], meta())
    except Exception as e:
        log.exception("agent failed")
        raise HTTPException(502, str(e))


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest, request: Request):
    """Newline-delimited JSON events: step / step_done / token / discard / lead_draft / done / error."""
    _rate_limit(request)
    history = [m.model_dump() for m in req.history]

    def gen():
        for ev in run_agent_stream(req.message, history, meta()):
            yield json.dumps(ev, ensure_ascii=False) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/leads")
def confirm_lead(lead: Lead, request: Request):
    """Called only when the user presses Confirm on the draft card (human-in-the-loop)."""
    _rate_limit(request)
    if not (lead.email or lead.phone):
        raise HTTPException(422, "An email or phone number is required.")
    ref = "DG-" + uuid.uuid4().hex[:8].upper()
    with _db() as con:
        con.execute("INSERT INTO leads VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (ref, time.time(), lead.project, lead.project_url, lead.name,
                     str(lead.email) if lead.email else None, lead.phone, lead.budget,
                     lead.preferred_contact, lead.notes))
    log.info("lead %s stored for project=%s", ref, lead.project)  # no PII in logs
    return {"reference": ref, "status": "recorded",
            "note": "Demo only: stored in this app's database, not sent to DarGlobal."}
