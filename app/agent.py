"""Tool-calling agent loop over OpenRouter (OpenAI-compatible API).

The model decides which tools to call (RAG search, structured listing search, market
stats, project lookup), sees the results, and iterates up to MAX_AGENT_STEPS before
answering. If a model is rate-limited or errors, the next model in
OPENROUTER_MODELS is tried transparently.
"""
from __future__ import annotations

import json
import logging
import time

from openai import OpenAI

from . import config
from .tools import TOOL_SCHEMAS, run_tool

log = logging.getLogger("agent")

SYSTEM_PROMPT = """You are the DarGlobal × Wasalt Property Agent, a real-estate assistant.

Your knowledge comes ONLY from tools over a snapshot of public data scraped on {snapshot}:
- DarGlobal (darglobal.co.uk): luxury branded developments in the UAE, Saudi Arabia, Oman,
  Qatar, UK, Spain and Maldives; company info, investor info, FAQs, press releases, insights.
- Wasalt (wasalt.sa): Saudi property portal — {n_listings} sample listings for sale/rent in
  Riyadh, Jeddah, Makkah, Madinah, Dammam, Al Khobar and Khamis Mushait, plus city market guides.

Rules:
1. Always call a tool before answering a factual question. Prefer:
   - list_darglobal_projects / get_darglobal_project for DarGlobal developments,
   - search_wasalt_listings for concrete Saudi listings with filters (city, budget, bedrooms),
   - wasalt_market_stats for price levels / comparisons,
   - search_knowledge for everything else (company, FAQs, news, city guides, payment plans).
   You may call several tools, or the same tool again with a refined query.
2. Answer only from tool results. If the data does not contain the answer, say so plainly and
   suggest contacting DarGlobal sales or checking wasalt.sa. Never invent prices, dates or units.
3. Cite sources as markdown links using the URLs returned by tools.
4. Wasalt listing prices are in SAR; rent prices are yearly unless stated. If a listing has a
   `price_note`, mention it. Remind users that listings are a snapshot and may have changed.
   You give information, not financial advice.
5. Reply in the user's language (English or Arabic, etc.). Be concise; use short lists or a small
   table when comparing options.
"""


def _client() -> OpenAI:
    return OpenAI(
        base_url=config.OPENROUTER_BASE_URL,
        api_key=config.OPENROUTER_API_KEY or "missing",
        timeout=config.REQUEST_TIMEOUT,
        max_retries=0,  # fail fast on 429 and fall through to the next model
        # HTTP headers must be ASCII (APP_TITLE contains "×")
        default_headers={"X-Title": "DarGlobal x Wasalt Property Agent"},
    )


def _complete(client: OpenAI, messages: list[dict], models: list[str], use_tools: bool = True):
    """Call the first model that works. Returns (response, model_used)."""
    last_err: Exception | None = None
    for model in models:
        try:
            kwargs = {"tools": TOOL_SCHEMAS} if use_tools else {}
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=0.2, **kwargs
            )
            if not resp.choices:
                raise RuntimeError(f"empty response from {model}")
            return resp, model
        except Exception as e:  # rate limit, provider error, unsupported tools, ...
            log.warning("model %s failed: %s", model, e)
            last_err = e
    raise RuntimeError(f"All models failed. Last error: {last_err}")


def run_agent(question: str, history: list[dict], meta: dict) -> dict:
    if not config.OPENROUTER_API_KEY:
        return {"answer": "Server is missing OPENROUTER_API_KEY.", "trace": [], "model": None}

    system = SYSTEM_PROMPT.format(snapshot=meta.get("snapshot", "unknown"),
                                  n_listings=meta.get("n_listings", "?"))
    messages: list[dict] = [{"role": "system", "content": system}]
    for m in history[-config.MAX_HISTORY_MESSAGES:]:
        if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str):
            messages.append({"role": m["role"], "content": m["content"][:4000]})
    messages.append({"role": "user", "content": question[:2000]})

    client = _client()
    models = list(config.OPENROUTER_MODELS)
    trace: list[dict] = []
    model_used = None
    t0 = time.time()

    for step in range(config.MAX_AGENT_STEPS):
        # On the last step, force a final answer by removing tools from the request.
        last = step == config.MAX_AGENT_STEPS - 1
        if last:
            messages.append({"role": "user",
                             "content": "(Tool budget exhausted. Answer now from the results above.)"})
        resp, model_used = _complete(client, messages, models, use_tools=not last)
        # stick to the model that worked for the rest of this conversation turn
        models = [model_used] + [m for m in models if m != model_used]
        msg = resp.choices[0].message
        calls = msg.tool_calls or []
        if not calls or step == config.MAX_AGENT_STEPS - 1:
            return {"answer": (msg.content or "").strip() or "Sorry, I could not produce an answer.",
                    "trace": trace, "model": model_used, "seconds": round(time.time() - t0, 1)}

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [{"id": c.id, "type": "function",
                            "function": {"name": c.function.name,
                                         "arguments": c.function.arguments or "{}"}}
                           for c in calls],
        })
        for c in calls:
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = run_tool(c.function.name, args)
            trace.append({"tool": c.function.name, "args": args,
                          "summary": result.get("_summary", "")})
            result.pop("_summary", None)
            messages.append({"role": "tool", "tool_call_id": c.id,
                             "content": json.dumps(result, ensure_ascii=False)[:12000]})

    return {"answer": "Sorry, I could not complete the request.", "trace": trace,
            "model": model_used}
