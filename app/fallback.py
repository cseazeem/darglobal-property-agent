"""Degraded mode: answer without the LLM when every model is unavailable (e.g. daily quota).

All tools and retrieval run locally, so we can still show grounded results, formatted
deterministically, instead of an error message.
"""
from __future__ import annotations

from .tools import run_tool


def _money(v) -> str:
    return f"{v:,.0f}" if isinstance(v, (int, float)) else "–"


def format_result(tool: str, r: dict) -> str:
    if "error" in r:
        return ""
    if tool == "list_darglobal_projects":
        rows = [f"| [{p['name']}]({p['url']}) | {p.get('location') or '–'} | {p.get('status') or '–'} |"
                for p in r["projects"][:20]]
        return "| Project | Location | Status |\n|---|---|---|\n" + "\n".join(rows)
    if tool == "get_darglobal_project":
        p = r["project"]
        specs = "\n".join(f"- **{k}:** {v}" for k, v in p.get("specs", {}).items())
        return f"**[{p['name']}]({p['url']})** · {p.get('location') or ''}\n\n{p.get('description', '')}\n\n{specs}"
    if tool == "search_wasalt_listings":
        rows = [f"| [{l['title']}]({l['url']}) | {l.get('district') or '–'}, {l['city']} | "
                f"{_money(l['price_sar'])}{' / yr' if l['purpose'] == 'rent' else ''} | {l.get('bedrooms') or '–'} |"
                for l in r["listings"]]
        return (f"{r['total_matches']} matching listings (SAR):\n\n| Listing | Location | Price | Beds |\n"
                "|---|---|---|---|\n" + "\n".join(rows))
    if tool == "wasalt_market_stats":
        o = r["overall"]
        return (f"Median price **SAR {_money(o['median_price_sar'])}** (range {_money(o['min_price_sar'])}–"
                f"{_money(o['max_price_sar'])}) across {o['listings']} listings.")
    if tool == "wasalt_rental_yield":
        data = r.get("by_city") or r.get("by_area") or r.get("by_district") or {}
        rows = [f"| {k} | {v['gross_yield_pct']}% | {_money(v['median_sale_price_sar'])} | "
                f"{_money(v['median_yearly_rent_sar'])} |" for k, v in data.items()]
        if not rows and r.get("citywide"):
            c = r["citywide"]
            rows = [f"| {r.get('city')} | {c['gross_yield_pct']}% | {_money(c['median_sale_price_sar'])} | "
                    f"{_money(c['median_yearly_rent_sar'])} |"]
        return ("| | Gross yield | Median price (SAR) | Median yearly rent (SAR) |\n|---|---|---|---|\n"
                + "\n".join(rows) + f"\n\n_{r.get('disclaimer', '')}_")
    if tool == "convert_currency":
        return f"{_money(r['amount'])} {r['from']} ≈ **{_money(r['converted'])} {r['to']}** (rate {r['rate']})."
    if tool == "search_knowledge":
        parts = []
        for h in r["results"][:3]:
            snippet = " ".join(h["text"].split())[:350]
            parts.append(f"**[{h['title']}]({h['url']})**\n> {snippet}…")
        return "\n\n".join(parts)
    return ""


_COUNTRIES = {"uae": "UAE", "dubai": "UAE", "emirates": "UAE", "saudi": "Saudi Arabia", "ksa": "Saudi Arabia",
              "jeddah": "Saudi Arabia", "riyadh": "Saudi Arabia", "oman": "Oman", "muscat": "Oman",
              "qatar": "Qatar", "doha": "Qatar", "london": "UK", "uk": "UK", "spain": "Spain",
              "marbella": "Spain", "maldives": "Maldives",
              "الإمارات": "UAE", "دبي": "UAE", "السعودية": "Saudi Arabia", "عمان": "Oman", "عُمان": "Oman",
              "قطر": "Qatar", "لندن": "UK", "إسبانيا": "Spain"}
_CITIES = ["riyadh", "jeddah", "makkah", "mecca", "madinah", "medina", "dammam", "khobar", "khamis"]


def _route(question: str) -> list[tuple[str, dict]]:
    """Keyword routing used only when no LLM is available."""
    q = question.lower()
    q = q.replace("?", " ").replace("؟", " ").replace(",", " ").replace("،", " ")
    words = set(q.split())
    if "yield" in q or "return" in q or "عائد" in q:
        return [("wasalt_rental_yield", {})]
    city = next((c for c in _CITIES if c in q), None)
    if city and any(w in q for w in ("rent", "sale", "buy", "apartment", "villa", "listing", "إيجار", "للبيع")):
        purpose = "rent" if ("rent" in q or "إيجار" in q) else "sale"
        ptype = "villa" if "villa" in q else ("apartment" if "apartment" in q or "flat" in q else None)
        return [("search_wasalt_listings", {"city": city, "purpose": purpose, "property_type": ptype})]
    proj = run_tool("get_darglobal_project", {"name": question})
    if "project" in proj and len(question.split()) <= 12 and not any(w in words for w in ("projects", "مشاريع")):
        return [("get_darglobal_project", {"name": question})]
    country = next((v for k, v in _COUNTRIES.items() if k in words or k in q.split()), None)
    if country and any(w in q for w in ("project", "development", "مشاريع", "مشروع")):
        return [("list_darglobal_projects", {"country": country})]
    return [("search_knowledge", {"query": question, "k": 3})]


def fallback_answer(question: str, tool_results: list[tuple[str, dict]], quota: bool) -> str:
    reason = ("The AI model's free daily quota has been reached (it resets at 00:00 UTC)"
              if quota else "The AI model is temporarily unavailable")
    head = f"⚠️ {reason}, so here are the raw results from the data tools instead of a written answer.\n\n"
    body = [f for f in (format_result(t, r) for t, r in tool_results) if f]
    if not body:  # nothing ran yet: route by keywords to a tool, all of which run without the LLM
        for tool, args in _route(question):
            body.append(format_result(tool, run_tool(tool, {k: v for k, v in args.items() if v})))
        if not any(body):
            body = [format_result("search_knowledge", run_tool("search_knowledge", {"query": question, "k": 3}))]
    return head + "\n\n---\n\n".join(b for b in body if b)
