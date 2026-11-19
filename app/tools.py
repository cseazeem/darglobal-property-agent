"""Tools exposed to the LLM agent (OpenAI function-calling format) and their implementations."""
from __future__ import annotations

import difflib
import json
import statistics
from functools import lru_cache

from . import config
from .retriever import get_retriever


@lru_cache(maxsize=1)
def _projects() -> list[dict]:
    return json.loads((config.DATA_DIR / "projects.json").read_text())


@lru_cache(maxsize=1)
def _listings() -> list[dict]:
    return json.loads((config.DATA_DIR / "listings.json").read_text())


@lru_cache(maxsize=1)
def meta() -> dict:
    return json.loads((config.DATA_DIR / "meta.json").read_text())


_ARTICLES = {"al", "an", "ar", "as", "ad", "at", "az", "ash", "el", "the"}


def _norm(s: str | None) -> str:
    """Lowercase, strip punctuation and Arabic articles so 'Al-Narjis' == 'An Narjis' == 'narjis'."""
    import re
    words = re.findall(r"[a-z0-9؀-ۿ]+", (s or "").lower())
    return " ".join(w for w in words if w not in _ARTICLES)


CITY_ALIASES = {"mecca": "makkah", "medina": "madinah", "khobar": "al khobar",
                "dammam": "dammam", "khamis": "khamis mushait"}


def _city_match(want: str | None, have: str) -> bool:
    if not want:
        return True
    w = want.strip().lower()
    w = CITY_ALIASES.get(w, w)
    return w in have.lower() or have.lower() in w


# --------------------------------------------------------------------------- tools

def search_knowledge(query: str, source: str = "any", k: int = 6) -> dict:
    hits = get_retriever().search(query, k=max(1, min(int(k or 6), 10)),
                                  source=None if source in (None, "", "any") else source)
    return {
        "results": [{"title": h["title"], "url": h["url"], "source": h["source"],
                     "category": h["category"], "text": h["text"][:1500]} for h in hits],
        "_summary": f"{len(hits)} passages",
    }


def list_darglobal_projects(country: str | None = None, status: str | None = None) -> dict:
    out = []
    for p in _projects():
        if country and _norm(country) not in _norm(p.get("country")) and \
                _norm(country) not in _norm(p.get("location")):
            continue
        if status and _norm(status) not in _norm(p.get("status")):
            continue
        out.append({"name": p["name"], "location": p["location"], "country": p["country"],
                    "status": p["status"], "property_type": p["specs"].get("Property Type"),
                    "unit_types": p["specs"].get("Unit Type") or p["specs"].get("Units"),
                    "url": p["url"]})
    return {"count": len(out), "projects": out, "_summary": f"{len(out)} projects"}


def get_darglobal_project(name: str) -> dict:
    projects = _projects()
    q = _norm(name)

    def score(p: dict) -> float:
        n = _norm(p["name"])
        s = 0.6 * difflib.SequenceMatcher(None, q, n).ratio()
        if q and (q in n or n in q):
            s += 0.5
        q_words = set(q.split())
        n_words = set(n.split()) | set(_norm(f"{p.get('location')} {p.get('country')}").split())
        if q_words:  # share of query words found in name/location ("trump dubai" -> Dubai tower)
            s += len(q_words & n_words) / len(q_words)
        return s + (0.05 if p.get("specs") else 0)  # prefer projects with a scraped detail page

    best = max(projects, key=score)
    if score(best) < 0.45:
        return {"error": f"No DarGlobal project matching '{name}'.",
                "available": [p["name"] for p in projects], "_summary": "not found"}
    p = dict(best)
    p["faqs"] = p.get("faqs", [])[:10]
    return {"project": p, "_summary": p["name"]}


def search_wasalt_listings(city: str | None = None, purpose: str | None = None,
                           property_type: str | None = None, district: str | None = None,
                           min_price: float | None = None, max_price: float | None = None,
                           min_bedrooms: int | None = None, max_bedrooms: int | None = None,
                           furnished: bool | None = None, sort: str = "price_asc",
                           limit: int = 8) -> dict:
    rows = []
    for l in _listings():
        if not _city_match(city, l["city"]):
            continue
        if purpose and l["purpose"] != purpose.lower():
            continue
        if property_type and _norm(property_type).rstrip("s") not in _norm(l["type"]):
            continue
        if district and _norm(district) not in _norm(l["district"]) + " " + _norm(l["area"]):
            continue
        if min_price is not None and l["price_sar"] < float(min_price):
            continue
        if max_price is not None and l["price_sar"] > float(max_price):
            continue
        if min_bedrooms is not None and (l["bedrooms"] or 0) < int(min_bedrooms):
            continue
        if max_bedrooms is not None and l["bedrooms"] and l["bedrooms"] > int(max_bedrooms):
            continue
        if furnished is True and not (l["furnishing"] or "").lower().startswith(("furnished", "semi")):
            continue
        rows.append(l)
    key = {"price_asc": lambda r: r["price_sar"], "price_desc": lambda r: -r["price_sar"],
           "newest": lambda r: r["published"] or "", "size_desc": lambda r: -(r["size_sqm"] or 0)}
    rows.sort(key=key.get(sort, key["price_asc"]), reverse=(sort == "newest"))
    limit = max(1, min(int(limit or 8), 15))
    fields = ("title", "type", "purpose", "city", "district", "price_sar", "rent_period",
              "bedrooms", "bathrooms", "size_sqm", "furnishing", "published", "url", "price_note")
    return {
        "total_matches": len(rows),
        "listings": [{f: r.get(f) for f in fields if r.get(f) is not None} for r in rows[:limit]],
        "note": "Sample snapshot of Wasalt listings; prices in SAR. Verify on wasalt.sa.",
        "_summary": f"{len(rows)} matches",
    }


def _trimmed(values: list[float]) -> list[float]:
    """Drop obvious data-entry outliers using the IQR rule."""
    if len(values) < 5:
        return values
    q = statistics.quantiles(values, n=4)
    lo, hi = q[0] - 1.5 * (q[2] - q[0]), q[2] + 1.5 * (q[2] - q[0])
    return [v for v in values if lo <= v <= hi]


def wasalt_market_stats(city: str | None = None, purpose: str = "sale",
                        property_type: str | None = None, group_by: str = "district") -> dict:
    rows = [l for l in _listings() if _city_match(city, l["city"]) and l["purpose"] == purpose
            and (not property_type or _norm(property_type).rstrip("s") in _norm(l["type"]))]
    if not rows:
        return {"error": "No listings for that filter.", "_summary": "no data"}
    groups: dict[str, list[dict]] = {}
    gkey = {"district": "district", "city": "city", "type": "type", "area": "area"}.get(group_by, "district")
    for r in rows:
        groups.setdefault(r[gkey] or "Unknown", []).append(r)

    def describe(rs):
        prices = _trimmed([r["price_sar"] for r in rs])
        ppsqm = _trimmed([r["price_sar"] / r["size_sqm"] for r in rs
                          if r["size_sqm"] and 20 < r["size_sqm"] < 5000])
        return {"listings": len(rs),
                "median_price_sar": round(statistics.median(prices)) if prices else None,
                "min_price_sar": round(min(prices)) if prices else None,
                "max_price_sar": round(max(prices)) if prices else None,
                "median_price_per_sqm": round(statistics.median(ppsqm)) if ppsqm else None}

    breakdown = sorted(((k, describe(v)) for k, v in groups.items()),
                       key=lambda kv: -kv[1]["listings"])[:12]
    return {"filter": {"city": city, "purpose": purpose, "property_type": property_type},
            "overall": describe(rows), "by_" + gkey: dict(breakdown),
            "note": "Computed from a sample of scraped listings (outliers removed). Indicative only.",
            "_summary": f"{len(rows)} listings"}


def wasalt_rental_yield(city: str | None = None, property_type: str = "apartment",
                        group_by: str = "area", min_listings: int = 2) -> dict:
    """Gross rental yield = median yearly rent / median sale price, per area or district.
    Without a city, ranks all cities in one call (saves agent steps)."""
    if not city:
        ranking = {}
        for c in tools_cities():
            r = wasalt_rental_yield(c, property_type, group_by, min_listings)
            if r.get("citywide"):
                ranking[c] = r["citywide"]
        ranking = dict(sorted(ranking.items(), key=lambda kv: -kv[1]["gross_yield_pct"]))
        return {"property_type": property_type, "by_city": ranking,
                "method": "Gross yield = median yearly asking rent / median asking sale price, outliers "
                          "and suspicious rents removed.",
                "disclaimer": "Indicative only, based on a small sample of listings. Not investment advice.",
                "_summary": f"{len(ranking)} cities ranked"}

    def rows(purpose):
        return [l for l in _listings() if _city_match(city, l["city"]) and l["purpose"] == purpose
                and _norm(property_type).rstrip("s") in _norm(l["type"])
                and not l.get("price_note")
                and (purpose == "sale" or (l.get("rent_period") or "yearly") == "yearly")]

    sale, rent = rows("sale"), rows("rent")
    if not sale or not rent:
        return {"error": f"Not enough {property_type} sale and rent listings in {city} to estimate a yield.",
                "_summary": "insufficient data"}

    def yield_of(s, r):
        ps, pr = _trimmed([x["price_sar"] for x in s]), _trimmed([x["price_sar"] for x in r])
        if not ps or not pr:
            return None
        ms, mr = statistics.median(ps), statistics.median(pr)
        return {"median_sale_price_sar": round(ms), "median_yearly_rent_sar": round(mr),
                "gross_yield_pct": round(100 * mr / ms, 2),
                "sale_listings": len(s), "rent_listings": len(r)}

    key = "district" if group_by == "district" else "area"
    groups = {}
    for name in {x[key] for x in sale if x[key]} & {x[key] for x in rent if x[key]}:
        s = [x for x in sale if x[key] == name]
        r = [x for x in rent if x[key] == name]
        if len(s) >= min_listings and len(r) >= min_listings:
            y = yield_of(s, r)
            if y:
                groups[name] = y
    ranked = dict(sorted(groups.items(), key=lambda kv: -kv[1]["gross_yield_pct"]))
    return {
        "city": city, "property_type": property_type,
        "citywide": yield_of(sale, rent), "by_" + key: ranked,
        "method": "Gross yield = median yearly asking rent / median asking sale price, outliers and "
                  "suspicious rents removed. Does not account for bedrooms, size, fees, vacancy or taxes.",
        "disclaimer": "Indicative only, based on a small sample of listings. Not investment advice.",
        "_summary": f"{len(ranked)} {key}s with enough data",
    }


def tools_cities() -> list[str]:
    return sorted({l["city"] for l in _listings()})


# Reference rates (USD per 1 unit). AED/SAR/QAR/OMR/BHD are USD-pegged; GBP as of 18 Nov 2026.
_USD_PER = {"USD": 1.0, "AED": 1 / 3.6725, "SAR": 1 / 3.75, "QAR": 1 / 3.64, "OMR": 1 / 0.3845,
            "BHD": 1 / 0.376, "GBP": 4.85 / 3.6725}


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    f, t = from_currency.upper().strip(), to_currency.upper().strip()
    if f not in _USD_PER or t not in _USD_PER:
        return {"error": f"Supported currencies: {', '.join(_USD_PER)}", "_summary": "unsupported"}
    value = float(amount) * _USD_PER[f] / _USD_PER[t]
    return {"amount": amount, "from": f, "to": t, "converted": round(value, 2),
            "rate": round(_USD_PER[f] / _USD_PER[t], 4),
            "note": "Gulf currencies are pegged to USD; GBP rate is indicative (18 Nov 2026).",
            "_summary": f"{amount:,.0f} {f} ≈ {value:,.0f} {t}"}


def draft_interest_registration(project: str, name: str, email: str | None = None,
                                phone: str | None = None, budget: str | None = None,
                                preferred_contact: str | None = None,
                                notes: str | None = None) -> dict:
    """Human-in-the-loop: build a DRAFT only. Nothing is stored until the user confirms in the UI."""
    import re
    if not (email or phone):
        return {"error": "Ask the user for an email or phone number first.", "_summary": "missing contact"}
    if email and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email.strip()):
        return {"error": "That email address looks invalid; ask the user to re-check it.",
                "_summary": "invalid email"}
    match = get_darglobal_project(project)
    proj = match.get("project", {})
    draft = {"project": proj.get("name") or project, "project_url": proj.get("url"),
             "name": name.strip(), "email": (email or "").strip() or None,
             "phone": (phone or "").strip() or None, "budget": budget,
             "preferred_contact": preferred_contact, "notes": notes}
    return {"status": "awaiting_user_confirmation", "draft": draft,
            "instruction": "Tell the user to review the confirmation card shown below this message and press Confirm. This is a demo: confirmed requests are stored by this app, NOT sent to DarGlobal. Never say it will be submitted or sent to DarGlobal.",
            "_summary": f"draft for {draft['project']}"}


# --------------------------------------------------------------------------- schemas

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "search_knowledge",
        "description": "Semantic + keyword search over scraped DarGlobal pages (projects, FAQs, "
                       "company/investor info, press releases, insights) and Wasalt city guides "
                       "and listings. Use for open questions.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Search query in English"},
            "source": {"type": "string", "enum": ["any", "darglobal", "wasalt"]},
            "k": {"type": "integer", "description": "Number of passages (1-10)"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "list_darglobal_projects",
        "description": "List DarGlobal developments, optionally filtered by country "
                       "(UAE, Saudi Arabia, Oman, Qatar, UK, Spain, Maldives) or status.",
        "parameters": {"type": "object", "properties": {
            "country": {"type": "string"},
            "status": {"type": "string", "description": "e.g. 'Under Development', 'Sold Out', 'Ready'"}}}}},
    {"type": "function", "function": {
        "name": "get_darglobal_project",
        "description": "Full details of one DarGlobal project: location, specs, unit types, sizes, "
                       "status, amenities, key features and FAQs (handover dates, payment info).",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "Project name, e.g. 'Trump Tower Jeddah'"}},
            "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "search_wasalt_listings",
        "description": "Filter Wasalt property listings in Saudi Arabia (Riyadh, Jeddah, Makkah, "
                       "Madinah, Dammam, Al Khobar, Khamis Mushait). Prices in SAR; rents yearly.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"},
            "purpose": {"type": "string", "enum": ["sale", "rent"]},
            "property_type": {"type": "string",
                              "description": "Apartment, Villa, Land, Building, Floor, Rest House"},
            "district": {"type": "string", "description": "Neighbourhood, e.g. 'Al-Narjis', or 'North Riyadh'"},
            "min_price": {"type": "number"}, "max_price": {"type": "number"},
            "min_bedrooms": {"type": "integer"}, "max_bedrooms": {"type": "integer"},
            "furnished": {"type": "boolean"},
            "sort": {"type": "string", "enum": ["price_asc", "price_desc", "newest", "size_desc"]},
            "limit": {"type": "integer"}}}}},
    {"type": "function", "function": {
        "name": "wasalt_market_stats",
        "description": "Price statistics (median, range, price per sqm) from Wasalt listings, "
                       "grouped by district, area, type or city. Use for comparisons and budgets.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"},
            "purpose": {"type": "string", "enum": ["sale", "rent"]},
            "property_type": {"type": "string"},
            "group_by": {"type": "string", "enum": ["district", "area", "type", "city"]}},
            "required": ["purpose"]}}},
]

TOOL_SCHEMAS += [
    {"type": "function", "function": {
        "name": "wasalt_rental_yield",
        "description": "Estimate gross rental yield (median yearly rent / median sale price) from "
                       "Wasalt listings. Omit `city` to rank ALL Saudi cities in one call; pass a city "
                       "for an area/district breakdown. Use for investment-return questions.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "Optional. Omit to compare all cities."},
            "property_type": {"type": "string", "description": "Apartment (default), Villa, Floor"},
            "group_by": {"type": "string", "enum": ["area", "district"]}}}}},
    {"type": "function", "function": {
        "name": "convert_currency",
        "description": "Convert an amount between AED, SAR, QAR, OMR, BHD, USD and GBP.",
        "parameters": {"type": "object", "properties": {
            "amount": {"type": "number"}, "from_currency": {"type": "string"},
            "to_currency": {"type": "string"}},
            "required": ["amount", "from_currency", "to_currency"]}}},
    {"type": "function", "function": {
        "name": "draft_interest_registration",
        "description": "Create a DRAFT 'register interest' request for a DarGlobal project once you "
                       "have the project, the person's name and an email or phone. The user must "
                       "confirm it in the UI; this tool does not submit anything.",
        "parameters": {"type": "object", "properties": {
            "project": {"type": "string"}, "name": {"type": "string"},
            "email": {"type": "string"}, "phone": {"type": "string"},
            "budget": {"type": "string"}, "preferred_contact": {"type": "string"},
            "notes": {"type": "string"}},
            "required": ["project", "name"]}}},
]

_IMPL = {
    "wasalt_rental_yield": wasalt_rental_yield,
    "convert_currency": convert_currency,
    "draft_interest_registration": draft_interest_registration,
    "search_knowledge": search_knowledge,
    "list_darglobal_projects": list_darglobal_projects,
    "get_darglobal_project": get_darglobal_project,
    "search_wasalt_listings": search_wasalt_listings,
    "wasalt_market_stats": wasalt_market_stats,
}


def run_tool(name: str, args: dict) -> dict:
    fn = _IMPL.get(name)
    if not fn:
        return {"error": f"Unknown tool {name}", "_summary": "unknown tool"}
    try:
        return fn(**{k: v for k, v in (args or {}).items() if v not in (None, "")})
    except TypeError as e:
        return {"error": f"Bad arguments: {e}", "_summary": "bad args"}
    except Exception as e:  # never crash the agent loop
        return {"error": str(e), "_summary": "error"}
