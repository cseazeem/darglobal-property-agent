"""Turn raw scraped JSON (data/raw/*.json) into clean, app-ready datasets.

Outputs (data/processed/):
  projects.json  – one record per DarGlobal development (structured)
  listings.json  – normalised Wasalt listings (structured, for filtering/stats)
  docs.jsonl     – text documents for RAG (DarGlobal pages, Wasalt city guides, listings)
  meta.json      – snapshot info shown in the UI / system prompt

Usage:  python scraper/build_kb.py
"""
from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

CITY_NORMALISE = {
    "aldammam": "Dammam", "dammam": "Dammam",
    "makkah al mukarramah": "Makkah", "makkah": "Makkah", "mecca": "Makkah",
    "khobar": "Al Khobar", "al khobar": "Al Khobar",
    "khamis mushait": "Khamis Mushait", "khamis mushayt": "Khamis Mushait",
    "riyadh": "Riyadh", "jeddah": "Jeddah", "madinah": "Madinah",
}

COUNTRY_HINTS = [
    ("Maldives", ["maldives"]),
    ("UAE", ["dubai", "uae", "united arab emirates", "rak", "ras al khaimah", "marjan"]),
    ("Saudi Arabia", ["ksa", "saudi", "riyadh", "jeddah", "jeddha", "wadi safar", "diriyah", "rayana", "amaya"]),
    ("Oman", ["oman", "muscat", "aida"]),
    ("Qatar", ["qatar", "doha", "simaisma", "lumaia"]),
    ("UK", ["london", "united kingdom", " uk"]),
    ("Spain", ["spain", "marbella", "benah", "costa del sol", "cortesin", "estepona"]),
]


def clean(s: str | None) -> str:
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"[ \t ]+", " ", re.sub(r"\n\s*\n+", "\n", s)).strip()


def rich_text(nodes) -> str:
    """Flatten Strapi rich-text blocks ([{type, children:[{text}]}]) to plain text."""
    if isinstance(nodes, str):
        return clean(nodes)
    out = []
    for n in nodes or []:
        if isinstance(n, dict):
            if "text" in n and isinstance(n["text"], str):
                out.append(n["text"])
            out.append(rich_text(n.get("children")))
    return clean(" ".join(x for x in out if x))


def guess_country(*texts: str) -> str | None:
    blob = " " + " ".join(t.lower() for t in texts if t) + " "
    for country, keys in COUNTRY_HINTS:
        if any(k in blob for k in keys):
            return country
    return None


def collect_titles(obj, keys=("title", "name", "Title", "Name")) -> list[str]:
    """Collect short labels (e.g. amenity names) from a nested CMS block."""
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys and isinstance(v, str) and 2 < len(v) < 80 and not v.isupper():
                found.append(v.strip())
            elif isinstance(v, (dict, list)) and k not in ("data", "formats", "image", "icon"):
                found += collect_titles(v, keys)
    elif isinstance(obj, list):
        for v in obj:
            found += collect_titles(v, keys)
    return found


def dg_category(url: str) -> str:
    path = url.replace("https://darglobal.co.uk", "").strip("/")
    first = path.split("/")[0] if path else "home"
    return {
        "press": "press-release", "insights": "insight", "partners": "partner",
        "projects": "country-listing", "one-of-one": "one-of-one",
    }.get(first, "page")


def build_darglobal():
    raw = json.loads((RAW / "darglobal_raw.json").read_text())
    projects, docs, seen_slugs = [], [], set()
    list_country = {p["name"].lower(): p.get("country") for p in raw.get("projectList") or []}

    for page in raw["pages"]:
        url, text = page["url"], clean(page.get("text"))
        title = clean(page.get("title")).replace(" | DarGlobal", "").strip() or url
        p = page.get("project")
        # the home page embeds a featured project; only treat a page as the project's own page
        if p and not url.rstrip("/").lower().endswith("/" + str(p.get("slug")).lower()):
            p = None
        if p and p.get("slug") not in seen_slugs:
            seen_slugs.add(p["slug"])
            det = p.get("details") or {}
            banner = det.get("ProjectBanner") or {}
            about = det.get("AboutProject") or {}
            specs = {clean(s.get("name")).title(): clean(s.get("value"))
                     for s in about.get("AboutDetailsType") or [] if s.get("value")}
            faqs = []
            for block in p.get("faq") or []:
                for f in block.get("faqs") or []:
                    faqs.append({"q": clean(f.get("question")), "a": rich_text(f.get("answer"))})
            amenities = list(dict.fromkeys(collect_titles(det.get("Amenities"))))[:30]
            features = list(dict.fromkeys(collect_titles(det.get("KeyFeatures"))))[:15]
            location = clean(banner.get("Location"))
            country = guess_country(location, p.get("title", "")) or \
                guess_country(list_country.get(p.get("title", "").lower()) or "")
            rec = {
                "name": clean(p.get("title")),
                "slug": p["slug"],
                "url": url,
                "location": location,
                "country": country,
                "starting_price": clean(str(banner.get("StartingPrice") or "")) or None,
                "description": clean(about.get("description")),
                "specs": specs,
                "status": specs.get("Status"),
                "amenities": amenities,
                "key_features": features,
                "faqs": faqs,
            }
            projects.append(rec)
            # enrich the page text with structured info + all FAQs (some are collapsed in HTML)
            extra = [f"Project: {rec['name']}", f"Location: {location} ({country})"]
            extra += [f"{k}: {v}" for k, v in specs.items()]
            if amenities:
                extra.append("Amenities: " + ", ".join(amenities))
            extra += [f"Q: {f['q']}\nA: {f['a']}" for f in faqs]
            text = "\n".join(extra) + "\n" + text
        if len(text) < 120:
            continue  # client-rendered shells with no content
        docs.append({"source": "darglobal", "category": "project" if p else dg_category(url),
                     "url": url, "title": title, "text": text})

    # projects that appear in DarGlobal's project list but have no dedicated page we scraped
    known = {p["name"].lower() for p in projects}
    for item in raw.get("projectList") or []:
        name = clean(item.get("name"))
        if name.lower() in known or any(name.lower()[:12] in k for k in known):
            continue
        projects.append({"name": name, "slug": None, "url": "https://darglobal.co.uk",
                         "location": clean(item.get("country")),
                         "country": guess_country(item.get("country") or "", name),
                         "starting_price": None, "description": "", "specs": {}, "status": None,
                         "amenities": [], "key_features": [], "faqs": [],
                         "note": "Listed in DarGlobal's project directory; no detail page scraped."})
    return projects, docs, raw.get("scrapedAt")


def _num(v) -> float | None:
    if v is None:
        return None
    m = re.search(r"[\d.]+", str(v).replace(",", ""))
    return float(m.group()) if m else None


def build_wasalt():
    raw = json.loads((RAW / "wasalt_raw.json").read_text())
    listings, docs = [], []
    for x in raw["listings"]:
        if not x.get("title") or not x.get("city") or x.get("price") in (None, 0):
            continue
        city = CITY_NORMALISE.get(x["city"].lower(), x["city"])
        at = x.get("attributes") or {}
        purpose = x.get("propertyFor")
        rent_period = None
        if purpose == "rent" and isinstance(x.get("rentFreq"), dict):
            default = next((k for k, v in x["rentFreq"].items() if v.get("default_freq")), None)
            rent_period = default or "yearly"
        rec = {
            "id": x["id"], "url": x["url"], "title": x["title"], "purpose": purpose,
            "category": x.get("mainType"), "type": x.get("subType"), "city": city,
            "area": x.get("territory"), "district": x.get("district"),
            "price_sar": x["price"], "rent_period": rent_period,
            "bedrooms": int(_num(at.get("noOfBedrooms")) or 0) or None,
            "bathrooms": int(_num(at.get("noOfBathrooms")) or 0) or None,
            "size_sqm": _num(at.get("builtUpArea") or at.get("carpetArea")),
            "age": at.get("completionYear"), "furnishing": x.get("furnishing"),
            "listed_by": x.get("companyRole"), "published": (x.get("publishedAt") or "")[:10] or None,
            # Some advertisers enter a monthly rent while the portal labels it yearly.
            "price_note": ("Very low for a yearly rent; may actually be monthly. Verify on Wasalt."
                           if purpose == "rent" and rent_period == "yearly" and x["price"] < 10000
                           else None),
            "lat": _num(x.get("lat")), "lon": _num(x.get("lon")),
        }
        listings.append(rec)
        price = f"SAR {rec['price_sar']:,}" + (f" per {rent_period.replace('ly', '')}" if rent_period else "")
        bits = [rec["title"], f"{rec['type']} for {purpose} in {rec['district']}, {city}", price]
        if rec["bedrooms"]:
            bits.append(f"{rec['bedrooms']} bedrooms")
        if rec["size_sqm"]:
            bits.append(f"{rec['size_sqm']:g} sqm")
        if rec["furnishing"]:
            bits.append(rec["furnishing"])
        docs.append({"source": "wasalt", "category": "listing", "url": rec["url"],
                     "title": f"Wasalt listing: {rec['title']} ({city})", "text": " · ".join(bits)})

    for c in raw["cityPages"]:
        slug = c["url"].rsplit("/", 1)[-1]
        title = slug.replace("-", " ").title().replace("Al ", "Al-")
        text = clean(c.get("about"))
        faq = "\n".join(f"Q: {q['q'].strip()}\nA: {q['a'].strip()}" for q in c.get("faq") or [])
        if c.get("total"):
            text = f"Wasalt shows {c['total']:,} active listings for '{title}'.\n" + text
        if len(text) + len(faq) < 120:
            continue
        docs.append({"source": "wasalt", "category": "city-guide", "url": c["url"],
                     "title": f"Wasalt guide: {title}", "text": text + "\n" + faq})
    return listings, docs, raw.get("scrapedAt")


def chunk(doc: dict, size: int = 1400, overlap: int = 200) -> list[dict]:
    text = doc["text"]
    if len(text) <= size:
        return [doc]
    parts, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):  # prefer to cut at a line break
            cut = text.rfind("\n", start + size // 2, end)
            end = cut if cut > 0 else end
        parts.append({**doc, "text": text[start:end].strip()})
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return parts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    projects, dg_docs, dg_time = build_darglobal()
    listings, ws_docs, ws_time = build_wasalt()

    docs = []
    for d in dg_docs + ws_docs:
        docs += chunk(d)
    for i, d in enumerate(docs):
        d["id"] = i

    (OUT / "projects.json").write_text(json.dumps(projects, ensure_ascii=False, indent=1))
    (OUT / "listings.json").write_text(json.dumps(listings, ensure_ascii=False))
    (OUT / "docs.jsonl").write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in docs))
    meta = {
        "snapshot": (dg_time or ws_time or "")[:10],
        "n_projects": len(projects), "n_listings": len(listings), "n_chunks": len(docs),
        "darglobal_pages": len(dg_docs),
        "wasalt_cities": sorted({l["city"] for l in listings}),
        "listing_counts": Counter(f"{l['city']}|{l['purpose']}" for l in listings),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "listing_counts"}, indent=1))
    prices = [l["price_sar"] for l in listings if l["purpose"] == "sale"]
    print("sale price median:", statistics.median(prices) if prices else None)


if __name__ == "__main__":
    main()
