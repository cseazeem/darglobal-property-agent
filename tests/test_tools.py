"""Deterministic tests for the agent's tools and data pipeline (no LLM calls, run in CI).

    pytest -q
"""
import json

import pytest

from app import tools
from app.agent import _parse_text_tool_calls
from app.retriever import get_retriever


# ---------------------------------------------------------------- data quality

def test_dataset_sizes():
    m = tools.meta()
    assert m["n_projects"] >= 40
    assert m["n_listings"] >= 600
    assert set(m["wasalt_cities"]) >= {"Riyadh", "Jeddah", "Makkah", "Madinah", "Dammam", "Al Khobar"}


def test_listings_have_no_personal_data():
    raw = json.dumps(tools._listings())
    for field in ("resEmployee", "ownerName", "phone", "falLicenseNumber", "mobile"):
        assert field not in raw


def test_every_project_has_country():
    missing = [p["name"] for p in tools._projects() if p.get("specs") and not p.get("country")]
    assert not missing, missing


# ---------------------------------------------------------------- project lookup

@pytest.mark.parametrize("query,expected", [
    ("Trump International Hotel & Tower Dubai", "Trump International Hotel & Tower Dubai"),
    ("Trump Dubai", "Trump International Hotel & Tower Dubai"),
    ("trump tower jeddah", "Trump Tower Jeddah"),
    ("astera", "The Astera, Interiors by Aston Martin"),
    ("pagani", "Da Vinci Tower interiors by Pagani"),
    ("Trump hotel oman", "Trump International Hotel"),
    ("les vagues doha", "Les Vagues"),
])
def test_project_matching(query, expected):
    assert tools.get_darglobal_project(query)["project"]["name"] == expected


def test_project_not_found():
    assert "error" in tools.get_darglobal_project("xyz random thing")


def test_list_projects_by_country():
    uae = {p["name"] for p in tools.list_darglobal_projects(country="UAE")["projects"]}
    assert "The Astera, Interiors by Aston Martin" in uae
    assert all(p["country"] == "Oman" for p in tools.list_darglobal_projects(country="Oman")["projects"])


def test_trump_dubai_handover_fact():
    faqs = " ".join(f["a"] for f in tools.get_darglobal_project("Trump Dubai")["project"]["faqs"])
    assert "December 2031" in faqs


# ---------------------------------------------------------------- listings & stats

def test_listing_filters_respected():
    r = tools.search_wasalt_listings(city="Riyadh", purpose="rent", property_type="apartment",
                                     max_price=70000, min_bedrooms=3, limit=15)
    assert r["total_matches"] > 0
    for l in r["listings"]:
        assert l["city"] == "Riyadh" and l["purpose"] == "rent" and l["type"] == "Apartment"
        assert l["price_sar"] <= 70000 and l["bedrooms"] >= 3


def test_arabic_article_normalisation():
    a = tools.search_wasalt_listings(city="Riyadh", district="Al Narjis")["total_matches"]
    b = tools.search_wasalt_listings(city="Riyadh", district="An-Narjis")["total_matches"]
    assert a == b > 0


def test_city_aliases():
    assert tools.search_wasalt_listings(city="Mecca")["total_matches"] == \
        tools.search_wasalt_listings(city="Makkah")["total_matches"] > 0


def test_market_stats_remove_outliers():
    s = tools.wasalt_market_stats(city="Riyadh", purpose="sale", property_type="land")
    assert s["overall"]["max_price_sar"] < 1_000_000_000  # the 13.8bn SAR typo is trimmed


def test_rental_yield_is_plausible():
    r = tools.wasalt_rental_yield("Riyadh")
    assert 2 < r["citywide"]["gross_yield_pct"] < 15
    assert "Not investment advice" in r["disclaimer"]


def test_low_rent_flagged():
    flagged = [l for l in tools._listings() if l.get("price_note")]
    assert flagged and all(l["purpose"] == "rent" for l in flagged)


# ---------------------------------------------------------------- currency & leads

def test_currency_pegs():
    assert tools.convert_currency(3.75, "SAR", "USD")["converted"] == 1.0
    assert tools.convert_currency(3.6725, "AED", "USD")["converted"] == 1.0
    assert "error" in tools.convert_currency(1, "XYZ", "AED")


def test_lead_draft_requires_contact_and_valid_email():
    assert "error" in tools.draft_interest_registration("Trump Tower Jeddah", "A")
    assert "error" in tools.draft_interest_registration("Trump Tower Jeddah", "A", email="not-an-email")
    d = tools.draft_interest_registration("trump tower jeddah", "A", phone="+966500000000")
    assert d["status"] == "awaiting_user_confirmation"
    assert d["draft"]["project"] == "Trump Tower Jeddah"


def test_run_tool_never_raises():
    assert "error" in tools.run_tool("does_not_exist", {})
    assert "error" in tools.run_tool("search_wasalt_listings", {"bogus_arg": 1})


# ---------------------------------------------------------------- retrieval & agent parsing

def test_retriever_finds_partnership_press_release():
    hits = get_retriever().search("DarGlobal Wasalt partnership Trump Tower Jeddah", k=5)
    assert any("wasalt" in h["url"].lower() for h in hits)


def test_retriever_source_filter():
    assert all(h["source"] == "wasalt" for h in get_retriever().search("villa Riyadh", k=5, source="wasalt"))


def test_text_tool_call_parsing():
    calls = _parse_text_tool_calls(
        "<tool_call>\n<function=convert_currency>\n<parameter=amount>\n100\n</parameter>\n"
        "<parameter=from_currency>\nSAR\n</parameter>\n</function>\n</tool_call>")
    assert calls[0]["name"] == "convert_currency"
    assert json.loads(calls[0]["args"]) == {"amount": 100, "from_currency": "SAR"}


def test_rental_yield_all_cities_in_one_call():
    r = tools.wasalt_rental_yield()
    assert len(r["by_city"]) >= 5
    ylds = [v["gross_yield_pct"] for v in r["by_city"].values()]
    assert ylds == sorted(ylds, reverse=True)
