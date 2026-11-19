"""End-to-end evaluation of the agent against a golden set (real LLM calls).

Checks per case:
  tool     - the agent called at least one of the expected tools (or none, for refusal cases)
  facts    - every `must` string and at least one `must_any` string appear in the answer
  safe     - no `must_not` string appears (hallucinated submission, leaked prompt, off-topic code)
  cited    - the answer contains a source link, when `cite` is true
  arabic   - the answer is written in Arabic, when `arabic` is true

Usage:
  python evals/run_eval.py                      # all cases, in-process
  python evals/run_eval.py --limit 6            # quick subset (free models have daily quotas)
  python evals/run_eval.py --base-url https://darglobal-property-agent.onrender.com
Writes evals/report.md and evals/results.json.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def ask_local(q: str) -> dict:
    from app.agent import run_agent
    from app.tools import meta
    return run_agent(q, [], meta())


def ask_remote(base: str, q: str) -> dict:
    import httpx
    r = httpx.post(f"{base.rstrip('/')}/api/chat", json={"message": q}, timeout=180)
    r.raise_for_status()
    return r.json()


def grade(case: dict, res: dict) -> dict:
    ans = res.get("answer", "")
    low = ans.lower()
    used = [t["tool"] for t in res.get("trace", [])]
    exp = case.get("tools", [])
    checks = {
        "tool": (not used) if exp == [] else any(t in used for t in exp),
        "facts": all(m.lower() in low for m in case.get("must", []))
                 and (not case.get("must_any") or any(m.lower() in low for m in case["must_any"])),
        "safe": not any(m.lower() in low for m in case.get("must_not", [])),
    }
    if case.get("cite"):
        checks["cited"] = bool(re.search(r"https?://", ans))
    if case.get("arabic"):
        checks["arabic"] = len(re.findall(r"[؀-ۿ]", ans)) > 20
    return {"id": case["id"], "q": case["q"], "tools_used": used, "checks": checks,
            "passed": all(checks.values()), "model": res.get("model"),
            "seconds": res.get("seconds"), "answer": ans}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", nargs="*", help="case ids to run")
    ap.add_argument("--base-url", default="")
    ap.add_argument("--pause", type=float, default=2.0, help="seconds between cases (rate limits)")
    args = ap.parse_args()

    cases = [json.loads(l) for l in (HERE / "golden.jsonl").read_text().splitlines() if l.strip()]
    if args.only:
        cases = [c for c in cases if c["id"] in args.only]
    if args.limit:
        cases = cases[:args.limit]

    results = []
    for i, c in enumerate(cases, 1):
        t0 = time.time()
        try:
            res = ask_remote(args.base_url, c["q"]) if args.base_url else ask_local(c["q"])
            r = grade(c, res)
        except Exception as e:
            r = {"id": c["id"], "q": c["q"], "tools_used": [], "checks": {"error": False},
                 "passed": False, "error": str(e)[:300], "seconds": round(time.time() - t0, 1)}
        results.append(r)
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"[{i}/{len(cases)}] {mark} {c['id']:<22} tools={r['tools_used']} "
              f"checks={r['checks']} {r.get('seconds')}s", flush=True)
        time.sleep(args.pause)

    n = len(results)
    passed = sum(r["passed"] for r in results)
    secs = [r["seconds"] for r in results if r.get("seconds")]
    by_check: dict[str, list[bool]] = {}
    for r in results:
        for k, v in r["checks"].items():
            by_check.setdefault(k, []).append(v)

    lines = ["# Evaluation report", "",
             f"Run: {time.strftime('%Y-%m-%d %H:%M')} · target: {args.base_url or 'local'} · cases: {n}", "",
             f"**Pass rate: {passed}/{n} ({100 * passed / max(n, 1):.0f}%)**", "",
             "| Check | Pass rate |", "|---|---|"]
    lines += [f"| {k} | {sum(v)}/{len(v)} |" for k, v in by_check.items()]
    if secs:
        lines += ["", f"Latency: median {statistics.median(secs):.1f}s, max {max(secs):.1f}s", ""]
    lines += ["| Case | Result | Tools used | Failed checks |", "|---|---|---|---|"]
    for r in results:
        failed = ", ".join(k for k, v in r["checks"].items() if not v) or "–"
        lines.append(f"| {r['id']} | {'✅' if r['passed'] else '❌'} | {', '.join(r['tools_used']) or '–'} | {failed} |")
    (HERE / "report.md").write_text("\n".join(lines) + "\n")
    (HERE / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(f"\nPass rate {passed}/{n}. Report: evals/report.md")


if __name__ == "__main__":
    main()
