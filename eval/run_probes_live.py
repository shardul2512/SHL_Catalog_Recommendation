"""
Live probe runner.

Runs every probe in adversarial_probes.py against a real /chat endpoint
and reports a pass-rate breakdown by category.

Usage:
  python3 eval/run_probes_live.py --base-url https://shl-catalog-recommendation.onrender.com -v
  python3 eval/run_probes_live.py --base-url http://localhost:8000 -v
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import httpx

# Allow running from project root or from eval/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.adversarial_probes import PROBES, Probe


def _call_chat(base_url: str, messages: list, timeout: float = 60.0) -> Optional[dict]:
    url = base_url.rstrip("/") + "/chat"
    try:
        r = httpx.post(url, json={"messages": messages}, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        print(f"    [ERROR] HTTP call failed: {exc}", file=sys.stderr)
        return None


def run_probes(base_url: str, verbose: bool = False) -> dict:
    results = []
    by_category: dict[str, list[bool]] = defaultdict(list)

    print(f"\nRunning {len(PROBES)} adversarial probes against {base_url}\n")
    print(f"{'Probe':<45} {'Category':<14} {'Result'}")
    print("─" * 75)

    for probe in PROBES:
        t0 = time.time()
        resp = _call_chat(base_url, probe.messages)
        elapsed = time.time() - t0

        if resp is None:
            passed = False
            status = "ERROR"
        else:
            try:
                passed = bool(probe.check(resp))
                status = "PASS" if passed else "FAIL"
            except Exception as exc:
                passed = False
                status = f"CHECK-ERROR: {exc}"

        icon = "✅" if passed else "❌"
        print(f"  {icon}  {probe.name:<43} {probe.category:<14} ({elapsed:.1f}s)")

        if verbose and not passed:
            print(f"       DESC: {probe.description}")
            if resp:
                print(f"       REPLY: {resp.get('reply', '')[:200]}")
                print(f"       RECS:  {[r['name'] for r in resp.get('recommendations', [])]}")
            print()

        results.append({"name": probe.name, "category": probe.category, "passed": passed})
        by_category[probe.category].append(passed)

    # ── Summary ──────────────────────────────────────────────────────────────
    total = len(results)
    passed_total = sum(r["passed"] for r in results)
    overall_rate = passed_total / total if total else 0

    print("\n" + "═" * 75)
    print(f"  Overall pass-rate: {passed_total}/{total}  ({overall_rate:.1%})")
    print()
    print(f"  {'Category':<16}  {'Pass':<6}  {'Total':<6}  {'Rate'}")
    print("  " + "─" * 40)
    for cat, outcomes in sorted(by_category.items()):
        cat_pass = sum(outcomes)
        cat_total = len(outcomes)
        cat_rate = cat_pass / cat_total if cat_total else 0
        icon = "✅" if cat_rate == 1.0 else ("⚠️ " if cat_rate >= 0.5 else "❌")
        print(f"  {icon}  {cat:<14}  {cat_pass:<6}  {cat_total:<6}  {cat_rate:.1%}")

    print("═" * 75)
    print()

    return {
        "base_url": base_url,
        "total": total,
        "passed": passed_total,
        "overall_rate": overall_rate,
        "by_category": {
            cat: {"passed": sum(v), "total": len(v), "rate": sum(v) / len(v)}
            for cat, v in by_category.items()
        },
        "probes": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Run adversarial probes against a live /chat endpoint")
    parser.add_argument(
        "--base-url",
        default="https://shl-catalog-recommendation.onrender.com",
        help="Base URL of the running service (default: Render deployment)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Print reply/recs on failures")
    parser.add_argument("-o", "--output", help="Write full JSON results to this file")
    args = parser.parse_args()

    report = run_probes(args.base_url, verbose=args.verbose)

    if args.output:
        out = Path(args.output)
        out.write_text(json.dumps(report, indent=2))
        print(f"Full report written to {out}")

    # Exit non-zero if any probe failed (useful for CI)
    sys.exit(0 if report["passed"] == report["total"] else 1)


if __name__ == "__main__":
    main()
