"""
End-to-end Recall@10 evaluation against the 10 provided conversation traces.

Unlike run_retrieval_eval.py (which only measures the retrieval pool ceiling),
this script actually calls the LLM via run_turn() to simulate a full
multi-turn conversation for each trace, then measures Recall@10 on the
final recommendations produced.

The simulated "user" replays each trace's User turns sequentially, feeding
the agent's responses back as assistant messages. The agent's final
shortlist (the last non-empty recommendations list) is compared against
the trace's ground-truth shortlist.

Usage:
    # Requires a working LLM API key (set in .env or environment)
    python3 eval/run_e2e_eval.py

    # Or run a single trace:
    python3 eval/run_e2e_eval.py C1
"""
import difflib
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.schemas import Message
from app.agent import run_turn

TRACE_DIR = Path(__file__).resolve().parent

NAME_RE = re.compile(r"\|\s*\d+\s*\|\s*(.+?)\s*\|")


def parse_trace(path: Path):
    """Parse a trace file to extract user turns and ground-truth shortlist."""
    text = path.read_text()
    user_turns = re.findall(
        r"\*\*User\*\*\s*\n\s*>\s*(.+?)(?:\n\n|\Z)", text, re.DOTALL
    )
    # ground truth = names in the LAST markdown table in the file
    tables = re.findall(r"\|\s*#\s*\|.*?\n((?:\|.*\n?)+)", text)
    final_names = []
    if tables:
        last_table = tables[-1]
        for line in last_table.strip().splitlines():
            if line.startswith("|---") or line.strip().startswith("|#"):
                continue
            m = NAME_RE.match(line)
            if m:
                final_names.append(m.group(1).strip())
    return user_turns, final_names


def fuzzy_match(expected_name: str, actual_names: set) -> bool:
    """Check if expected_name matches any actual name (exact, containment, or fuzzy)."""
    if expected_name in actual_names:
        return True
    e_lower = expected_name.lower()
    actual_lower = {n.lower() for n in actual_names}
    if e_lower in actual_lower:
        return True
    for n in actual_lower:
        if e_lower in n or n in e_lower:
            return True
    matches = difflib.get_close_matches(e_lower, list(actual_lower), n=1, cutoff=0.72)
    return bool(matches)


def run_trace(path: Path, verbose: bool = True) -> dict:
    """Run a single trace end-to-end and return results."""
    user_turns, expected = parse_trace(path)
    if not expected:
        return {"trace": path.stem, "error": "no ground truth found"}

    messages = []
    last_recommendations = []
    turn_details = []

    for i, user_text in enumerate(user_turns):
        messages.append(Message(role="user", content=user_text.strip()))

        t0 = time.time()
        result = run_turn(messages)
        elapsed = time.time() - t0

        reply = result.get("reply", "")
        recs = result.get("recommendations", [])
        eoc = result.get("end_of_conversation", False)

        if recs:
            last_recommendations = recs

        turn_info = {
            "turn": i + 1,
            "user": user_text.strip()[:80],
            "reply_len": len(reply),
            "n_recs": len(recs),
            "eoc": eoc,
            "elapsed": f"{elapsed:.1f}s",
        }
        turn_details.append(turn_info)

        if verbose:
            print(f"  Turn {i+1}: user='{user_text.strip()[:60]}...'")
            print(f"    -> reply={len(reply)} chars, recs={len(recs)}, eoc={eoc}, {elapsed:.1f}s")
            if recs:
                for r in recs:
                    print(f"       - {r['name']}")

        # Add assistant response to history for next turn
        messages.append(Message(role="assistant", content=reply))

        if eoc:
            break
            
        # Rate-limiting sleep to avoid hitting TPM/RPM limits on free API keys
        time.sleep(2.0)

    # Compute Recall@10
    recommended_names = {r["name"] for r in last_recommendations}
    hits = sum(1 for e in expected if fuzzy_match(e, recommended_names))
    recall = hits / len(expected) if expected else 0.0
    missing = [e for e in expected if not fuzzy_match(e, recommended_names)]
    extra = [n for n in recommended_names if not any(fuzzy_match(e, {n}) for e in expected)]

    return {
        "trace": path.stem,
        "expected": len(expected),
        "recommended": len(recommended_names),
        "hits": hits,
        "recall_at_10": recall,
        "missing": missing,
        "extra": extra,
        "turns_used": len(turn_details),
        "turn_details": turn_details,
    }


def main():
    # Optional: run a single trace
    single_trace = None
    if len(sys.argv) > 1:
        single_trace = sys.argv[1]

    traces = sorted(TRACE_DIR.glob("C*.md"))
    if single_trace:
        traces = [t for t in traces if t.stem == single_trace]
        if not traces:
            print(f"Trace {single_trace} not found")
            sys.exit(1)

    total_recall = 0.0
    n = 0
    results = []

    print(f"{'trace':<8} {'expected':<10} {'recommended':<13} {'hits':<6} {'recall@10':<10} {'turns':<6}")
    print("-" * 70)

    for path in traces:
        print(f"\n=== {path.stem} ===")
        result = run_trace(path, verbose=True)
        results.append(result)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
            continue

        total_recall += result["recall_at_10"]
        n += 1

        print(
            f"\n{result['trace']:<8} {result['expected']:<10} "
            f"{result['recommended']:<13} {result['hits']:<6} "
            f"{result['recall_at_10']:.2f}      {result['turns_used']:<6}"
        )
        if result["missing"]:
            print(f"  MISSING: {result['missing']}")
        if result["extra"]:
            print(f"  EXTRA:   {result['extra']}")
        
        # Sleep between traces to stay safe from rate limits
        time.sleep(5.0)

    if n:
        mean_recall = total_recall / n
        print(f"\n{'=' * 70}")
        print(f"Mean Recall@10 across {n} traces: {mean_recall:.3f}")
        print(f"{'=' * 70}")

        if mean_recall >= 0.8:
            print("✅ PASS: Mean Recall@10 >= 0.80")
        else:
            print(f"❌ BELOW TARGET: Mean Recall@10 = {mean_recall:.3f} (target: >= 0.80)")
            print("Consider: prompt tuning, retrieval expansion, or model upgrade")


if __name__ == "__main__":
    main()
