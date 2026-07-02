"""
Offline eval against the 10 provided conversation traces (C1.md .. C10.md).

This sandbox can't reach the Groq/OpenRouter/Gemini APIs (network is
allow-listed to package registries only), so this script tests the piece
that's fully testable offline and is the ceiling on your Recall@10 score:
retrieval. If the right assessment never makes it into the candidate list
handed to the LLM, the LLM can't recommend it -- so this tells you where
retrieval needs tuning before you even spend API calls on the live agent.

Usage:
    python3 eval/run_retrieval_eval.py

For a full live eval (agent behavior, not just retrieval), point
LLM_API_KEY/LLM_PROVIDER at a real key and adapt this script to call
run_turn(...) turn-by-turn using the "User" lines parsed from each trace,
then diff the final `recommendations` against the parsed ground truth.
"""
import difflib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import _build_candidates
from app.retrieval import get_index

TRACE_DIR = Path(__file__).resolve().parent

NAME_RE = re.compile(r"\|\s*\d+\s*\|\s*(.+?)\s*\|")


def parse_trace(path: Path):
    text = path.read_text()
    user_turns = re.findall(r"\*\*User\*\*\s*\n\s*>\s*(.+?)(?:\n\n|\Z)", text, re.DOTALL)
    # ground truth = names in the LAST markdown table in the file (final confirmed shortlist)
    tables = re.findall(r"\|\s*#\s*\|.*?\n((?:\|.*\n?)+)", text)
    final_names = []
    if tables:
        last_table = tables[-1]
        for line in last_table.strip().splitlines():
            if line.startswith("|---") or line.strip().startswith("|#") :
                continue
            m = NAME_RE.match(line)
            if m:
                final_names.append(m.group(1).strip())
    return user_turns, final_names


def main():
    index = get_index()
    total_recall = 0.0
    n = 0
    print(f"{'trace':<8} {'expected':<10} {'found_in_pool':<16} pool_recall  (pool size)")
    for path in sorted(TRACE_DIR.glob("C*.md")):
        user_turns, expected = parse_trace(path)
        if not expected:
            continue
        full_query = " ".join(user_turns)
        candidates = _build_candidates(index, full_query, full_query)
        all_names = {c.name for c in candidates}
        all_names_lower = {n.lower() for n in all_names}

        def _is_hit(expected_name: str) -> bool:
            # exact match
            if expected_name in all_names:
                return True
            # case-insensitive containment (handles extra suffixes, dashes, etc.)
            e_lower = expected_name.lower()
            for n in all_names_lower:
                if e_lower in n or n in e_lower:
                    return True
            # fuzzy match (handles minor spelling/punctuation differences)
            matches = difflib.get_close_matches(
                e_lower, list(all_names_lower), n=1, cutoff=0.72
            )
            return bool(matches)

        hits = sum(1 for e in expected if _is_hit(e))
        recall = hits / len(expected)
        total_recall += recall
        n += 1
        missing = [e for e in expected if not _is_hit(e)]
        print(
            f"{path.stem:<8} {len(expected):<10} {hits}/{len(expected):<10} "
            f"{recall:.2f}         ({len(candidates)})   missing: {missing}"
        )

    if n:
        print(f"\nMean pool Recall (ceiling for the LLM's picks) across {n} traces: {total_recall / n:.3f}")
        print("\nNote: this is an UPPER BOUND on Recall@10 -- it only checks whether the right item")
        print("reaches the LLM at all. The agent's actual Recall@10 also depends on the LLM choosing")
        print("correctly from this pool, which needs a live LLM call to measure (see docstring above).")


if __name__ == "__main__":
    main()
