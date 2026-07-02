import sys
from pathlib import Path

# Add workspace root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
from app.agent import run_turn, get_index
from app.schemas import Message

def test_off_topic_refusal():
    print("Running Probe 1: Off-topic refusal...")
    msg = Message(role="user", content="How do I bake a chocolate chip cookie?")
    res = run_turn([msg])
    assert not res["recommendations"], "Recommendations must be empty for off-topic requests"
    assert not res["end_of_conversation"], "Conversation should not end on off-topic refusal"
    print("✅ Probe 1 Passed (refused off-topic correctly)")

def test_vague_query_clarification():
    print("Running Probe 2: Vague query clarification...")
    msg = Message(role="user", content="I want to hire someone. What tests do you have?")
    res = run_turn([msg])
    assert not res["recommendations"], "Recommendations must be empty for vague queries on turn 1"
    assert not res["end_of_conversation"], "Conversation should not end on clarification"
    print("✅ Probe 2 Passed (requested clarification correctly)")

def test_honors_edits_drop_assessment():
    print("Running Probe 3: Honors edits (dropping assessments)...")
    history = [
        Message(role="user", content="I need to assess a backend developer with Java and SQL knowledge."),
        Message(role="assistant", content="I recommend Core Java (Advanced Level) (New) and SQL (New)."),
        Message(role="user", content="Actually, drop the Java test. Only recommend SQL.")
    ]
    res = run_turn(history)
    rec_names = {r["name"] for r in res["recommendations"]}
    assert "SQL (New)" in rec_names or "sql" in "".join(rec_names).lower(), "Should still recommend SQL"
    assert "Core Java (Advanced Level) (New)" not in rec_names, "Should have dropped Core Java"
    print("✅ Probe 3 Passed (honored dropped assessment correctly)")

def test_no_hallucinations_and_catalog_only():
    print("Running Probe 4: No hallucinations (catalog-only matching)...")
    index = get_index()
    # Query with a made-up test name to see if it's ignored or resolved to a valid one
    msg = Message(role="user", content="Recommend the 'Super Duper Rust Programming Extreme Edition 2026' test.")
    res = run_turn([msg])
    for rec in res["recommendations"]:
        assert rec["name"] in index._name_lower_to_item, f"Recommended item {rec['name']} is not in the catalog!"
    print("✅ Probe 4 Passed (no hallucinated catalog items)")

def main():
    try:
        test_off_topic_refusal()
        time.sleep(2)
        test_vague_query_clarification()
        time.sleep(2)
        test_honors_edits_drop_assessment()
        time.sleep(2)
        test_no_hallucinations_and_catalog_only()
        print("\n🎉 ALL BEHAVIOR PROBES PASSED!")
    except AssertionError as e:
        print(f"\n❌ Probe Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
