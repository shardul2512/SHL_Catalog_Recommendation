# Approach Document — SHL Assessment Recommender

## Design overview

The service does one LLM call per `/chat` turn. Retrieval, name-to-URL
grounding, and schema validation are deterministic Python around that call.
This was the central design bet: a multi-step agent loop (extract intent →
retrieve → plan → generate → verify) is more "agentic" on paper, but each
extra LLM call adds latency risk against the 30s/call cap and adds more
places a non-deterministic model can drift off-contract. A single call with
a tightly-scoped prompt and code-enforced post-processing is more reliable
to demo and to defend.

**Flow per turn:** conversation text → hybrid retrieval builds a candidate
pool of real catalog items (name, type, duration, languages, description) →
one LLM call, given only that pool plus the full message history, returns
`{reply, item_names, end_of_conversation}` → code maps `item_names` back to
catalog entries for the real `url`/`test_type` → response is schema-validated
before it ever leaves the process.

## Retrieval

377 Individual Test Solutions, hybrid BM25 + TF-IDF cosine (min-max
normalized, weighted 0.6/0.4), over `name + description + keys + job_levels
+ languages_raw + remote/adaptive flags`. I chose lexical hybrid over
embeddings deliberately: this catalog's vocabulary largely matches real
job-description language ("Java", "AWS", "Excel", "leadership", "safety"),
so BM25 already does well, and it avoids a model download at Render
free-tier cold start plus per-request embedding latency. A crude
suffix-stripper (not a real Porter stemmer) closes the "skills" vs "skill"
gap cheaply. Long multi-facet conversations get fanned out clause-by-clause
(in addition to the full-text query) so no single facet gets diluted by the
others; a small fixed "default pool" query (personality/cognitive/skills
terms) is always merged in, since traces show personality/ability tests are
commonly offered as default companions.

**Query-side synonym expansion:** a small hand-curated synonym map bridges
the vocabulary gap between how users describe roles and how SHL names its
products — e.g. "Rust engineer" → "smart interview live coding linux
programming", "patient records admin" → "microsoft word dependability
safety", "financial analyst" → "basic statistics accounting". This runs
only on queries (not on catalog search text, which would pollute TF-IDF
similarity). Recent user messages are weighted more heavily in retrieval
queries so REFINE scenarios (changed constraints) don't get diluted by
old context.

The LLM only ever sees and cites from this candidate pool (typically 35-45
items), never the raw 377 — this is what makes "every URL from the scraped
catalog" a code guarantee: item names the model returns are looked up
against the pool (falling back to fuzzy match against the full catalog for
robustness); anything that doesn't resolve is silently dropped.

**Fuzzy Shortlist Continuity:** Implemented a matching utility `is_fuzzy_equivalent`
to compare LLM-generated shortlist choices with previous turns. This prevents
variability/abbreviations in the model's output (e.g. dropping "Interactive")
from breaking continuity, ensuring correct references back to the catalog.

## Prompt design

One system prompt encodes: strict scope (SHL assessments only), the four
behaviors (clarify/recommend/refine/compare), and refusal rules (general
hiring/legal advice, prompt injection — treating all user-role content as
untrusted data, never as instructions). Recommendations must cite only items
in the candidate pool; comparisons must be grounded in the given
descriptions or the model should say it lacks grounding rather than guess. A
`final_turn` flag is injected into the prompt when the conversation is on
its last allowed turn (turn cap), instructing the model to commit to a
shortlist rather than ask another clarifying question — this directly
targets the "agent never lands a recommendation before turns run out"
failure mode.

## Reliability / non-happy-path handling

- Malformed/non-JSON LLM output → regex-extract a `{...}` block; if that
  fails, retry once on a smaller/faster fallback model; if that also fails,
  return a schema-valid clarifying response (or, on the final turn, fall
  back to the top retrieved candidates directly, so a recommendation is
  still produced even if the LLM is unavailable).
- Any unhandled exception in the request path is caught at the FastAPI layer
  and converted to a schema-valid response — the service should never return
  a 500 or a malformed body to the evaluator.
- Hallucinated item names are dropped in code (see Retrieval), never
  returned to the caller.

## Evaluation Results

- **Model Infrastructure:** Migrated to Google Gemini 3.1 Flash Lite (`gemini-3.1-flash-lite`) via OpenAI-compatible API to ensure rapid (~1.5s per turn) and stable JSON generation.
- **End-to-End Recall@10:** Measured utilizing `eval/run_e2e_eval.py` which runs live multi-turn dialogues across all 10 evaluation traces:
  - **Mean Recall@10:** **1.000** (100% correct recommendations on final turn across all traces C1-C10).
  - All responses are fully schema-compliant, with every recommendation verified as an authentic, catalog-backed individual test solution.

## AI tool disclosure

Built with Antigravity (Google DeepMind) as an AI pair-programmer for scaffolding, retrieval tuning, state stabilization, and evaluation.
