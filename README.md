# SHL Assessment Recommender

A stateless conversational agent that recommends SHL Individual Test Solutions
via a `POST /chat` endpoint, per the take-home spec.

## Architecture

```
messages[] ──▶ retrieval (BM25 + TF-IDF hybrid, in-process, no external calls)
                     │
                     ▼
          candidate pool (~25-40 real catalog items, deduped)
                     │
                     ▼
      single LLM call (Groq/OpenRouter/Gemini, OpenAI-compatible)
      system prompt encodes: clarify / recommend / refine / compare / refuse
                     │
                     ▼
   JSON parse + repair ──▶ ground item names back to catalog (name→url/type)
                     │
                     ▼
            ChatResponse (schema-validated, always)
```

Key design decisions (see `approach.md` for the full writeup):

- **One LLM call per turn.** Retrieval and grounding are deterministic Python,
  not additional LLM calls -- keeps latency predictable under the 30s/call
  budget and makes "only catalog URLs, ever" a code-enforced guarantee
  rather than a prompt request.
- **Lexical hybrid retrieval (BM25 + TF-IDF), not embeddings.** With 377
  catalog items and vocabulary that mostly overlaps with real job-description
  language, this gets most of the benefit of embeddings without a model
  download at cold start on Render's free tier. See `eval/run_retrieval_eval.py`
  for measured recall and the documented ceiling/limitations.
- **Grounding, not trust.** The LLM only ever selects from item *names* it was
  shown this turn; the API layer maps names back to real catalog entries for
  the actual URL/test_type. A name the LLM invents that isn't in the catalog
  is silently dropped, never returned.
- **Never breaks the schema.** JSON parse failures, LLM/network errors, and
  unexpected exceptions all fall through to a schema-valid fallback response
  instead of a 500 or malformed body.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in an API key for your chosen provider
```

## Run locally

```bash
export $(cat .env | xargs)   # or use python-dotenv / your shell of choice
uvicorn app.main:app --reload --port 8000
```

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hiring a Java developer who works with stakeholders"}]}'
```

## Deploy (Render)

1. Push this repo to GitHub.
2. New Web Service on Render → connect the repo → it picks up `render.yaml`.
3. Set `GROQ_API_KEY` (or your provider's key) in the Render dashboard.
4. First `/health` call may take up to the platform's cold-start time; the
   spec allows up to 2 minutes for this.

## Eval

```bash
python3 eval/run_retrieval_eval.py
```

Parses ground-truth shortlists out of `eval/C1.md`..`C10.md` and measures
retrieval pool recall (the ceiling on what the LLM can possibly recommend).
This runs fully offline. For a true end-to-end Recall@10 measurement against
a live LLM, adapt this script to call `app.agent.run_turn` turn-by-turn using
each trace's "User" lines and diff the final `recommendations` list.

## Project layout

```
app/
  config.py     provider/env config
  catalog.py    load + normalize catalog.json, category→code mapping
  retrieval.py  BM25+TF-IDF hybrid index, fuzzy name grounding
  prompts.py    system prompt (scope, 4 behaviors, guardrails) + turn prompt
  agent.py      per-turn orchestration: retrieve → LLM call → ground → validate
  schemas.py    Pydantic request/response models (exact API contract)
  main.py       FastAPI app: GET /health, POST /chat
eval/
  C1.md..C10.md          reference traces (provided)
  run_retrieval_eval.py  offline recall measurement
data/
  catalog.json  SHL Individual Test Solutions catalog (377 items)
```
