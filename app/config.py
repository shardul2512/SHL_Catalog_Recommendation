"""
Centralized configuration, loaded from environment variables.

Supports any OpenAI-compatible chat completions endpoint, so you can point
this at Groq, OpenRouter, or Gemini's OpenAI-compat endpoint just by
changing env vars -- no code changes needed.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent

# Load environment variables from the common local env locations before any
# os.getenv() calls below. This lets the app run directly from uvicorn without
# requiring manual export of secrets in the shell.
load_dotenv(BASE_DIR / ".env")
load_dotenv(APP_DIR / ".env")
CATALOG_PATH = os.getenv("CATALOG_PATH", str(BASE_DIR / "data" / "catalog.json"))

# --- LLM provider config -----------------------------------------------
# Pick a provider via LLM_PROVIDER=groq|openrouter|gemini|custom
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

_PROVIDER_DEFAULTS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "api_key_env": "GROQ_API_KEY",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-3.1-flash-lite",
        "api_key_env": "GEMINI_API_KEY",
    },
}

_provider_cfg = _PROVIDER_DEFAULTS.get(LLM_PROVIDER, _PROVIDER_DEFAULTS["groq"])

LLM_BASE_URL = os.getenv("LLM_BASE_URL", _provider_cfg["base_url"])
LLM_MODEL = os.getenv("LLM_MODEL", _provider_cfg["model"])
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv(_provider_cfg["api_key_env"], "")

# Fallback model used if the primary call fails or times out once, before
# we give up and return a safe clarifying response. Keep this small/fast.
LLM_FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "gemini-3.1-flash-lite")

# Hard budget so a single /chat call never blows the evaluator's 30s cap.
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "18"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "25"))

# Retrieval
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "28"))

MAX_TURNS = int(os.getenv("MAX_TURNS", "8"))
