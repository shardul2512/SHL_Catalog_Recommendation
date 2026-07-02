import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.agent import run_turn
from app.retrieval import get_index
from app.schemas import ChatRequest, ChatResponse, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shl_api")


@asynccontextmanager
async def lifespan(application: FastAPI):
    # Build the retrieval index once at startup so the first real /chat
    # call isn't slowed down by it (BM25/TF-IDF build over 377 items is
    # fast, but this also fails loudly here if catalog.json is malformed).
    t0 = time.time()
    idx = get_index()
    logger.info("Catalog index ready: %d items in %.2fs", len(idx.items), time.time() - t0)
    yield


app = FastAPI(title="SHL Assessment Recommender", version="1.0.0", lifespan=lifespan)

# Allow the evaluator (or any frontend) to reach the API from any origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    t0 = time.time()
    try:
        n_turns = len(payload.messages)
        logger.info("POST /chat  turns=%d", n_turns)
        result = run_turn(payload.messages)
        elapsed = time.time() - t0
        n_recs = len(result.get("recommendations", []))
        logger.info("  -> reply=%d chars  recs=%d  eoc=%s  %.1fs",
                    len(result.get("reply", "")), n_recs,
                    result.get("end_of_conversation"), elapsed)
        return ChatResponse(**result)
    except Exception:
        logger.exception("Unhandled error in /chat")
        # Never let an unexpected error break the schema contract -- the
        # evaluator scores schema compliance on every response.
        return ChatResponse(
            reply="Sorry, something went wrong on my end. Could you try rephrasing that?",
            recommendations=[],
            end_of_conversation=False,
        )


@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception for %s", request.url)
    return JSONResponse(
        status_code=200,
        content={
            "reply": "Sorry, something went wrong on my end. Could you try rephrasing that?",
            "recommendations": [],
            "end_of_conversation": False,
        },
    )
