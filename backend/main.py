"""FastAPI application. Run with:  uv run uvicorn backend.main:app --reload --port 8000

The backend is independent of Streamlit: any client (Streamlit, curl, a mobile
app) can call it. Interactive docs are served at http://localhost:8000/docs
"""
import logging
import os

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from backend.routes import chat, quiz, study_plan, upload  # noqa: E402
from src.llm.fallback import LLMUnavailableError  # noqa: E402
from src.llm.model import configured_providers  # noqa: E402
from src.rag.document_loader import DocumentError  # noqa: E402
from src.schemas.chat_schema import HealthResponse  # noqa: E402

VERSION = "1.0.0"
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

app = FastAPI(
    title="Student Assistant API",
    version=VERSION,
    description="RAG + LangGraph backend for the Agentic AI Student Assistant.",
)

# Comma separated list of allowed browser origins. Streamlit talks to this API
# from the server side, so the default is strict.
allowed_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


@app.exception_handler(LLMUnavailableError)
async def llm_error_handler(_request: Request, exc: LLMUnavailableError):
    """Rate limits, bad keys, timeouts... always a short, safe message."""
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


@app.exception_handler(DocumentError)
async def document_error_handler(_request: Request, exc: DocumentError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def unexpected_error_handler(_request: Request, exc: Exception):
    logging.getLogger(__name__).exception("Unhandled error")
    return JSONResponse(status_code=500, content={"detail": "Something went wrong on the server. Please try again."})


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Liveness check. Lists which providers have a key; never returns the keys."""
    return HealthResponse(status="ok", version=VERSION, providers_configured=configured_providers())


app.include_router(upload.router)
app.include_router(chat.router)
app.include_router(study_plan.router)
app.include_router(quiz.router)
