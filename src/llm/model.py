"""Builds the Groq and Gemini chat models from environment variables.

Nothing is hardcoded: API keys and model names all come from the environment
(or a local .env file), so no secret ever lives in the source code.

Environment variables
---------------------
PRIMARY_LLM_PROVIDER  "groq" (default) or "gemini"
GROQ_API_KEY          key for Groq
GOOGLE_API_KEY        key for Gemini (GEMINI_API_KEY is accepted as an alias)
GROQ_MODEL            default "openai/gpt-oss-120b"   (the model your project already used)
GEMINI_MODEL          default "gemini-3.6-flash"      (the model your project already used)
"""
import logging
import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()
logger = logging.getLogger(__name__)

VALID_PROVIDERS = ("groq", "gemini")
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
REQUEST_TIMEOUT_SECONDS = 30


def primary_provider() -> str:
    """Which provider is tried first. Falls back to groq on a bad value."""
    value = os.getenv("PRIMARY_LLM_PROVIDER", "groq").strip().lower()
    if value not in VALID_PROVIDERS:
        logger.warning("PRIMARY_LLM_PROVIDER=%r is not valid; using 'groq'.", value)
        return "groq"
    return value


def _clean_key(value: Optional[str]) -> Optional[str]:
    """Treat empty values and copy-pasted placeholders as 'not set'."""
    if not value:
        return None
    value = value.strip().strip('"').strip("'")
    if not value or value.lower().startswith(("your", "groq_api_key", "<")):
        return None
    return value


def get_api_key(provider: str) -> Optional[str]:
    if provider == "groq":
        return _clean_key(os.getenv("GROQ_API_KEY"))
    return _clean_key(os.getenv("GOOGLE_API_KEY")) or _clean_key(os.getenv("GEMINI_API_KEY"))


def provider_order() -> List[str]:
    """Primary provider first, the other one second (used as the fallback)."""
    first = primary_provider()
    return [first] + [p for p in VALID_PROVIDERS if p != first]


def configured_providers() -> List[str]:
    """Provider names that have a usable key. Never returns the keys."""
    return [p for p in provider_order() if get_api_key(p)]


def build_model(provider: str, temperature: float = 0.2) -> Optional[BaseChatModel]:
    """Create one chat model, or None when its API key is missing."""
    key = get_api_key(provider)
    if not key:
        return None

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
            temperature=temperature,
            api_key=key,
            timeout=REQUEST_TIMEOUT_SECONDS,
            max_retries=1,
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        temperature=temperature,
        google_api_key=key,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=1,
    )


def available_models(temperature: float = 0.2) -> List[BaseChatModel]:
    """All models that can be built, in priority order (primary first)."""
    models = []
    for provider in provider_order():
        model = build_model(provider, temperature)
        if model is not None:
            models.append(model)
    return models
