"""Primary -> secondary model fallback, retries for structured output, and friendly errors.

Every part of the app gets its model from `get_chat_model()`, so the fallback
behaviour is defined in exactly one place.
"""
import logging
from typing import Callable, List, Optional, Tuple, Type

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ValidationError

from src.llm.model import available_models

logger = logging.getLogger(__name__)


class LLMUnavailableError(Exception):
    """Raised when no model can answer. The message is safe to show to students."""

    def __init__(self, message: str, status_code: int = 503):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def friendly_error(exc: BaseException) -> Tuple[str, int]:
    """Turn a provider exception into (student-friendly message, HTTP status).

    The raw exception text is never returned, so nothing sensitive (such as a
    key echoed back by a provider) can leak to the user interface.
    """
    text = f"{type(exc).__name__} {exc}".lower()

    if any(word in text for word in ("429", "rate limit", "rate_limit", "quota", "resourceexhausted", "too many requests")):
        return ("The AI service is busy or the free-tier limit was reached. Please wait a minute and try again.", 429)
    if any(word in text for word in ("401", "403", "api key", "api_key", "unauthorized", "permission", "authentication")):
        return ("The AI service rejected the API key. Please check the keys in your .env file or deployment secrets.", 503)
    if any(word in text for word in ("timeout", "timed out", "connect", "connection", "network")):
        return ("The AI service took too long to respond. Please try again.", 504)
    if "404" in text or "not found" in text:
        return ("The configured AI model was not found. Check GROQ_MODEL / GEMINI_MODEL in your settings.", 503)
    return ("The AI service returned an error. Please try again in a moment.", 502)


def get_chat_model(
    temperature: float = 0.2,
    tools: Optional[list] = None,
    schema: Optional[Type[BaseModel]] = None,
) -> Runnable:
    """Return a chat model that tries the primary provider, then the secondary.

    tools:  bind LangChain tools so the model can ask to call them.
    schema: make the model return a validated Pydantic object.
    Each provider model is prepared first and the fallback chain is built after,
    because the fallback wrapper itself cannot bind tools.
    """
    models = available_models(temperature)
    if not models:
        raise LLMUnavailableError(
            "No AI model is configured. Add GROQ_API_KEY and/or GOOGLE_API_KEY to your .env file.",
            status_code=503,
        )

    prepared: List[Runnable] = []
    for model in models:
        if tools:
            model = model.bind_tools(tools)
        if schema is not None:
            model = model.with_structured_output(schema)
        prepared.append(model)

    if len(prepared) == 1:
        return prepared[0]
    return prepared[0].with_fallbacks(prepared[1:])


def safe_invoke(runnable: Runnable, prompt):
    """Call a model and convert provider failures into LLMUnavailableError.

    Parsing/validation problems are passed through unchanged so callers can
    decide to retry or choose a safe default.
    """
    try:
        return runnable.invoke(prompt)
    except LLMUnavailableError:
        raise
    except (ValidationError, OutputParserException):
        raise
    except Exception as exc:  # provider errors: rate limit, auth, network...
        logger.warning("LLM call failed: %s", type(exc).__name__)
        message, status = friendly_error(exc)
        raise LLMUnavailableError(message, status) from exc


def invoke_structured(
    schema: Type[BaseModel],
    messages: List[BaseMessage],
    temperature: float = 0.3,
    validate: Optional[Callable[[BaseModel], None]] = None,
    attempts: int = 2,
):
    """Ask for a Pydantic object, re-asking once if the answer is invalid.

    `validate` may raise ValueError for rules that the schema alone cannot
    express (for example "exactly 7 days"). The error text is sent back to the
    model on the retry so it can correct itself.
    """
    model = get_chat_model(temperature=temperature, schema=schema)
    last_problem = ""
    for attempt in range(attempts):
        prompt = list(messages)
        if last_problem:
            prompt.append(
                HumanMessage(content=f"Your previous answer was invalid: {last_problem} Please fix it and answer again.")
            )
        try:
            result = safe_invoke(model, prompt)
            if result is None:
                raise ValueError("the model returned no structured data")
            if not isinstance(result, schema):
                result = schema.model_validate(result)
            if validate is not None:
                validate(result)
            return result
        except (ValidationError, OutputParserException, ValueError) as exc:
            last_problem = str(exc)[:300]
            logger.info("Structured output attempt %d invalid: %s", attempt + 1, last_problem)

    raise LLMUnavailableError(
        "The AI could not produce a valid result. Please try again or simplify your request.", status_code=502
    )
