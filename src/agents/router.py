"""Router node (decides the workflow) and Clarification node (asks when unsure)."""
import logging

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from src.llm.fallback import get_chat_model, safe_invoke
from src.llm.prompts import CLARIFY_PROMPT, ROUTER_PROMPT
from src.rag.vector_store import get_index_store
from src.schemas.chat_schema import RouteDecision
from src.workflows.state import AgentState, build_history, last_user_text, message_text

logger = logging.getLogger(__name__)
VALID_ROUTES = ("rag", "planner", "quiz", "general", "clarify")


def _short_transcript(messages, limit: int = 4) -> str:
    """The last few chat messages as plain text, so the router can read follow-ups."""
    lines = []
    for message in messages[:-1]:
        text = message_text(message).strip()
        if isinstance(message, HumanMessage):
            lines.append(f"Student: {text[:300]}")
        elif isinstance(message, AIMessage) and text and not message.tool_calls:
            lines.append(f"Assistant: {text[:300]}")
    return "\n".join(lines[-limit:]) or "(no earlier messages)"


def router_node(state: AgentState, config: RunnableConfig) -> dict:
    """Pick the workflow for the newest student message.

    The result is validated twice: `RouteDecision` only allows five values, and
    anything unexpected (or unparsable) is sent to the Clarification node instead
    of guessing. Per-message fields are reset here because the checkpointer keeps
    the previous turn's state.
    """
    session_id = config.get("configurable", {}).get("session_id", "")
    has_documents = get_index_store().has_documents(session_id)
    reset = {"has_documents": has_documents, "sources": [], "contexts": [], "structured": None}

    forced = state.get("forced_route")
    if forced in VALID_ROUTES:
        return {**reset, "route": forced}

    messages = state["messages"]
    prompt = [
        SystemMessage(content=ROUTER_PROMPT.format(has_documents="yes" if has_documents else "no")),
        HumanMessage(
            content=f"Recent conversation:\n{_short_transcript(messages)}\n\nLatest student message:\n{last_user_text(messages)}"
        ),
    ]
    try:
        decision = safe_invoke(get_chat_model(temperature=0, schema=RouteDecision), prompt)
        route = decision.route if decision and decision.route in VALID_ROUTES else "clarify"
    except (ValidationError, OutputParserException):
        logger.info("Router output was invalid; asking for clarification.")
        route = "clarify"
    return {**reset, "route": route}


def clarify_node(state: AgentState, config: RunnableConfig) -> dict:
    """Ask one short question when the request is ambiguous."""
    model = get_chat_model(temperature=0.3)
    reply = safe_invoke(model, [SystemMessage(content=CLARIFY_PROMPT)] + build_history(state["messages"]))
    return {"messages": [AIMessage(content=message_text(reply))]}
