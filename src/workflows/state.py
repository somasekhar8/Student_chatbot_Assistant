"""The shared state that flows through the LangGraph workflow, plus small message helpers."""
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages

# Safe stopping condition for the tool loop: the agent may call tools at most
# this many times per student message, then it must answer.
MAX_TOOL_ROUNDS = 2
# How many earlier student messages are sent to the LLM as context.
MAX_HISTORY_TURNS = 6


class AgentState(TypedDict, total=False):
    # Whole conversation. `add_messages` appends new messages (and replaces
    # a message when the same id is returned again).
    messages: Annotated[List[AnyMessage], add_messages]

    # Input: "rag" when the PDF tab forces document-only answers, else None.
    forced_route: Optional[str]

    # Set by the router for the current message:
    route: str
    has_documents: bool

    # Filled by the finalize node for the current message:
    sources: List[Dict[str, Any]]
    contexts: List[str]
    structured: Optional[Dict[str, Any]]


def message_text(message) -> str:
    """Plain text of a message (Gemini sometimes returns a list of content blocks)."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def current_turn(messages: List[AnyMessage]) -> List[AnyMessage]:
    """Messages added since the student's latest message (that message included)."""
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return messages[index:]
    return list(messages)


def tool_rounds(messages: List[AnyMessage]) -> int:
    """How many tool results exist for the current student message."""
    return sum(isinstance(m, ToolMessage) for m in current_turn(messages))


def last_user_text(messages: List[AnyMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message_text(message)
    return ""


def build_history(messages: List[AnyMessage], max_turns: int = MAX_HISTORY_TURNS) -> List[AnyMessage]:
    """Messages to send to the LLM.

    Earlier turns are reduced to the student's questions and the final answers
    (old tool calls and search results are dropped to save tokens). The current
    turn is kept in full, including any tool calls and results.
    """
    turn = current_turn(messages)
    earlier = messages[: len(messages) - len(turn)]

    compact: List[AnyMessage] = []
    for message in earlier:
        if isinstance(message, HumanMessage):
            compact.append(message)
        elif isinstance(message, AIMessage) and not message.tool_calls and message_text(message).strip():
            compact.append(message)

    human_positions = [i for i, m in enumerate(compact) if isinstance(m, HumanMessage)]
    if len(human_positions) > max_turns:
        compact = compact[human_positions[-max_turns]:]
    return compact + list(turn)
