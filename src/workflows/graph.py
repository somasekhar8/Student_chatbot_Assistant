"""The LangGraph workflow.

        START
          |
        router ──(conditional edge on state["route"])──────────────┐
          |            |             |            |                |
      rag_agent   planner_agent  quiz_agent  general_agent     clarify
          |            |             |            |                |
          └─ has tool calls? ─┐      │           END              END
             yes          no  │      │
              |           |   │      │
            tools ──┐     └─> finalize ─> END
              |     │
   (rag) back to    (planner / quiz) go to finalize
   rag_agent

The tool loop (rag_agent -> tools -> rag_agent) stops safely because:
1. after MAX_TOOL_ROUNDS tool results the agent is given NO tools, so it must answer;
2. `recursion_limit` in chat() is a second, hard limit set by LangGraph itself.
"""
import logging
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.agents.general_agent import general_agent_node
from src.agents.planner_agent import planner_agent_node
from src.agents.quiz_agent import quiz_agent_node
from src.agents.rag_agent import rag_agent_node
from src.agents.router import clarify_node, router_node
from src.llm.fallback import LLMUnavailableError, friendly_error
from src.llm.prompts import NO_DOCUMENTS_MESSAGE, NOT_FOUND_MESSAGE
from src.memory.chat_history import get_checkpointer, thread_id_for
from src.schemas.planner_schema import StudyPlan
from src.schemas.quiz_schema import Quiz
from src.tools import ALL_TOOLS
from src.workflows.state import MAX_TOOL_ROUNDS, AgentState, current_turn, message_text, tool_rounds

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Conditional edges (plain functions that return the name of the next node)
# --------------------------------------------------------------------------- #
def route_from_router(state: AgentState) -> str:
    return state["route"]


def after_agent(state: AgentState) -> str:
    """Go to the tool node only if the model asked for tools and the budget allows."""
    last = state["messages"][-1]
    wants_tools = bool(getattr(last, "tool_calls", None))
    if wants_tools and tool_rounds(state["messages"]) < MAX_TOOL_ROUNDS:
        return "tools"
    return "finalize"


def after_tools(state: AgentState) -> str:
    """The RAG agent needs the search results to write its answer; the others are done."""
    return "rag_agent" if state["route"] == "rag" else "finalize"


# --------------------------------------------------------------------------- #
# Finalize node: turns tool results into the final answer (no LLM call)
# --------------------------------------------------------------------------- #
def _collect_sources(chunks: List[dict]) -> List[dict]:
    """One citation per (document, page), best score first."""
    best: Dict[tuple, dict] = {}
    for chunk in chunks:
        key = (chunk["document"], chunk["page"])
        if key not in best or chunk["score"] > best[key]["score"]:
            best[key] = chunk
    ordered = sorted(best.values(), key=lambda c: c["score"], reverse=True)
    return [
        {
            "document": c["document"],
            "page": c["page"],
            "score": c["score"],
            "snippet": " ".join(c["text"].split())[:200],
        }
        for c in ordered
    ]


def _finalize_rag(state: AgentState) -> dict:
    messages = state["messages"]
    last = messages[-1]
    answer = message_text(last).strip()

    if answer == NO_DOCUMENTS_MESSAGE:
        return {}

    chunks: List[dict] = []
    for message in current_turn(messages):
        if isinstance(message, ToolMessage) and isinstance(message.artifact, list):
            chunks.extend(message.artifact)

    pending_tool_call = bool(getattr(last, "tool_calls", None))
    says_not_found = NOT_FOUND_MESSAGE.lower() in answer.lower()

    # Hallucination control, enforced in code (not just in the prompt):
    # no relevant passages, an empty answer, or the model giving up -> the fixed message.
    if not chunks or not answer or pending_tool_call or says_not_found:
        final = AIMessage(content=NOT_FOUND_MESSAGE) if pending_tool_call else AIMessage(content=NOT_FOUND_MESSAGE, id=last.id)
        return {"messages": [final], "sources": [], "contexts": [c["text"] for c in chunks]}

    return {
        "messages": [AIMessage(content=answer, id=last.id)],
        "sources": _collect_sources(chunks),
        "contexts": [c["text"] for c in chunks],
    }


def _finalize_structured(state: AgentState) -> dict:
    kind = "study_plan" if state["route"] == "planner" else "quiz"
    model_class = StudyPlan if kind == "study_plan" else Quiz

    tool_messages = [m for m in current_turn(state["messages"]) if isinstance(m, ToolMessage)]
    artifact = next((m.artifact for m in reversed(tool_messages) if isinstance(m.artifact, dict)), None)

    if artifact is None:
        if tool_messages:  # the tool was called but failed (bad arguments)
            what = "study plan" if kind == "study_plan" else "quiz"
            return {
                "messages": [
                    AIMessage(
                        content=f"Sorry, I could not create that {what}. Please tell me the topic and "
                        "the details (for example level, hours per day and number of days) and try again."
                    )
                ]
            }
        return {}  # the model replied with plain text (e.g. a question); keep it

    result = model_class.model_validate(artifact)
    return {
        "messages": [AIMessage(content=result.to_markdown())],
        "structured": {"type": kind, "data": result.model_dump()},
    }


def finalize_node(state: AgentState) -> dict:
    if state["route"] == "rag":
        return _finalize_rag(state)
    if state["route"] in ("planner", "quiz"):
        return _finalize_structured(state)
    return {}


# --------------------------------------------------------------------------- #
# Build the graph
# --------------------------------------------------------------------------- #
def build_graph(checkpointer=None):
    builder = StateGraph(AgentState)

    builder.add_node("router", router_node)
    builder.add_node("rag_agent", rag_agent_node)
    builder.add_node("planner_agent", planner_agent_node)
    builder.add_node("quiz_agent", quiz_agent_node)
    builder.add_node("general_agent", general_agent_node)
    builder.add_node("clarify", clarify_node)
    builder.add_node("tools", ToolNode(ALL_TOOLS))
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        route_from_router,
        {
            "rag": "rag_agent",
            "planner": "planner_agent",
            "quiz": "quiz_agent",
            "general": "general_agent",
            "clarify": "clarify",
        },
    )
    for agent in ("rag_agent", "planner_agent", "quiz_agent"):
        builder.add_conditional_edges(agent, after_agent, {"tools": "tools", "finalize": "finalize"})
    builder.add_conditional_edges("tools", after_tools, {"rag_agent": "rag_agent", "finalize": "finalize"})

    builder.add_edge("general_agent", END)
    builder.add_edge("clarify", END)
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)


_graph_cache = {"checkpointer": None, "graph": None}


def get_graph():
    """The compiled graph, built once per checkpointer."""
    checkpointer = get_checkpointer()
    if _graph_cache["checkpointer"] is not checkpointer:
        _graph_cache["graph"] = build_graph(checkpointer)
        _graph_cache["checkpointer"] = checkpointer
    return _graph_cache["graph"]


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def _run_config(session_id: str, mode: str) -> dict:
    return {
        "configurable": {
            "thread_id": thread_id_for(session_id, mode),  # conversation memory key
            "session_id": session_id,  # document index key (used by the search tool)
        },
        "recursion_limit": 25,  # hard safety net against endless loops
    }


def _rollback_failed_turn(graph, config: dict) -> None:
    """Remove a half-finished turn so the saved history never has a question without an answer."""
    try:
        snapshot = graph.get_state(config)
        messages = snapshot.values.get("messages", [])
        turn = current_turn(messages)
        if turn and isinstance(turn[0], HumanMessage):
            graph.update_state(config, {"messages": [RemoveMessage(id=m.id) for m in turn if m.id]})
    except Exception:  # never hide the original error
        logger.exception("Could not roll back a failed turn")


def chat(session_id: str, message: str, mode: str = "auto") -> Dict[str, Any]:
    """Run one student message through the workflow.

    mode="pdf" skips the router and forces the document-only RAG workflow.
    Returns: answer, route, sources, structured, contexts.
    """
    graph = get_graph()
    config = _run_config(session_id, mode)
    forced_route = "rag" if mode == "pdf" else None

    try:
        state = graph.invoke({"messages": [HumanMessage(content=message)], "forced_route": forced_route}, config)
    except LLMUnavailableError:
        _rollback_failed_turn(graph, config)
        raise
    except GraphRecursionError as exc:
        _rollback_failed_turn(graph, config)
        raise LLMUnavailableError("The assistant took too many steps. Please rephrase your question.", 500) from exc
    except Exception as exc:
        logger.exception("Chat workflow failed")
        _rollback_failed_turn(graph, config)
        text, status = friendly_error(exc)
        raise LLMUnavailableError(text, status) from exc

    return {
        "answer": message_text(state["messages"][-1]).strip(),
        "route": state.get("route", "general"),
        "sources": state.get("sources", []),
        "structured": state.get("structured"),
        "contexts": state.get("contexts", []),
    }


def get_chat_history(session_id: str, mode: str = "auto") -> List[Dict[str, str]]:
    """The saved conversation as [{role, content}, ...] (questions and final answers only)."""
    snapshot = get_graph().get_state(_run_config(session_id, mode))
    history = []
    for message in snapshot.values.get("messages", []):
        if isinstance(message, HumanMessage):
            history.append({"role": "user", "content": message_text(message)})
        elif isinstance(message, AIMessage) and not message.tool_calls and message_text(message).strip():
            history.append({"role": "assistant", "content": message_text(message)})
    return history
