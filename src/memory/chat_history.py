"""Conversation memory = LangGraph checkpointing, keyed by thread id.

How it works
------------
Every time the graph runs, LangGraph saves the conversation state (all messages)
under a `thread_id`. The next message with the same thread id starts from the
saved state, so follow-up questions such as "Explain its types" can be
understood. Different thread ids never see each other's messages.

We use the in-memory checkpointer (InMemorySaver). Its limitations:
- history disappears when the server restarts;
- it lives in one process, so it is not shared between several workers or servers;
- it grows with usage (old conversations are only removed by "Clear chat").
For a real deployment swap it for SqliteSaver or PostgresSaver; the graph code
does not change.
"""
from typing import List

from langgraph.checkpoint.memory import InMemorySaver

# Two separate conversations per session: the general chat and the PDF Q&A tab.
THREAD_NAMESPACES = ("chat", "pdf")

_checkpointer = InMemorySaver()


def get_checkpointer() -> InMemorySaver:
    return _checkpointer


def reset_checkpointer() -> InMemorySaver:
    """Replace the checkpointer with an empty one (used by tests)."""
    global _checkpointer
    _checkpointer = InMemorySaver()
    return _checkpointer


def namespace_for_mode(mode: str) -> str:
    return "pdf" if mode == "pdf" else "chat"


def thread_id_for(session_id: str, mode: str = "auto") -> str:
    """The id under which this session's conversation is stored."""
    return f"{session_id}:{namespace_for_mode(mode)}"


def clear_history(session_id: str, scope: str = "all") -> List[str]:
    """Delete saved conversation(s). scope: 'chat', 'pdf' or 'all'. Returns cleared thread ids."""
    namespaces = THREAD_NAMESPACES if scope == "all" else (scope,)
    cleared = []
    for namespace in namespaces:
        thread_id = f"{session_id}:{namespace}"
        _checkpointer.delete_thread(thread_id)
        cleared.append(thread_id)
    return cleared
