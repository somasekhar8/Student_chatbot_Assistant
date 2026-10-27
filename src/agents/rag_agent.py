"""RAG Agent: an LLM that decides what to search for, calls `search_documents`, then answers.

Loop (see graph.py):  rag_agent -> tools -> rag_agent -> finalize
The model writes the search query (so follow-ups like "explain its types" become
"types of machine learning"). The retrieval itself is done by the tool.
"""
import uuid

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from src.llm.fallback import get_chat_model, safe_invoke
from src.llm.prompts import NO_DOCUMENTS_MESSAGE, NOT_FOUND_MESSAGE, RAG_AGENT_PROMPT
from src.tools.document_search import search_documents
from src.workflows.state import MAX_TOOL_ROUNDS, AgentState, build_history, last_user_text, tool_rounds


def rag_agent_node(state: AgentState, config: RunnableConfig) -> dict:
    messages = state["messages"]

    # Nothing uploaded: answer immediately, no LLM call and no guessing.
    if not state.get("has_documents"):
        return {"messages": [AIMessage(content=NO_DOCUMENTS_MESSAGE)]}

    rounds = tool_rounds(messages)
    system = SystemMessage(content=RAG_AGENT_PROMPT.format(not_found=NOT_FOUND_MESSAGE))

    if rounds >= MAX_TOOL_ROUNDS:
        # Stopping condition: the tool budget is used up, so no tools are offered
        # and the model has to write its answer from what it already found.
        model = get_chat_model(temperature=0.1)
    else:
        model = get_chat_model(temperature=0.1, tools=[search_documents])

    reply = safe_invoke(model, [system] + build_history(messages))

    if rounds == 0 and not reply.tool_calls:
        # Hallucination guard: the first step must be a search. If the model tried to
        # answer from memory (or ignored the tool), run the search ourselves.
        reply = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_documents",
                    "args": {"query": last_user_text(messages)},
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "tool_call",
                }
            ],
        )
    return {"messages": [reply]}
