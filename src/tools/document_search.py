"""Tool 1: search the student's uploaded documents."""
from typing import List, Tuple

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from src.rag.retriever import retrieve

# Text the LLM sees when the search finds nothing. The graph also checks the
# artifact (an empty list), so the final answer never depends on the LLM obeying.
NO_DOCUMENTS_RESULT = "NO_DOCUMENTS_UPLOADED: the student has not uploaded any documents."
NO_RELEVANT_RESULT = (
    "NO_RELEVANT_PASSAGES_FOUND: the uploaded documents do not contain information about this query."
)


@tool(response_format="content_and_artifact")
def search_documents(query: str, config: RunnableConfig) -> Tuple[str, List[dict]]:
    """Search the student's uploaded documents (PDF/TXT notes) for passages relevant to a query.

    Use a short, specific query. Returns the matching passages with their file name and page number.
    """
    # The session id comes from the server-side config, NOT from the model, so the
    # LLM can never ask for another student's documents.
    session_id = (config or {}).get("configurable", {}).get("session_id", "")
    result = retrieve(session_id, query)

    if not result.has_documents:
        return NO_DOCUMENTS_RESULT, []
    if not result.found_relevant:
        return NO_RELEVANT_RESULT, []

    blocks = [
        f"[{number}] (source: {chunk.source}, page {chunk.page})\n{chunk.text}"
        for number, chunk in enumerate(result.chunks, start=1)
    ]
    return "\n\n".join(blocks), [chunk.to_dict() for chunk in result.chunks]
