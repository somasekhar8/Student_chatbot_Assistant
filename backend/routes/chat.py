"""POST /chat plus history and clear endpoints."""
from typing import Dict, List

from fastapi import APIRouter

from src.memory.chat_history import clear_history
from src.rag.vector_store import get_index_store
from src.schemas.chat_schema import ChatRequest, ChatResponse, ClearResponse, ClearScope, SESSION_ID_PATTERN
from src.workflows.graph import chat, get_chat_history

from fastapi import HTTPException
import re

router = APIRouter(tags=["chat"])


def _check_session(session_id: str) -> str:
    if not re.match(SESSION_ID_PATTERN, session_id or ""):
        raise HTTPException(status_code=422, detail="session_id must be 8-64 letters, digits, '-' or '_'.")
    return session_id


@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Send one message through the LangGraph workflow (router -> agent -> tools)."""
    result = chat(request.session_id, request.message, request.mode)
    return ChatResponse(
        session_id=request.session_id,
        answer=result["answer"],
        route=result["route"],
        sources=result["sources"],
        structured=result["structured"],
    )


@router.get("/chat/{session_id}/history")
def history(session_id: str, mode: str = "auto") -> Dict[str, List[Dict[str, str]]]:
    session_id = _check_session(session_id)
    return {"messages": get_chat_history(session_id, "pdf" if mode == "pdf" else "auto")}


@router.delete("/chat/{session_id}", response_model=ClearResponse)
def clear_chat(session_id: str, scope: ClearScope = "chat", delete_documents: bool = False) -> ClearResponse:
    """Clear Chat button: forget the conversation (and optionally the uploaded documents)."""
    session_id = _check_session(session_id)
    clear_history(session_id, scope)
    if delete_documents:
        get_index_store().clear(session_id)
    return ClearResponse(session_id=session_id, cleared=scope)
