"""Pydantic models for chat, routing, uploads and health checks."""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

# Session ids are random strings made by the frontend. We restrict the
# characters so they are safe to use as dictionary keys and file-free ids.
SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]{8,64}$"

RouteName = Literal["rag", "planner", "quiz", "general", "clarify"]
ChatMode = Literal["auto", "pdf"]
ClearScope = Literal["chat", "pdf", "all"]


class RouteDecision(BaseModel):
    """Structured output of the router. `route` can only be one of five values."""

    route: RouteName = Field(description="The workflow that should handle the message.")
    reason: str = Field(default="", max_length=300, description="One short sentence explaining the choice.")


class ChatRequest(BaseModel):
    session_id: str = Field(pattern=SESSION_ID_PATTERN)
    message: str = Field(min_length=1, max_length=4000)
    mode: ChatMode = Field(
        default="auto",
        description="'auto' lets the router decide. 'pdf' forces answers from uploaded documents only.",
    )

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message must not be empty")
        return value


class SourceInfo(BaseModel):
    document: str
    page: int
    score: float = Field(description="Cosine similarity between the question and the passage (0-1).")
    snippet: str = ""


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    route: RouteName
    sources: List[SourceInfo] = Field(default_factory=list)
    # For study plans and quizzes the validated object is returned here so the
    # UI can draw it nicely: {"type": "study_plan" | "quiz", "data": {...}}
    structured: Optional[Dict[str, Any]] = None


class UploadResponse(BaseModel):
    session_id: str
    filename: str
    pages: int
    chunks: int
    already_indexed: bool = False
    documents: List[str]


class DocumentsResponse(BaseModel):
    session_id: str
    documents: List[str]


class ClearResponse(BaseModel):
    session_id: str
    cleared: str


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    providers_configured: List[str] = Field(
        description="Names of providers that have an API key set (never the keys themselves)."
    )
