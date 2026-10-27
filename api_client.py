"""How the Streamlit app talks to the backend.

Two interchangeable clients with the same methods:

* HttpClient  - calls the FastAPI server at API_URL (the "real" architecture).
* LocalClient - calls the same Python functions in-process. Used when API_URL is
                not set, so the app runs on Streamlit Community Cloud as a single
                service without hosting FastAPI separately.

Both raise ApiError with a message that is safe to show to students.
"""
import os
from typing import Any, Dict, List, Optional

import requests


class ApiError(Exception):
    """A problem to show to the student (never contains API keys)."""


class HttpClient:
    def __init__(self, base_url: str, session_id: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs) -> Any:
        try:
            response = requests.request(method, f"{self.base_url}{path}", timeout=self.timeout, **kwargs)
        except requests.ConnectionError as exc:
            raise ApiError(f"Cannot reach the backend at {self.base_url}. Is it running?") from exc
        except requests.Timeout as exc:
            raise ApiError("The backend took too long to respond. Please try again.") from exc
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", "")
            except ValueError:
                detail = ""
            if isinstance(detail, list):  # FastAPI validation errors
                detail = "; ".join(str(item.get("msg", "")) for item in detail)
            raise ApiError(detail or f"The backend returned an error ({response.status_code}).")
        return response.json()

    def health(self) -> Dict:
        return self._request("GET", "/health")

    def upload(self, data: bytes, filename: str) -> Dict:
        return self._request("POST", "/upload", data={"session_id": self.session_id}, files={"file": (filename, data)})

    def documents(self) -> List[str]:
        return self._request("GET", f"/documents/{self.session_id}")["documents"]

    def chat(self, message: str, mode: str = "auto") -> Dict:
        return self._request("POST", "/chat", json={"session_id": self.session_id, "message": message, "mode": mode})

    def clear(self, scope: str = "chat", delete_documents: bool = False) -> None:
        self._request(
            "DELETE", f"/chat/{self.session_id}", params={"scope": scope, "delete_documents": delete_documents}
        )

    def study_plan(self, payload: Dict) -> Dict:
        return self._request("POST", "/study-plan", json=payload)

    def quiz(self, payload: Dict) -> Dict:
        return self._request("POST", "/quiz", json=payload)


class LocalClient:
    """Runs the backend code inside the Streamlit process."""

    mode_name = "embedded"

    def __init__(self, session_id: str):
        self.session_id = session_id

    @staticmethod
    def _guard(function, *args, **kwargs):
        from pydantic import ValidationError

        from src.llm.fallback import LLMUnavailableError
        from src.rag.document_loader import DocumentError

        try:
            return function(*args, **kwargs)
        except (LLMUnavailableError, DocumentError) as exc:
            raise ApiError(getattr(exc, "message", None) or str(exc)) from exc
        except ValidationError as exc:
            raise ApiError("; ".join(error["msg"] for error in exc.errors())) from exc

    def health(self) -> Dict:
        from src.llm.model import configured_providers

        return {"status": "ok", "providers_configured": configured_providers()}

    def upload(self, data: bytes, filename: str) -> Dict:
        from src.rag.document_loader import load_document
        from src.rag.text_splitter import split_documents
        from src.rag.vector_store import get_index_store

        def run():
            pages = load_document(data, filename)
            name = pages[0].metadata["source"]
            chunks = split_documents(pages)
            added = get_index_store().add_documents(self.session_id, chunks, name)
            return {
                "filename": name,
                "pages": len(pages),
                "chunks": len(chunks),
                "already_indexed": not added,
                "documents": get_index_store().list_files(self.session_id),
            }

        return self._guard(run)

    def documents(self) -> List[str]:
        from src.rag.vector_store import get_index_store

        return get_index_store().list_files(self.session_id)

    def chat(self, message: str, mode: str = "auto") -> Dict:
        from src.workflows.graph import chat

        return self._guard(chat, self.session_id, message, mode)

    def clear(self, scope: str = "chat", delete_documents: bool = False) -> None:
        from src.memory.chat_history import clear_history
        from src.rag.vector_store import get_index_store

        clear_history(self.session_id, scope)
        if delete_documents:
            get_index_store().clear(self.session_id)

    def study_plan(self, payload: Dict) -> Dict:
        from src.schemas.planner_schema import StudyPlanRequest
        from src.tools.study_planner import build_study_plan

        return self._guard(lambda: build_study_plan(StudyPlanRequest(**payload)).model_dump())

    def quiz(self, payload: Dict) -> Dict:
        from src.schemas.quiz_schema import QuizRequest
        from src.tools.quiz_generator import build_quiz

        return self._guard(lambda: build_quiz(QuizRequest(**payload)).model_dump())


def get_api_url() -> Optional[str]:
    """API_URL from the environment or Streamlit secrets; None means 'run embedded'."""
    url = os.getenv("API_URL", "").strip()
    if url:
        return url
    try:
        import streamlit as st

        return str(st.secrets.get("API_URL", "")).strip() or None
    except Exception:
        return None


def make_client(session_id: str):
    url = get_api_url()
    return HttpClient(url, session_id) if url else LocalClient(session_id)
