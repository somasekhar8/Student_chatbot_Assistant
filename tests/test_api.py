"""API tests with a fake LLM: no network, no API keys."""
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from src.llm.fallback import LLMUnavailableError
from src.llm.prompts import NOT_FOUND_MESSAGE
from tests.factories import make_plan, make_quiz
from tests.fakes import tool_call_message

client = TestClient(app)
SID = "student-session-1"
OTHER = "student-session-2"


def upload(session_id, data, name="ml.pdf"):
    return client.post("/upload", data={"session_id": session_id}, files={"file": (name, data, "application/pdf")})


def test_health_never_returns_keys(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_secret_value")
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["providers_configured"] == ["groq"]
    assert "gsk_secret_value" not in str(body)


def test_upload_and_documents(sample_pdf):
    response = upload(SID, sample_pdf)
    assert response.status_code == 200
    body = response.json()
    assert body["pages"] == 2 and body["chunks"] >= 2 and body["documents"] == ["ml.pdf"]
    assert upload(SID, sample_pdf).json()["already_indexed"] is True
    assert client.get(f"/documents/{SID}").json()["documents"] == ["ml.pdf"]
    assert client.get(f"/documents/{OTHER}").json()["documents"] == []  # isolation


def test_upload_rejects_bad_files():
    assert upload(SID, b"not a pdf").status_code == 400
    assert upload(SID, b"hello", name="x.exe").status_code == 400
    assert upload("bad id!", b"x").status_code == 422


def test_chat_general_follow_up_memory(install_llm):
    from tests.fakes import FakeChat  # noqa: F401
    from src.schemas.chat_schema import RouteDecision

    def reply(messages):
        text = " ".join(str(m.content) for m in messages)
        if "route" in text.lower() and "clarify" in text.lower():
            return RouteDecision(route="general", reason="general question")
        return f"saw {sum(1 for m in messages if m.type == 'human')} questions"

    install_llm(reply)
    first = client.post("/chat", json={"session_id": SID, "message": "What is Machine Learning?"}).json()
    second = client.post("/chat", json={"session_id": SID, "message": "Explain its types."}).json()
    assert first["route"] == "general" and "saw 1" in first["answer"]
    assert "saw 2" in second["answer"]  # the follow-up sees the earlier question
    other = client.post("/chat", json={"session_id": OTHER, "message": "Hi there"}).json()
    assert "saw 1" in other["answer"]  # other session has separate memory

    assert len(client.get(f"/chat/{SID}/history").json()["messages"]) == 4
    assert client.delete(f"/chat/{SID}").status_code == 200
    assert client.get(f"/chat/{SID}/history").json()["messages"] == []


def test_pdf_mode_answers_with_sources(install_llm, sample_pdf):
    upload(SID, sample_pdf)

    def reply(messages):
        if any(m.type == "tool" for m in messages):
            return "Supervised, unsupervised and reinforcement learning."
        return tool_call_message("search_documents", {"query": "types of machine learning"})

    install_llm(reply)
    body = client.post(
        "/chat", json={"session_id": SID, "message": "What are the types of machine learning?", "mode": "pdf"}
    ).json()
    assert body["route"] == "rag"
    assert body["sources"] and body["sources"][0]["document"] == "ml.pdf" and body["sources"][0]["page"] >= 1


def test_pdf_mode_without_relevant_context_is_not_answered_from_memory(install_llm, sample_pdf):
    upload(SID, sample_pdf)
    install_llm(lambda m: "The capital of France is Paris." if any(x.type == "tool" for x in m) else tool_call_message("search_documents", {"query": "capital of France"}))
    body = client.post("/chat", json={"session_id": SID, "message": "Capital of France?", "mode": "pdf"}).json()
    assert body["answer"] == NOT_FOUND_MESSAGE and body["sources"] == []


def test_study_plan_and_quiz(install_llm):
    install_llm(lambda m: make_plan(days=3) if "plan" in str(m[0].content).lower() else make_quiz(count=2))
    plan = client.post("/study-plan", json={"subject": "Python", "level": "beginner", "hours_per_day": 2, "days": 3})
    assert plan.status_code == 200
    body = plan.json()
    assert body["total_estimated_hours"] == 6 and len(body["daily_topics"]) == 3  # recomputed in code

    install_llm(lambda m: make_quiz(count=2))
    quiz = client.post("/quiz", json={"topic": "ML", "difficulty": "easy", "num_questions": 2})
    assert quiz.status_code == 200 and len(quiz.json()["questions"]) == 2


def test_validation_errors():
    assert client.post("/study-plan", json={"subject": "", "days": 3}).status_code == 422
    assert client.post("/quiz", json={"topic": "ML", "num_questions": 99}).status_code == 422
    assert client.post("/chat", json={"session_id": SID, "message": "   "}).status_code == 422


def test_llm_errors_become_friendly_messages(monkeypatch):
    def boom(*_args, **_kwargs):
        raise LLMUnavailableError("The AI service is busy or the free-tier limit was reached.", 429)

    monkeypatch.setattr("backend.routes.chat.chat", boom)
    response = client.post("/chat", json={"session_id": SID, "message": "hello"})
    assert response.status_code == 429 and "busy" in response.json()["detail"]


def test_no_api_key_gives_clear_message():
    response = client.post("/quiz", json={"topic": "ML"})
    assert response.status_code == 503 and "GROQ_API_KEY" in response.json()["detail"]
