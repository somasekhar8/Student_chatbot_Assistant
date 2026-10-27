"""Phases 2-5 tests: routing, RAG tool loop, hallucination guards, memory, planner and quiz tools."""
import pytest
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from src.llm.fallback import LLMUnavailableError
from src.llm.prompts import NO_DOCUMENTS_MESSAGE, NOT_FOUND_MESSAGE
from src.memory.chat_history import clear_history
from src.rag.document_loader import load_document
from src.rag.text_splitter import split_documents
from src.rag.vector_store import get_index_store
from src.schemas.chat_schema import RouteDecision
from src.schemas.planner_schema import StudyPlan
from src.schemas.quiz_schema import Quiz
from src.workflows.graph import chat, get_chat_history
from tests.factories import make_plan, make_quiz
from tests.fakes import FailingChat, tool_call_message

SESSION = "session-aaaa"


def system_text(messages):
    first = messages[0]
    return first.content if isinstance(first, SystemMessage) else ""


def last_text(messages):
    return messages[-1].content if isinstance(messages[-1].content, str) else ""


def keyword_router(text: str) -> RouteDecision:
    text = text.lower()
    if "pdf" in text or "my notes" in text:
        return RouteDecision(route="rag", reason="refers to documents")
    if "study plan" in text:
        return RouteDecision(route="planner", reason="wants a plan")
    if "questions" in text or "quiz" in text:
        return RouteDecision(route="quiz", reason="wants a quiz")
    if "what is" in text or "explain" in text:
        return RouteDecision(route="general", reason="general question")
    return RouteDecision(route="clarify", reason="unclear")


def scripted(general="General answer.", rag_answer="Machine learning means learning from data (ml.pdf, page 1)."):
    """A fake LLM that behaves like the real workflow would, depending on which prompt it receives."""

    def reply(messages):
        system = system_text(messages)
        if system.startswith("You route"):
            latest = messages[-1].content.split("Latest student message:")[-1]
            return keyword_router(latest)
        if "careful academic assistant" in system:  # RAG agent
            if isinstance(messages[-1], ToolMessage):
                return rag_answer
            return tool_call_message("search_documents", {"query": "What are the types of machine learning?"})
        if "make study plans" in system:
            return tool_call_message("generate_study_plan", {"subject": "Python", "level": "beginner", "hours_per_day": 2, "days": 3})
        if "practise with quizzes" in system:
            return tool_call_message("generate_quiz", {"topic": "Machine Learning", "difficulty": "easy", "num_questions": 3})
        if "senior academic advisor" in system:
            return make_plan(days=3)
        if "experienced teacher" in system:
            return make_quiz(count=3)
        if "unclear" in system:
            return "Could you tell me more about what you need?"
        return general

    return reply


@pytest.fixture
def with_documents(sample_pdf):
    chunks = split_documents(load_document(sample_pdf, "ml.pdf"))
    get_index_store().add_documents(SESSION, chunks, "ml.pdf")


# ---------------------------------------------------------------- routing --
@pytest.mark.parametrize(
    "message,expected",
    [
        ("Explain this uploaded PDF", "rag"),
        ("Create a Python study plan", "planner"),
        ("Generate five ML questions", "quiz"),
        ("What is Artificial Intelligence?", "general"),
        ("hmm help", "clarify"),
    ],
)
def test_router_selects_the_right_workflow(install_llm, with_documents, message, expected):
    install_llm(scripted())
    assert chat(SESSION, message)["route"] == expected


def test_invalid_router_output_goes_to_clarification(install_llm):
    def reply(messages):
        if system_text(messages).startswith("You route"):
            return '{"route": "make_coffee", "reason": "nonsense"}'  # not an allowed route
        return "Could you tell me more?"

    install_llm(reply)
    result = chat(SESSION, "do the thing")
    assert result["route"] == "clarify"
    assert "tell me more" in result["answer"]


def test_pdf_mode_skips_the_router(install_llm, with_documents):
    llm = install_llm(scripted())
    result = chat(SESSION, "What is machine learning?", mode="pdf")
    assert result["route"] == "rag"
    assert not any(system_text(call).startswith("You route") for call in llm.calls)


# ------------------------------------------------------------ RAG + tools --
def test_rag_uses_the_search_tool_and_cites_sources(install_llm, with_documents):
    install_llm(scripted())
    result = chat(SESSION, "Explain this uploaded PDF")
    assert result["route"] == "rag"
    assert "learning from data" in result["answer"]
    assert result["sources"], "answers from documents must carry citations"
    assert result["sources"][0]["document"] == "ml.pdf"
    assert result["sources"][0]["page"] in (1, 2)
    assert result["contexts"]


def test_unanswerable_question_gets_the_fixed_message_even_if_llm_invents(install_llm, with_documents):
    def reply(messages):
        system = system_text(messages)
        if "careful academic assistant" in system:
            if isinstance(messages[-1], ToolMessage):
                return "Brazil won the cup in 2002."  # invented
            return tool_call_message("search_documents", {"query": "football world cup winner"})
        return "x"

    install_llm(reply)
    result = chat(SESSION, "Who won the football world cup?", mode="pdf")
    assert result["answer"] == NOT_FOUND_MESSAGE
    assert result["sources"] == []


def test_no_documents_means_no_llm_call_and_no_guessing(install_llm):
    llm = install_llm(scripted())
    result = chat(SESSION, "What does my PDF say about AI?", mode="pdf")
    assert result["answer"] == NO_DOCUMENTS_MESSAGE
    assert llm.calls == []  # not a single model call


def test_model_that_skips_the_tool_is_forced_to_search(install_llm, with_documents):
    def reply(messages):
        if isinstance(messages[-1], ToolMessage):
            return "Supervised, unsupervised and reinforcement learning (ml.pdf, page 2)."
        return "I already know this: it has three types."  # tries to answer without searching

    install_llm(reply)
    result = chat(SESSION, "What are the types of machine learning?", mode="pdf")
    assert "reinforcement" in result["answer"]
    assert result["sources"]


def test_tool_loop_stops_after_the_round_limit(install_llm, with_documents):
    from src.workflows.state import MAX_TOOL_ROUNDS

    def endless(messages):  # always asks for another search
        return tool_call_message("search_documents", {"query": "machine learning"}, call_id=f"c{len(messages)}")

    install_llm(endless)
    result = chat(SESSION, "Tell me about machine learning", mode="pdf")  # must not raise or hang
    assert result["answer"] == NOT_FOUND_MESSAGE
    history = get_graph_messages()
    tool_results = [m for m in history if isinstance(m, ToolMessage)]
    assert len(tool_results) <= MAX_TOOL_ROUNDS


def get_graph_messages():
    from src.memory.chat_history import thread_id_for
    from src.workflows.graph import get_graph

    config = {"configurable": {"thread_id": thread_id_for(SESSION, "pdf")}}
    return get_graph().get_state(config).values["messages"]


# ----------------------------------------------------------------- memory --
def test_follow_up_questions_see_earlier_messages(install_llm):
    llm = install_llm(scripted(general="Machine learning is learning from data."))
    chat(SESSION, "What is Machine Learning?")
    chat(SESSION, "Explain its types")

    last_prompt = llm.calls[-1]
    texts = [m.content for m in last_prompt if isinstance(m.content, str)]
    assert any("What is Machine Learning?" in t for t in texts)
    assert any("learning from data" in t for t in texts)
    assert [h["role"] for h in get_chat_history(SESSION)] == ["user", "assistant", "user", "assistant"]


def test_sessions_do_not_mix(install_llm):
    llm = install_llm(scripted())
    chat("session-aaaa", "What is secret topic alpha?")
    chat("session-bbbb", "What is topic beta?")
    second_user_prompt = " ".join(m.content for m in llm.calls[-1] if isinstance(m.content, str))
    assert "alpha" not in second_user_prompt
    assert len(get_chat_history("session-bbbb")) == 2


def test_clear_chat_forgets_the_conversation(install_llm):
    llm = install_llm(scripted())
    chat(SESSION, "What is topic gamma?")
    clear_history(SESSION)
    assert get_chat_history(SESSION) == []
    chat(SESSION, "What is topic delta?")
    prompt = " ".join(m.content for m in llm.calls[-1] if isinstance(m.content, str))
    assert "gamma" not in prompt


def test_chat_and_pdf_tabs_have_separate_memory(install_llm, with_documents):
    install_llm(scripted())
    chat(SESSION, "What is a general topic?")
    chat(SESSION, "What does the pdf say?", mode="pdf")
    assert len(get_chat_history(SESSION, "auto")) == 2
    assert len(get_chat_history(SESSION, "pdf")) == 2


# ------------------------------------------------------ planner and quiz --
def test_planner_workflow_runs_the_tool_and_returns_structured_plan(install_llm):
    install_llm(scripted())
    result = chat(SESSION, "Create a Python study plan")
    assert result["route"] == "planner"
    assert result["structured"]["type"] == "study_plan"
    plan = StudyPlan.model_validate(result["structured"]["data"])
    assert len(plan.daily_topics) == 3
    assert plan.total_estimated_hours == 6  # recomputed in code: 3 days x 2 h
    assert "Study plan: Python" in result["answer"]


def test_quiz_workflow_runs_the_tool_and_returns_validated_quiz(install_llm):
    install_llm(scripted())
    result = chat(SESSION, "Generate five ML questions")
    assert result["route"] == "quiz"
    quiz = Quiz.model_validate(result["structured"]["data"])
    assert len(quiz.questions) == 3
    assert all(len(q.options) == 4 for q in quiz.questions)


# ---------------------------------------------------------------- errors --
def test_failure_is_friendly_and_leaves_no_half_saved_turn(monkeypatch):
    model = FailingChat(error=RuntimeError("Error code: 429 - rate limit exceeded, key=gsk_SECRET123"))
    monkeypatch.setattr("src.llm.fallback.available_models", lambda temperature=0.2: [model])

    with pytest.raises(LLMUnavailableError) as info:
        chat(SESSION, "What is AI?")
    assert info.value.status_code == 429
    assert "gsk_SECRET123" not in info.value.message
    assert get_chat_history(SESSION) == []


def test_missing_api_keys_give_a_clear_message():
    with pytest.raises(LLMUnavailableError) as info:
        chat(SESSION, "What is AI?")
    assert "GROQ_API_KEY" in info.value.message
