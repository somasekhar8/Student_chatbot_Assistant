"""Phase 5 + LLM module tests: structured plans, quizzes, fallback, rate limits, key safety."""
import pytest
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from src.llm import model as model_module
from src.llm.fallback import LLMUnavailableError, friendly_error, get_chat_model, invoke_structured
from src.schemas.planner_schema import StudyPlan, StudyPlanRequest
from src.schemas.quiz_schema import QuizQuestion, QuizRequest
from src.tools.quiz_generator import build_quiz
from src.tools.study_planner import build_study_plan
from tests.factories import make_plan, make_quiz
from tests.fakes import FailingChat, FakeChat


# ----------------------------------------------------------- study plan --
def test_request_normalises_level_and_validates_ranges():
    assert StudyPlanRequest(subject=" Python ", level="beginner").subject == "Python"
    assert StudyPlanRequest(subject="x", level="ADVANCED").level == "Advanced"
    for bad in ({"hours_per_day": 0}, {"hours_per_day": 20}, {"days": 0}, {"days": 100}, {"level": "Expert"}):
        with pytest.raises(ValidationError):
            StudyPlanRequest(subject="Python", **bad)


def test_build_study_plan_returns_all_required_parts(install_llm):
    install_llm(lambda messages: make_plan(days=4, hours=1.5))
    plan = build_study_plan(StudyPlanRequest(subject="Python", level="Beginner", hours_per_day=1.5, days=4))
    assert plan.subject == "Python"
    assert plan.learning_goals and plan.recommended_activities and plan.revision_schedule
    assert [d.day for d in plan.daily_topics] == [1, 2, 3, 4]
    assert plan.total_estimated_hours == 6.0  # computed by code (4 x 1.5), the LLM said 999


def test_plan_with_wrong_number_of_days_is_retried_then_accepted(install_llm):
    answers = iter([make_plan(days=2), make_plan(days=5)])  # first answer has too few days
    llm = install_llm(lambda messages: next(answers))
    plan = build_study_plan(StudyPlanRequest(subject="Python", days=5))
    assert len(plan.daily_topics) == 5
    # The retry told the model what was wrong.
    assert "previous answer was invalid" in str(llm.calls[-1][-1].content)


def test_plan_that_stays_invalid_raises_friendly_error(install_llm):
    install_llm(lambda messages: make_plan(days=2))
    with pytest.raises(LLMUnavailableError):
        build_study_plan(StudyPlanRequest(subject="Python", days=5))


def test_plan_markdown_is_readable():
    text = make_plan(days=2).to_markdown()
    assert "Learning goals" in text and "Day 1" in text and "Revision schedule" in text


# ----------------------------------------------------------------- quiz --
def test_quiz_question_requires_four_distinct_options():
    base = {"question": "What is 2 + 2?", "correct_answer": "A", "explanation": "Arithmetic."}
    QuizQuestion(options=["4", "3", "5", "6"], **base)
    with pytest.raises(ValidationError):
        QuizQuestion(options=["4", "3", "5"], **base)
    with pytest.raises(ValidationError):
        QuizQuestion(options=["4", "4", "5", "6"], **base)
    with pytest.raises(ValidationError):
        QuizQuestion(options=["4", "3", "5", "6"], **{**base, "correct_answer": "E"})


def test_quiz_answer_letter_is_cleaned_and_mapped_to_text():
    question = QuizQuestion(
        question="Pick the second one", options=["a1", "b2", "c3", "d4"], correct_answer="b)", explanation="It is b."
    )
    assert question.correct_answer == "B"
    assert question.correct_text == "b2"


def test_build_quiz_trims_extra_questions(install_llm):
    install_llm(lambda messages: make_quiz(count=6))
    quiz = build_quiz(QuizRequest(topic="Machine Learning", difficulty="easy", num_questions=4))
    assert len(quiz.questions) == 4
    assert quiz.difficulty == "Easy"


def test_build_quiz_retries_when_too_few_questions(install_llm):
    answers = iter([make_quiz(count=2), make_quiz(count=4)])
    install_llm(lambda messages: next(answers))
    assert len(build_quiz(QuizRequest(topic="ML", num_questions=4)).questions) == 4


def test_quiz_markdown_has_answer_key():
    assert "Answer key" in make_quiz(count=2).to_markdown()


# ----------------------------------------------------- fallback + errors --
def test_secondary_model_is_used_when_primary_fails(monkeypatch):
    primary = FailingChat(error=RuntimeError("429 rate limit"))
    secondary = FakeChat(reply_fn=lambda messages: "answer from backup")
    monkeypatch.setattr("src.llm.fallback.available_models", lambda temperature=0.2: [primary, secondary])
    assert get_chat_model().invoke("hi").content == "answer from backup"


def test_structured_output_also_falls_back(monkeypatch):
    primary = FailingChat(error=RuntimeError("503 unavailable"))
    secondary = FakeChat(reply_fn=lambda messages: make_plan(days=2))
    monkeypatch.setattr("src.llm.fallback.available_models", lambda temperature=0.2: [primary, secondary])
    result = invoke_structured(StudyPlan, [HumanMessage(content="make a plan")])
    assert isinstance(result, StudyPlan)


@pytest.mark.parametrize(
    "error,status",
    [
        (RuntimeError("Error code: 429 Too Many Requests"), 429),
        (RuntimeError("RESOURCE_EXHAUSTED: quota exceeded"), 429),
        (RuntimeError("401 invalid api key provided"), 503),
        (TimeoutError("request timed out"), 504),
        (RuntimeError("model not found"), 503),
        (RuntimeError("something odd"), 502),
    ],
)
def test_errors_are_translated_without_leaking_details(error, status):
    message, code = friendly_error(error)
    assert code == status
    assert str(error) not in message


# ---------------------------------------------------------- configuration --
def test_provider_order_follows_environment(monkeypatch):
    assert model_module.provider_order() == ["groq", "gemini"]
    monkeypatch.setenv("PRIMARY_LLM_PROVIDER", "gemini")
    assert model_module.provider_order() == ["gemini", "groq"]
    monkeypatch.setenv("PRIMARY_LLM_PROVIDER", "nonsense")
    assert model_module.primary_provider() == "groq"


def test_placeholder_keys_count_as_missing(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "Groq_api_key")
    monkeypatch.setenv("GEMINI_API_KEY", "YOUR GEMINI API KEY")
    assert model_module.configured_providers() == []
    assert model_module.available_models() == []


def test_real_provider_classes_can_be_built_with_dummy_keys(monkeypatch):
    """Construction only (no network): proves the Groq and Gemini integrations are wired correctly."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_dummy_key_for_tests")
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-gemini-key-for-tests")  # alias for GOOGLE_API_KEY
    monkeypatch.setenv("GROQ_MODEL", "my-groq-model")
    models = model_module.available_models()
    assert [type(m).__name__ for m in models] == ["ChatGroq", "ChatGoogleGenerativeAI"]
    assert models[0].model_name == "my-groq-model"


def test_keys_never_appear_in_repr_or_health(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_super_secret_value")
    assert "gsk_super_secret_value" not in repr(model_module.build_model("groq"))
    assert model_module.configured_providers() == ["groq"]
