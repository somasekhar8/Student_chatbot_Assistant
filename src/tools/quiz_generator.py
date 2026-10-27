"""Tool 3: generate a validated multiple-choice quiz (shared by the tool and FastAPI)."""
from typing import Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from src.llm.fallback import invoke_structured
from src.llm.prompts import QUIZ_PROMPT
from src.schemas.quiz_schema import Quiz, QuizRequest


def _check_quiz(quiz: Quiz, request: QuizRequest) -> None:
    if len(quiz.questions) < request.num_questions:
        raise ValueError(f"Produce exactly {request.num_questions} questions; you produced {len(quiz.questions)}.")


def build_quiz(request: QuizRequest) -> Quiz:
    prompt = QUIZ_PROMPT.format(
        topic=request.topic, difficulty=request.difficulty, count=request.num_questions
    )
    quiz = invoke_structured(
        Quiz,
        [SystemMessage(content=prompt), HumanMessage(content=f"Write the quiz about {request.topic}.")],
        temperature=0.5,
        validate=lambda result: _check_quiz(result, request),
    )
    quiz.questions = quiz.questions[: request.num_questions]  # drop extras if the model over-produced
    quiz.topic = request.topic
    quiz.difficulty = request.difficulty
    return quiz


@tool(response_format="content_and_artifact")
def generate_quiz(topic: str, difficulty: str = "Medium", num_questions: int = 5) -> Tuple[str, dict]:
    """Create a multiple-choice quiz with four options per question, the correct answer and an explanation.

    Args:
        topic: The subject of the quiz, e.g. "Machine Learning basics".
        difficulty: "Easy", "Medium" or "Hard".
        num_questions: How many questions (1 to 15).
    """
    request = QuizRequest(topic=topic, difficulty=difficulty, num_questions=num_questions)
    quiz = build_quiz(request)
    return f"Quiz created on {quiz.topic}: {len(quiz.questions)} questions.", quiz.model_dump()
