"""POST /quiz - validated multiple-choice quiz."""
from fastapi import APIRouter

from src.schemas.quiz_schema import Quiz, QuizRequest
from src.tools.quiz_generator import build_quiz

router = APIRouter(tags=["quiz"])


@router.post("/quiz", response_model=Quiz)
def create_quiz(request: QuizRequest) -> Quiz:
    return build_quiz(request)
