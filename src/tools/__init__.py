from src.tools.document_search import search_documents
from src.tools.quiz_generator import generate_quiz
from src.tools.study_planner import generate_study_plan

ALL_TOOLS = [search_documents, generate_study_plan, generate_quiz]

__all__ = ["search_documents", "generate_study_plan", "generate_quiz", "ALL_TOOLS"]
