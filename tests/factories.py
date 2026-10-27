"""Builders for valid sample plans and quizzes used in tests."""
from src.schemas.planner_schema import StudyPlan
from src.schemas.quiz_schema import Quiz


def make_plan(days: int = 3, subject: str = "Python", level: str = "Beginner", hours: float = 2.0) -> StudyPlan:
    return StudyPlan(
        subject=subject,
        level=level,
        learning_goals=["Understand the basics", "Write small programs"],
        daily_topics=[
            {"day": d, "topic": f"Topic {d}", "activities": ["Read", "Practise"], "duration_hours": hours}
            for d in range(1, days + 1)
        ],
        total_estimated_hours=999,  # wrong on purpose: the code must recompute it
        recommended_activities=["Keep a notebook"],
        revision_schedule=[{"day": days, "focus": "Review everything"}],
    )


def make_quiz(count: int = 3, topic: str = "Machine Learning", difficulty: str = "Medium") -> Quiz:
    letters = ["A", "B", "C", "D"]
    return Quiz(
        topic=topic,
        difficulty=difficulty,
        questions=[
            {
                "question": f"Sample question number {i + 1}?",
                "options": [f"Option {i}-1", f"Option {i}-2", f"Option {i}-3", f"Option {i}-4"],
                "correct_answer": letters[i % 4],
                "explanation": "Because it is the right one.",
            }
            for i in range(count)
        ],
    )
