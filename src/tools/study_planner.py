"""Tool 2: generate a structured study plan.

`build_study_plan` holds the real logic. It is used by BOTH the LangChain tool
(when the LLM calls it from chat) and the FastAPI /study-plan endpoint, so the
code exists only once.
"""
from typing import Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from src.llm.fallback import invoke_structured
from src.llm.prompts import STUDY_PLAN_PROMPT
from src.schemas.planner_schema import StudyPlan, StudyPlanRequest


def _check_plan(plan: StudyPlan, request: StudyPlanRequest) -> None:
    """Rules the schema cannot express. A ValueError makes the LLM try again."""
    days = sorted(item.day for item in plan.daily_topics)
    if days != list(range(1, request.days + 1)):
        raise ValueError(f"daily_topics must contain exactly one entry for each day 1 to {request.days}.")
    if any(item.day > request.days for item in plan.revision_schedule):
        raise ValueError(f"revision_schedule days must be between 1 and {request.days}.")


def build_study_plan(request: StudyPlanRequest) -> StudyPlan:
    prompt = STUDY_PLAN_PROMPT.format(
        subject=request.subject,
        level=request.level,
        hours=f"{request.hours_per_day:g}",
        days=request.days,
    )
    plan = invoke_structured(
        StudyPlan,
        [SystemMessage(content=prompt), HumanMessage(content=f"Create the study plan for {request.subject}.")],
        temperature=0.3,
        validate=lambda result: _check_plan(result, request),
    )
    # Do the arithmetic in code instead of trusting the LLM.
    plan.daily_topics.sort(key=lambda item: item.day)
    plan.revision_schedule.sort(key=lambda item: item.day)
    plan.total_estimated_hours = round(sum(item.duration_hours for item in plan.daily_topics), 2)
    plan.subject = request.subject
    plan.level = request.level
    return plan


@tool(response_format="content_and_artifact")
def generate_study_plan(
    subject: str, level: str = "Beginner", hours_per_day: float = 2.0, days: int = 7
) -> Tuple[str, dict]:
    """Create a personalised study plan.

    Args:
        subject: What the student wants to learn, e.g. "Python" or "Machine Learning".
        level: "Beginner", "Intermediate" or "Advanced".
        hours_per_day: Hours the student can study each day (0.5 to 16).
        days: Number of study days (1 to 60).
    """
    request = StudyPlanRequest(subject=subject, level=level, hours_per_day=hours_per_day, days=days)
    plan = build_study_plan(request)
    summary = f"Study plan created for {plan.subject}: {len(plan.daily_topics)} days, {plan.total_estimated_hours:g} hours."
    return summary, plan.model_dump()
