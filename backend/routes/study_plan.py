"""POST /study-plan - validated study plan from the form."""
from fastapi import APIRouter

from src.schemas.planner_schema import StudyPlan, StudyPlanRequest
from src.tools.study_planner import build_study_plan

router = APIRouter(tags=["study planner"])


@router.post("/study-plan", response_model=StudyPlan)
def create_study_plan(request: StudyPlanRequest) -> StudyPlan:
    return build_study_plan(request)
