"""Pydantic models for the AI Study Planner."""
from typing import List, Literal

from pydantic import BaseModel, Field, field_validator

Level = Literal["Beginner", "Intermediate", "Advanced"]


class StudyPlanRequest(BaseModel):
    """What the student enters in the form (or what the LLM extracts from chat)."""

    subject: str = Field(min_length=1, max_length=100)
    level: Level = "Beginner"
    hours_per_day: float = Field(default=2.0, gt=0, le=16)
    days: int = Field(default=7, ge=1, le=60)

    @field_validator("subject")
    @classmethod
    def clean_subject(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("subject must not be empty")
        return value

    @field_validator("level", mode="before")
    @classmethod
    def normalise_level(cls, value):
        # LLMs often send "beginner" or "BEGINNER"; accept any capitalisation.
        if isinstance(value, str):
            return value.strip().title()
        return value


class DailyTopic(BaseModel):
    day: int = Field(ge=1, description="Day number, starting at 1.")
    topic: str = Field(description="Main topic to study on this day.")
    activities: List[str] = Field(min_length=1, description="2-3 concrete practice activities for the day.")
    duration_hours: float = Field(gt=0, description="Hours planned for this day.")


class RevisionItem(BaseModel):
    day: int = Field(ge=1, description="Day on which this revision happens.")
    focus: str = Field(description="What to revise on that day.")


class StudyPlan(BaseModel):
    """The validated study plan returned by the LLM."""

    subject: str
    level: Level
    learning_goals: List[str] = Field(min_length=1, description="3-5 clear learning goals.")
    daily_topics: List[DailyTopic] = Field(min_length=1)
    total_estimated_hours: float = Field(gt=0, description="Total study time in hours.")
    recommended_activities: List[str] = Field(min_length=1, description="General activities/resources that help.")
    revision_schedule: List[RevisionItem] = Field(min_length=1)

    @field_validator("level", mode="before")
    @classmethod
    def normalise_level(cls, value):
        if isinstance(value, str):
            return value.strip().title()
        return value

    def to_markdown(self) -> str:
        """Readable text version, used in the chat window."""
        lines = [
            f"## Study plan: {self.subject}",
            f"**Level:** {self.level} | **Total time:** {self.total_estimated_hours:g} hours "
            f"over {len(self.daily_topics)} days",
            "",
            "### Learning goals",
            *[f"- {goal}" for goal in self.learning_goals],
            "",
            "### Daily topics",
        ]
        for item in self.daily_topics:
            lines.append(f"**Day {item.day} - {item.topic}** ({item.duration_hours:g} h)")
            lines.extend(f"- {activity}" for activity in item.activities)
            lines.append("")
        lines.append("### Recommended activities")
        lines.extend(f"- {activity}" for activity in self.recommended_activities)
        lines.append("")
        lines.append("### Revision schedule")
        lines.extend(f"- Day {item.day}: {item.focus}" for item in self.revision_schedule)
        return "\n".join(lines)
