"""Pydantic models for the AI Quiz Generator."""
from typing import List, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Difficulty = Literal["Easy", "Medium", "Hard"]
OPTION_LETTERS = ["A", "B", "C", "D"]


class QuizRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=150)
    difficulty: Difficulty = "Medium"
    num_questions: int = Field(default=5, ge=1, le=15)

    @field_validator("topic")
    @classmethod
    def clean_topic(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("topic must not be empty")
        return value

    @field_validator("difficulty", mode="before")
    @classmethod
    def normalise_difficulty(cls, value):
        if isinstance(value, str):
            return value.strip().title()
        return value


class QuizQuestion(BaseModel):
    question: str = Field(min_length=5)
    options: List[str] = Field(min_length=4, max_length=4, description="Exactly four answer options, without A/B/C/D prefixes.")
    correct_answer: Literal["A", "B", "C", "D"] = Field(description="Letter of the correct option.")
    explanation: str = Field(min_length=3, description="One or two sentences explaining why the answer is correct.")

    @field_validator("correct_answer", mode="before")
    @classmethod
    def clean_letter(cls, value):
        # Accept "b", "B.", "B)" etc.
        if isinstance(value, str):
            return value.strip().upper().rstrip(".):")[:1]
        return value

    @model_validator(mode="after")
    def options_are_distinct(self):
        cleaned = [option.strip().lower() for option in self.options]
        if any(not option for option in cleaned):
            raise ValueError("options must not be empty")
        if len(set(cleaned)) != 4:
            raise ValueError("the four options must be different from each other")
        return self

    @property
    def correct_text(self) -> str:
        return self.options[OPTION_LETTERS.index(self.correct_answer)]


class Quiz(BaseModel):
    topic: str
    difficulty: Difficulty
    questions: List[QuizQuestion] = Field(min_length=1)

    @field_validator("difficulty", mode="before")
    @classmethod
    def normalise_difficulty(cls, value):
        if isinstance(value, str):
            return value.strip().title()
        return value

    def to_markdown(self) -> str:
        """Readable text version (questions first, answer key at the end)."""
        lines = [f"## Quiz: {self.topic} ({self.difficulty})", ""]
        for number, item in enumerate(self.questions, start=1):
            lines.append(f"**Q{number}. {item.question}**")
            lines.extend(f"- {letter}. {text}" for letter, text in zip(OPTION_LETTERS, item.options))
            lines.append("")
        lines.append("### Answer key")
        for number, item in enumerate(self.questions, start=1):
            lines.append(f"{number}. **{item.correct_answer}** - {item.explanation}")
        return "\n".join(lines)
