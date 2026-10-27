"""Offline evaluation of the RAG pipeline with an LLM acting as judge.

This module is separate from the chatbot: nothing here runs during normal chat,
so students never trigger extra API calls. Run it yourself when you want to
measure quality:

    uv run python -m src.evaluation.rag_evaluator
    uv run python -m src.evaluation.rag_evaluator --limit 3 --delay 2

What is measured (each on a 1-5 scale, see JUDGE_PROMPT for the rubric)
-----------------------------------------------------------------------
1. Retrieval relevance - do the retrieved passages contain what is needed?
2. Faithfulness        - is the answer supported by the retrieved passages?
3. Answer correctness  - does the answer match the reference answer?

Questions marked `answerable: false` are not covered by the document. For them
the correct behaviour is to say "I could not find sufficient information...", and
that is checked in code (no LLM judge needed).

All numbers are calculated from real runs. Nothing is hardcoded.
"""
import argparse
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.llm.fallback import LLMUnavailableError, invoke_structured
from src.llm.prompts import JUDGE_PROMPT, NOT_FOUND_MESSAGE
from src.memory.chat_history import clear_history
from src.rag.document_loader import load_document
from src.rag.text_splitter import split_documents
from src.rag.vector_store import get_index_store
from src.workflows.graph import chat

logger = logging.getLogger(__name__)

DEFAULT_DATASET = Path(__file__).parent / "test_dataset.json"
DEFAULT_OUTPUT = Path("evaluation_results.json")

LIMITATIONS = [
    "The judge is an LLM, so scores are opinions: they can change between runs, even at temperature 0.",
    "LLM judges tend to favour longer, more confident answers and answers written in their own style.",
    "If the judge and the answering model are the same model family, scores can be too generous (self-preference).",
    "Faithfulness is checked only against the retrieved passages. It cannot tell whether the passages were the best ones in the document.",
    "Correctness depends on the quality and wording of the reference answers; a correct answer phrased differently may lose points.",
    "12 questions about one short document is a smoke test, not a benchmark. Averages move a lot when one question changes.",
    "Free-tier rate limits can make some questions fail; those are reported as errors and left out of the averages.",
]


# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #
class EvalQuestion(BaseModel):
    id: int
    question: str
    reference_answer: str
    answerable: bool = True


class EvalDataset(BaseModel):
    description: str = ""
    documents: List[str] = Field(min_length=1)
    questions: List[EvalQuestion] = Field(min_length=10)  # the project requires at least 10


class JudgeScores(BaseModel):
    """Structured rubric returned by the judge LLM."""

    retrieval_relevance: int = Field(ge=1, le=5, description="1-5: how useful the retrieved passages are for the question.")
    faithfulness: int = Field(ge=1, le=5, description="1-5: how well the answer is supported by the retrieved passages.")
    correctness: int = Field(ge=1, le=5, description="1-5: how well the answer matches the reference answer.")
    reasoning: str = Field(default="", description="Two short sentences explaining the scores.")


def load_dataset(path: Optional[Path] = None) -> EvalDataset:
    path = Path(path or DEFAULT_DATASET)
    return EvalDataset.model_validate(json.loads(path.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- #
# Judging
# --------------------------------------------------------------------------- #
def judge_answer(question: str, contexts: List[str], answer: str, reference: str) -> JudgeScores:
    """Ask the judge LLM to score one answer with the rubric."""
    context_text = "\n\n".join(f"[{i}] {text[:1200]}" for i, text in enumerate(contexts, 1)) or "(no passages were retrieved)"
    prompt = JUDGE_PROMPT.format(question=question, contexts=context_text, answer=answer, reference=reference)
    return invoke_structured(
        JudgeScores,
        [SystemMessage(content=prompt), HumanMessage(content="Grade the answer now.")],
        temperature=0,
    )


def evaluate_question(item: EvalQuestion, session_id: str) -> Dict:
    """Run one question through the real chatbot pipeline and score it."""
    row = {
        "id": item.id,
        "question": item.question,
        "answerable": item.answerable,
        "answer": "",
        "retrieval_relevance": None,
        "faithfulness": None,
        "correctness": None,
        "notes": "",
    }
    try:
        result = chat(session_id, item.question, mode="pdf")  # same code path students use
        clear_history(session_id)  # every question starts a fresh conversation
        answer = result["answer"]
        row["answer"] = answer
        refused = NOT_FOUND_MESSAGE.lower() in answer.lower()

        if not item.answerable:
            # Checked in code: the system must refuse instead of inventing something.
            row["correctness"] = 5 if refused else 1
            row["notes"] = "correctly refused" if refused else "answered an unanswerable question"
            if refused:
                row["faithfulness"] = 5  # no claims were made
            else:
                scores = judge_answer(item.question, result["contexts"], answer, item.reference_answer)
                row["faithfulness"] = scores.faithfulness
            return row

        scores = judge_answer(item.question, result["contexts"], answer, item.reference_answer)
        row.update(
            retrieval_relevance=scores.retrieval_relevance,
            faithfulness=scores.faithfulness,
            correctness=scores.correctness,
            notes=scores.reasoning,
        )
        if refused:
            row["notes"] = "refused to answer an answerable question. " + row["notes"]
    except LLMUnavailableError as exc:
        row["notes"] = f"error: {exc.message}"
    return row


def _average(values: List[Optional[int]]) -> Optional[float]:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 2) if values else None


def summarise(rows: List[Dict]) -> Dict:
    unanswerable = [r for r in rows if not r["answerable"] and r["correctness"] is not None]
    answerable = [r for r in rows if r["answerable"] and r["correctness"] is not None]
    return {
        "questions": len(rows),
        "scored": len([r for r in rows if r["correctness"] is not None]),
        "errors": len([r for r in rows if r["notes"].startswith("error:")]),
        "avg_retrieval_relevance": _average([r["retrieval_relevance"] for r in rows]),
        "avg_faithfulness": _average([r["faithfulness"] for r in rows]),
        "avg_correctness": _average([r["correctness"] for r in rows]),
        # share of unanswerable questions where the bot correctly said "not found"
        "refusal_rate_on_unanswerable": (
            round(sum(r["correctness"] == 5 for r in unanswerable) / len(unanswerable), 2) if unanswerable else None
        ),
        # share of answerable questions the bot wrongly refused
        "wrong_refusal_rate_on_answerable": (
            round(sum("refused" in r["notes"] for r in answerable) / len(answerable), 2) if answerable else None
        ),
    }


# --------------------------------------------------------------------------- #
# Running the whole evaluation
# --------------------------------------------------------------------------- #
def run_evaluation(
    dataset_path: Optional[Path] = None,
    limit: Optional[int] = None,
    delay_seconds: float = 0.0,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Dict:
    dataset_path = Path(dataset_path or DEFAULT_DATASET)
    dataset = load_dataset(dataset_path)
    questions = dataset.questions[:limit] if limit else dataset.questions

    session_id = f"eval-{uuid.uuid4().hex[:12]}"
    store = get_index_store()
    try:
        for relative_path in dataset.documents:
            file_path = dataset_path.parent / relative_path
            pages = load_document(file_path.read_bytes(), file_path.name)
            store.add_documents(session_id, split_documents(pages), file_path.name)

        rows = []
        for number, item in enumerate(questions, start=1):
            rows.append(evaluate_question(item, session_id))
            if progress:
                progress(number, len(questions))
            if delay_seconds and number < len(questions):
                time.sleep(delay_seconds)
    finally:
        store.clear(session_id)
        clear_history(session_id)

    return {"rows": rows, "summary": summarise(rows), "limitations": LIMITATIONS}


def _cell(value) -> str:
    return "n/a" if value is None else str(value)


def format_table(result: Dict) -> str:
    """Markdown table with one row per question and an average row."""
    lines = [
        "| ID | Question | Relevance | Faithfulness | Correctness | Notes |",
        "|---|---|---|---|---|---|",
    ]
    for row in result["rows"]:
        question = row["question"] if len(row["question"]) <= 48 else row["question"][:45] + "..."
        notes = row["notes"].replace("|", "/").replace("\n", " ")[:90]
        lines.append(
            f"| {row['id']} | {question} | {_cell(row['retrieval_relevance'])} | "
            f"{_cell(row['faithfulness'])} | {_cell(row['correctness'])} | {notes} |"
        )
    s = result["summary"]
    lines.append(
        f"| | **Average (1-5)** | **{_cell(s['avg_retrieval_relevance'])}** | "
        f"**{_cell(s['avg_faithfulness'])}** | **{_cell(s['avg_correctness'])}** | |"
    )
    return "\n".join(lines)


def save_results(result: Dict, path: Optional[Path] = None) -> Path:
    path = Path(path or DEFAULT_OUTPUT)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline with an LLM judge.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    parser.add_argument("--delay", type=float, default=1.0, help="seconds to wait between questions (rate limits)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    print("Running evaluation. This makes several LLM calls per question...")
    result = run_evaluation(
        args.dataset,
        limit=args.limit,
        delay_seconds=args.delay,
        progress=lambda done, total: print(f"  question {done}/{total} done", flush=True),
    )
    print()
    print(format_table(result))
    print()
    for key, value in result["summary"].items():
        print(f"{key}: {value}")
    print("\nLimitations of LLM-based evaluation:")
    for item in LIMITATIONS:
        print(f"- {item}")
    print(f"\nSaved to {save_results(result, args.output)}")


if __name__ == "__main__":
    main()
