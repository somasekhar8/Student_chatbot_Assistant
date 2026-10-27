"""Phase 6 tests: the evaluator calculates its numbers from real runs (nothing hardcoded)."""
import pytest
from langchain_core.messages import SystemMessage, ToolMessage

from src.evaluation.rag_evaluator import (
    DEFAULT_DATASET,
    JudgeScores,
    format_table,
    load_dataset,
    run_evaluation,
    summarise,
)
from src.llm.prompts import NOT_FOUND_MESSAGE
from src.rag.vector_store import get_index_store
from tests.fakes import tool_call_message


def system_text(messages):
    first = messages[0]
    return first.content if isinstance(first, SystemMessage) else ""


def make_eval_llm(judge_scores, answer_unanswerable_with_refusal=True):
    """Fake LLM that plays both the chatbot (RAG agent) and the judge."""

    def reply(messages):
        system = system_text(messages)
        if "strict, fair grader" in system:
            return judge_scores
        if "careful academic assistant" in system:
            if isinstance(messages[-1], ToolMessage):
                if "NO_RELEVANT" in str(messages[-1].content) and not answer_unanswerable_with_refusal:
                    return "It was Brazil."  # invented; the code guard must still replace it
                return "Answer built from the passages."
            question = next(m.content for m in reversed(messages) if m.type == "human")
            return tool_call_message("search_documents", {"query": question})
        return "unused"

    return reply


def test_dataset_has_at_least_ten_questions_with_references():
    dataset = load_dataset(DEFAULT_DATASET)
    assert len(dataset.questions) >= 10
    assert all(q.reference_answer.strip() for q in dataset.questions)
    assert any(not q.answerable for q in dataset.questions)


def test_scores_come_from_the_judge_not_from_constants(install_llm):
    install_llm(make_eval_llm(JudgeScores(retrieval_relevance=5, faithfulness=4, correctness=3, reasoning="ok")))
    high = run_evaluation()["summary"]

    install_llm(make_eval_llm(JudgeScores(retrieval_relevance=2, faithfulness=2, correctness=1, reasoning="poor")))
    low = run_evaluation()["summary"]

    assert high["avg_retrieval_relevance"] > low["avg_retrieval_relevance"]
    assert high["avg_correctness"] > low["avg_correctness"]


def test_averages_are_calculated_from_rows(install_llm):
    install_llm(make_eval_llm(JudgeScores(retrieval_relevance=4, faithfulness=4, correctness=4, reasoning="fine")))
    result = run_evaluation()
    answerable = [r for r in result["rows"] if r["answerable"]]
    assert len(result["rows"]) == len(load_dataset().questions)
    assert result["summary"]["avg_retrieval_relevance"] == 4.0  # refusal rows have no retrieval score
    assert all(r["correctness"] == 4 for r in answerable)


def test_unanswerable_questions_are_checked_in_code(install_llm):
    install_llm(make_eval_llm(JudgeScores(retrieval_relevance=3, faithfulness=3, correctness=3, reasoning="x"), answer_unanswerable_with_refusal=False))
    result = run_evaluation()
    refusal_rows = [r for r in result["rows"] if not r["answerable"]]
    assert refusal_rows
    # Even though the fake LLM tried to invent "Brazil", the app replaced it with the fixed message.
    assert all(r["answer"] == NOT_FOUND_MESSAGE for r in refusal_rows)
    assert result["summary"]["refusal_rate_on_unanswerable"] == 1.0


def test_summary_handles_errors_and_empty_scores():
    rows = [
        {"id": 1, "question": "q", "answerable": True, "answer": "", "retrieval_relevance": None,
         "faithfulness": None, "correctness": None, "notes": "error: rate limit"},
    ]
    summary = summarise(rows)
    assert summary["errors"] == 1
    assert summary["avg_correctness"] is None


def test_limit_and_cleanup(install_llm):
    install_llm(make_eval_llm(JudgeScores(retrieval_relevance=5, faithfulness=5, correctness=5, reasoning="x")))
    result = run_evaluation(limit=2)
    assert len(result["rows"]) == 2
    assert not any(get_index_store().has_documents(f) for f in [])  # evaluation session removed
    assert get_index_store()._indexes == {}


def test_table_contains_every_question_and_the_average_row(install_llm):
    install_llm(make_eval_llm(JudgeScores(retrieval_relevance=5, faithfulness=4, correctness=3, reasoning="ok")))
    result = run_evaluation()
    table = format_table(result)
    assert table.count("\n") >= len(result["rows"]) + 2
    assert "Average" in table
    assert result["limitations"]


def test_judge_score_range_is_validated():
    with pytest.raises(Exception):
        JudgeScores(retrieval_relevance=9, faithfulness=1, correctness=1)
