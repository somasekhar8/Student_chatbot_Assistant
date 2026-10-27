"""Smoke test of the Streamlit UI with Streamlit's AppTest (embedded mode, fake LLM)."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

from tests.factories import make_plan, make_quiz


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def run_app():
    app = AppTest.from_file(str(APP_PATH), default_timeout=30)
    app.run()
    return app


def test_app_loads_with_four_tabs_and_no_exception():
    app = run_app()
    assert not app.exception
    assert [t.label for t in app.tabs] == [
        "💬 AI Chat Assistant",
        "📄 PDF Question Answering",
        "📅 AI Study Planner",
        "📝 AI Quiz Generator",
    ]
    assert any("No API key" in w.value for w in app.sidebar.warning)


def test_missing_key_shows_friendly_error_not_a_crash():
    app = run_app()
    app.text_input[0].set_value("Python")  # study planner subject
    next(b for b in app.button if "Generate study plan" in b.label).click().run()
    assert not app.exception
    assert any("No AI model is configured" in e.value for e in app.error)


def test_quiz_flow_scores_answers(install_llm):
    install_llm(lambda messages: make_quiz(count=2))
    app = run_app()
    app.text_input[1].set_value("Machine Learning")
    app.number_input[2].set_value(2)  # the fake model returns 2 questions
    next(b for b in app.button if "Generate quiz" in b.label).click().run()
    assert not app.exception and len(app.radio) == 2

    for radio, letter_index in zip(app.radio, (0, 1)):  # make_quiz: answers A then B
        radio.set_value(radio.options[letter_index])
    app.run()  # in a browser every click on an option triggers its own rerun
    next(b for b in app.button if "Submit" in b.label).click().run()
    assert app.metric[0].value == "2 / 2"


def test_study_plan_is_displayed(install_llm):
    install_llm(lambda messages: make_plan(days=3))
    app = run_app()
    app.text_input[0].set_value("Python")
    app.number_input[1].set_value(3)
    next(b for b in app.button if "Generate study plan" in b.label).click().run()
    assert not app.exception
    assert any("Learning goals" in m.value for m in app.markdown)
