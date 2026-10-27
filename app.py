"""Streamlit frontend: four sections (Chat, PDF Q&A, Study Planner, Quiz).

Run:  uv run streamlit run app.py
Set API_URL (for example http://localhost:8000) to use the FastAPI backend;
leave it empty to run everything inside Streamlit.
"""
import uuid

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from api_client import ApiError, make_client  # noqa: E402

st.set_page_config(page_title="Student Assistant", page_icon="🎓", layout="wide")

LETTERS = ["A", "B", "C", "D"]

# ---------------------------------------------------------------- session state
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex  # one private session per browser tab
state = st.session_state
state.setdefault("chat_messages", [])
state.setdefault("pdf_messages", [])
state.setdefault("study_plan", None)
state.setdefault("quiz", None)
state.setdefault("quiz_submitted", False)
state.setdefault("quiz_round", 0)
state.setdefault("documents", [])

client = make_client(state.session_id)


def show_error(error: Exception) -> None:
    st.error(str(error))


def render_sources(sources) -> None:
    """Document name and page number for every passage the answer used."""
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})"):
        for source in sources:
            st.markdown(f"**{source['document']}** - page {source['page']}  \n_{source.get('snippet', '')}_")


def render_messages(messages) -> None:
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_sources(message.get("sources"))


def render_plan(plan: dict) -> None:
    st.subheader(f"📅 {plan['subject']} ({plan['level']})")
    st.caption(f"Estimated total: {plan['total_estimated_hours']:g} hours over {len(plan['daily_topics'])} days")
    st.markdown("**Learning goals**")
    for goal in plan["learning_goals"]:
        st.markdown(f"- {goal}")
    st.markdown("**Daily topics**")
    for day in plan["daily_topics"]:
        with st.expander(f"Day {day['day']}: {day['topic']}  ({day['duration_hours']:g} h)"):
            for activity in day["activities"]:
                st.markdown(f"- {activity}")
    st.markdown("**Recommended activities**")
    for activity in plan["recommended_activities"]:
        st.markdown(f"- {activity}")
    st.markdown("**Revision schedule**")
    for item in plan["revision_schedule"]:
        st.markdown(f"- **Day {item['day']}:** {item['focus']}")


def ask(messages: list, text: str, mode: str) -> None:
    """Send one message, store the answer, and draw it."""
    messages.append({"role": "user", "content": text})
    with st.chat_message("user"):
        st.markdown(text)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = client.chat(text, mode)
            except ApiError as error:
                messages.pop()  # the backend rolls the turn back too
                show_error(error)
                return
        st.markdown(result["answer"])
        render_sources(result.get("sources"))
    messages.append({"role": "assistant", "content": result["answer"], "sources": result.get("sources", [])})
    structured = result.get("structured")
    if structured and structured["type"] == "study_plan":
        state.study_plan = structured["data"]
    elif structured and structured["type"] == "quiz":
        state.quiz, state.quiz_submitted = structured["data"], False
        state.quiz_round += 1


# --------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🎓 Student Assistant")
    st.caption("RAG + LangGraph agents · Gemini / Groq")
    try:
        health = client.health()
        providers = health.get("providers_configured", [])
        if providers:
            st.success("AI ready: " + ", ".join(providers))
        else:
            st.warning("No API key found. Add GROQ_API_KEY and/or GOOGLE_API_KEY.")
    except ApiError as error:
        show_error(error)
    st.caption("Mode: " + ("FastAPI backend" if hasattr(client, "base_url") else "embedded (no separate backend)"))

    st.markdown("---")
    st.subheader("📁 Your documents")
    try:
        state.documents = client.documents()
    except ApiError:
        pass
    for name in state.documents:
        st.text(f"• {name}")
    if not state.documents:
        st.caption("None yet. Upload in the PDF Q&A tab.")
    if st.button("🗑️ Delete documents and all chats", use_container_width=True):
        try:
            client.clear("all", delete_documents=True)
            state.chat_messages, state.pdf_messages, state.documents = [], [], []
            st.rerun()
        except ApiError as error:
            show_error(error)

tab_chat, tab_pdf, tab_plan, tab_quiz = st.tabs(
    ["💬 AI Chat Assistant", "📄 PDF Question Answering", "📅 AI Study Planner", "📝 AI Quiz Generator"]
)

# ------------------------------------------------------------------- 1. chat
with tab_chat:
    st.header("AI Chat Assistant")
    st.write(
        "Ask anything academic. The assistant picks the right skill automatically: "
        "documents, study plans, quizzes or general explanations. It remembers the conversation."
    )
    if st.button("🧹 Clear chat", key="clear_chat"):
        try:
            client.clear("chat")
            state.chat_messages = []
            st.rerun()
        except ApiError as error:
            show_error(error)
    render_messages(state.chat_messages)
    if prompt := st.chat_input("Ask a question, e.g. 'What is machine learning?'", key="chat_input"):
        ask(state.chat_messages, prompt, "auto")

# --------------------------------------------------------------------- 2. pdf
with tab_pdf:
    st.header("PDF Question Answering")
    st.write("Answers come **only** from your uploaded documents, with file name and page number.")
    uploads = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
    if st.button("⚙️ Index documents", type="primary", disabled=not uploads):
        for upload in uploads:
            with st.spinner(f"Indexing {upload.name}..."):
                try:
                    info = client.upload(upload.getvalue(), upload.name)
                except ApiError as error:
                    show_error(error)
                    continue
            if info["already_indexed"]:
                st.info(f"{info['filename']} was already indexed.")
            else:
                st.success(f"{info['filename']}: {info['pages']} page(s), {info['chunks']} chunks indexed.")
        st.rerun()
    if st.button("🧹 Clear PDF chat", key="clear_pdf"):
        try:
            client.clear("pdf")
            state.pdf_messages = []
            st.rerun()
        except ApiError as error:
            show_error(error)
    render_messages(state.pdf_messages)
    if prompt := st.chat_input("Ask about your documents...", key="pdf_input"):
        ask(state.pdf_messages, prompt, "pdf")

# ---------------------------------------------------------------- 3. planner
with tab_plan:
    st.header("AI Study Planner")
    left, right = st.columns(2)
    with left:
        subject = st.text_input("Subject", placeholder="e.g. Machine Learning")
        level = st.selectbox("Knowledge level", ["Beginner", "Intermediate", "Advanced"])
    with right:
        hours = st.number_input("Hours available per day", 0.5, 16.0, 2.0, 0.5)
        days = st.number_input("Number of study days", 1, 60, 7, 1)
    if st.button("🚀 Generate study plan", type="primary"):
        if not subject.strip():
            st.warning("Please enter a subject.")
        else:
            with st.spinner("Creating your plan..."):
                try:
                    state.study_plan = client.study_plan(
                        {"subject": subject, "level": level, "hours_per_day": hours, "days": int(days)}
                    )
                except ApiError as error:
                    show_error(error)
    if state.study_plan:
        render_plan(state.study_plan)

# ------------------------------------------------------------------- 4. quiz
with tab_quiz:
    st.header("AI Quiz Generator")
    c1, c2, c3 = st.columns(3)
    topic = c1.text_input("Topic", placeholder="e.g. Python lists")
    difficulty = c2.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)
    count = c3.number_input("Number of questions", 1, 15, 5, 1)
    if st.button("🎯 Generate quiz", type="primary"):
        if not topic.strip():
            st.warning("Please enter a topic.")
        else:
            with st.spinner("Writing questions..."):
                try:
                    state.quiz = client.quiz({"topic": topic, "difficulty": difficulty, "num_questions": int(count)})
                    state.quiz_submitted = False
                    state.quiz_round += 1  # new widget keys, so old answers disappear
                except ApiError as error:
                    show_error(error)

    quiz = state.quiz
    if quiz:
        st.subheader(f"{quiz['topic']} ({quiz['difficulty']})")
        answers = []
        for number, item in enumerate(quiz["questions"]):
            labels = [f"{letter}. {text}" for letter, text in zip(LETTERS, item["options"])]
            choice = st.radio(
                f"**Q{number + 1}. {item['question']}**",
                labels,
                index=None,
                key=f"quiz_{state.quiz_round}_{number}",
                disabled=state.quiz_submitted,
            )
            answers.append(LETTERS[labels.index(choice)] if choice else None)
            if state.quiz_submitted:
                if answers[-1] == item["correct_answer"]:
                    st.success(f"Correct. {item['explanation']}")
                else:
                    correct = LETTERS.index(item["correct_answer"])
                    st.error(f"Correct answer: {labels[correct]}. {item['explanation']}")
        if not state.quiz_submitted:
            if st.button("✅ Submit answers"):
                if None in answers:
                    st.warning("Please answer every question first.")
                else:
                    state.quiz_submitted = True
                    st.rerun()
        else:
            score = sum(a == q["correct_answer"] for a, q in zip(answers, quiz["questions"]))
            st.metric("Your score", f"{score} / {len(quiz['questions'])}")
