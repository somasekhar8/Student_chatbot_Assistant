"""Quiz Generator agent: the LLM reads the request and calls `generate_quiz` with the right arguments."""
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

from src.llm.fallback import get_chat_model, safe_invoke
from src.llm.prompts import QUIZ_AGENT_PROMPT
from src.tools.quiz_generator import generate_quiz
from src.workflows.state import AgentState, build_history


def quiz_agent_node(state: AgentState, config: RunnableConfig) -> dict:
    model = get_chat_model(temperature=0.1, tools=[generate_quiz])
    reply = safe_invoke(model, [SystemMessage(content=QUIZ_AGENT_PROMPT)] + build_history(state["messages"]))
    return {"messages": [reply]}
