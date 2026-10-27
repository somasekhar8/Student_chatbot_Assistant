"""Study Planner agent: the LLM reads the request and calls `generate_study_plan` with the right arguments."""
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

from src.llm.fallback import get_chat_model, safe_invoke
from src.llm.prompts import PLANNER_AGENT_PROMPT
from src.tools.study_planner import generate_study_plan
from src.workflows.state import AgentState, build_history


def planner_agent_node(state: AgentState, config: RunnableConfig) -> dict:
    model = get_chat_model(temperature=0.1, tools=[generate_study_plan])
    reply = safe_invoke(model, [SystemMessage(content=PLANNER_AGENT_PROMPT)] + build_history(state["messages"]))
    return {"messages": [reply]}
