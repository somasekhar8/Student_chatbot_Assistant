"""General Academic Assistant: answers normal study questions from the model's own knowledge."""
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from src.llm.fallback import get_chat_model, safe_invoke
from src.llm.prompts import GENERAL_PROMPT
from src.workflows.state import AgentState, build_history, message_text


def general_agent_node(state: AgentState, config: RunnableConfig) -> dict:
    model = get_chat_model(temperature=0.4)
    reply = safe_invoke(model, [SystemMessage(content=GENERAL_PROMPT)] + build_history(state["messages"]))
    return {"messages": [AIMessage(content=message_text(reply))]}
