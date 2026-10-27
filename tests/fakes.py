"""Test doubles: a scripted chat model and a simple word-overlap embedding.

They let the whole app (retrieval, LangGraph, tools, API) be tested without
network access, API keys or model downloads.
"""
import hashlib
import math
import re
from typing import Any, Callable, List

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

DIMENSIONS = 512
STOP_WORDS = {"the", "a", "an", "is", "are", "of", "to", "and", "in", "what", "does", "do", "how", "it", "its", "for", "on", "by", "that", "this", "with", "as", "or"}


class HashingEmbeddings(Embeddings):
    """Bag-of-words vectors: texts that share words get a high cosine similarity."""

    def _embed(self, text: str) -> List[float]:
        vector = [0.0] * DIMENSIONS
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            if word in STOP_WORDS:
                continue
            index = int(hashlib.md5(word.encode()).hexdigest(), 16) % DIMENSIONS
            vector[index] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


class FakeChat(BaseChatModel):
    """A chat model whose replies come from `reply_fn(messages)`.

    reply_fn may return a str, an AIMessage (use it for tool calls) or, when the
    model is used with `with_structured_output`, a Pydantic object / dict / JSON str.
    """

    reply_fn: Any
    calls: List[Any] = Field(default_factory=list)  # every prompt this model saw

    @property
    def _llm_type(self) -> str:
        return "fake-chat"

    def _generate(self, messages: List[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls.append(list(messages))
        out = self.reply_fn(messages)
        if isinstance(out, BaseMessage):
            pass  # already a message (for example one carrying tool calls)
        elif isinstance(out, BaseModel):
            out = AIMessage(content=out.model_dump_json())
        else:
            out = AIMessage(content=str(out))
        return ChatResult(generations=[ChatGeneration(message=out)])

    def bind_tools(self, tools, **kwargs):
        return self  # the scripted reply decides whether to call a tool

    def with_structured_output(self, schema, **kwargs):
        def run(prompt):
            messages = prompt.to_messages() if hasattr(prompt, "to_messages") else list(prompt)
            self.calls.append(messages)
            out = self.reply_fn(messages)
            if isinstance(out, schema):
                return out
            if isinstance(out, BaseModel):
                return schema.model_validate(out.model_dump())
            if isinstance(out, dict):
                return schema.model_validate(out)
            if isinstance(out, str):
                return schema.model_validate_json(out)
            raise ValueError("unsupported fake reply")

        return RunnableLambda(run)


class FailingChat(BaseChatModel):
    """Always raises, to test the primary -> secondary fallback."""

    error: Any

    @property
    def _llm_type(self) -> str:
        return "failing-chat"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise self.error

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        def run(_prompt):
            raise self.error

        return RunnableLambda(run)


def tool_call_message(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])
