"""Shared test fixtures and a scripted fake chat model for the ReAct agent."""

import numpy as np
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from music_recommender.config import Settings
from music_recommender.ingest import ingest
from music_recommender.store import CatalogStore


def vector(axis=0, size=1024):
    result = np.zeros(size, dtype=np.float32)
    result[axis] = 1
    return result.tolist()


class FakeEmbedder:
    def documents(self, texts):
        return [vector(1 if "rock" in text.lower() else 0) for text in texts]

    def query(self, text, purpose="tracks"):
        return vector(1 if "rock" in text.lower() else 0)


@pytest.fixture
def catalog(tmp_path):
    settings = Settings(data_dir=tmp_path / "catalog", cache_dir=tmp_path / "cache")
    rows = {
        "sessions": [
            {
                "session_id": "s1",
                "user_id": "u1",
                "user_profile": {"preferred_musical_culture": "ambient"},
                "conversation_goal": {"listener_goal": "relax", "specificity": "LL"},
            },
            {
                "session_id": "s2",
                "user_id": "u1",
                "user_profile": {"preferred_musical_culture": "ambient"},
                "conversation_goal": {"listener_goal": "rest", "specificity": "LL"},
            },
            {
                "session_id": "s3",
                "user_id": "u2",
                "user_profile": {"preferred_musical_culture": "rock"},
                "conversation_goal": {"listener_goal": "rock", "specificity": "HH"},
            },
        ],
        "tracks": [
            {
                "track_id": "t1",
                "track_name": ["Quiet"],
                "artist_name": ["A"],
                "tag_list": ["ambient"],
            },
            {"track_id": "t2", "track_name": ["Loud"], "artist_name": ["B"], "tag_list": ["rock"]},
            {"track_id": "t3", "track_name": ["Still"], "artist_name": ["C"], "tag_list": ["calm"]},
        ],
        "users": [{"user_id": "u1"}, {"user_id": "u2"}],
        "track_vectors": [
            {
                "track_id": "t1",
                "metadata-qwen3_embedding_0.6b": vector(),
                "cf-bpr": vector(size=128),
            },
            {"track_id": "t2", "metadata-qwen3_embedding_0.6b": vector(1), "cf-bpr": []},
            {
                "track_id": "t3",
                "metadata-qwen3_embedding_0.6b": vector(),
                "cf-bpr": vector(size=128),
            },
            {"track_id": "empty", "metadata-qwen3_embedding_0.6b": [], "cf-bpr": []},
        ],
        "user_vectors": [{"user_id": "u1", "cf-bpr": vector(size=128)}],
    }
    loader = lambda key, cache_dir: rows[key]
    report = ingest(settings, loader=loader, embedder=FakeEmbedder())
    return settings, CatalogStore(settings), report, loader


class ScriptedLLM:
    """A fake chat model that drives a fixed tool-call script through a ReAct loop.

    `script` is a list of steps, each either ``{"tool": name, "args": {...}}``
    (one forced tool call) or ``{"text": "final answer"}`` (ends the loop).
    Which step runs next is derived from how many AIMessages with tool_calls
    already appear in the running message list, so it stays correct no matter
    how many times the harness calls ``bind_tools`` per turn.
    """

    def __init__(self, script, explanation=None):
        self.script = script
        self.explanation = explanation
        self.tool_calls_made: list[str] = []
        self.structured_requests: list[list] = []

    def bind_tools(self, tools, **kwargs):
        def run(messages):
            done = sum(1 for m in messages if isinstance(m, AIMessage) and m.tool_calls)
            if done >= len(self.script):
                return AIMessage(content="No more scripted steps.")
            step = self.script[done]
            if "text" in step:
                return AIMessage(content=step["text"])
            self.tool_calls_made.append(step["tool"])
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": step["tool"],
                        "args": step.get("args", {}),
                        "id": f"call-{done}",
                        "type": "tool_call",
                    }
                ],
            )

        return RunnableLambda(run)

    def with_structured_output(self, schema, **kwargs):
        def run(messages):
            self.structured_requests.append(messages)
            if self.explanation is None:
                raise RuntimeError("This ScriptedLLM has no explanation configured.")
            return self.explanation

        return RunnableLambda(run)
