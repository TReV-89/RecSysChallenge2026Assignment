from conftest import ScriptedLLM
from fastapi.testclient import TestClient

from main import app
from music_recommender.agent import build_agent
from music_recommender.config import Settings
from music_recommender.schemas import RecommendationResponse


class StubAgent:
    def __init__(self):
        self.calls = []

    def invoke(self, state, config):
        self.calls.append((state, config))
        return {
            "response": RecommendationResponse(
                recommendations=[],
                neighbors=[],
                notices=[],
                mode="groq",
            )
        }


def test_recommendation_endpoint_returns_response_and_checkpoint_id(monkeypatch):
    agent = StubAgent()
    monkeypatch.setattr(app.state, "agent", agent, raising=False)
    client = TestClient(app)

    result = client.post(
        "/recommendations",
        json={"message": "mellow instrumental music"},
        headers={"X-Conversation-ID": "conversation-123"},
    )

    assert result.status_code == 200
    assert result.json()["mode"] == "groq"
    assert result.headers["X-Conversation-ID"] == "conversation-123"
    assert agent.calls[0][0]["request"].message == "mellow instrumental music"
    assert agent.calls[0][1] == {"configurable": {"thread_id": "conversation-123"}}


def test_recommendation_endpoint_generates_checkpoint_id(monkeypatch):
    monkeypatch.setattr(app.state, "agent", StubAgent(), raising=False)
    result = TestClient(app).post("/recommendations", json={"message": "ambient"})

    assert result.status_code == 200
    assert result.headers["X-Conversation-ID"]


def test_recommendation_endpoint_reports_unavailable_service(monkeypatch):
    monkeypatch.delattr(app.state, "agent", raising=False)
    result = TestClient(app).post("/recommendations", json={"message": "ambient"})

    assert result.status_code == 503
    assert result.json()["detail"] == "Recommendation service is not ready."


def test_greeting_endpoint_answers_without_search(monkeypatch):
    class NoSearch:
        def retrieve_text(self, request, query):
            raise AssertionError("Greeting triggered music retrieval")

    llm = ScriptedLLM(script=[{"text": "Hi! Tell me what music you're after."}])
    monkeypatch.setattr(app.state, "agent", build_agent(Settings(), NoSearch(), llm), raising=False)
    result = TestClient(app).post("/chat", json={"message": "hello"})
    assert result.status_code == 200
    assert result.json()["kind"] == "chat"
    assert result.json()["message"]
    assert result.json()["recommendations"] == []
