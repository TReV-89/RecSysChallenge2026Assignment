import json
from io import BytesIO

import pytest

from music_recommender.agent import (
    create_explanation_llm,
    create_llm,
    explanation_chain,
    parse_explanation_call,
)
from music_recommender.config import Settings
from music_recommender.evaluation.openrouter import require_turn_budget


def test_explanation_llm_falls_back_to_main_loop_without_a_second_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPEN_ROUTER_API_KEY", raising=False)
    settings = Settings(llm_provider="groq")
    llm, used_settings = create_explanation_llm(settings)
    assert llm is None
    assert used_settings is settings


def test_same_provider_explanations_ignore_router_credentials(monkeypatch):
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "unused-test-key")
    settings = Settings(llm_provider="groq", explanation_provider="same")
    llm, used_settings = create_explanation_llm(settings)
    assert llm is None
    assert used_settings is settings


def test_react_budget_does_not_reduce_explanation_budget():
    from music_recommender.agent import build_agent

    settings = Settings(explanation_provider="same", max_output_tokens=4096,
                        react_max_output_tokens=1024)
    llm = create_llm(settings, "test-key")
    agent = build_agent(settings, None, llm)
    assert agent.tool_llm.max_tokens == 1024
    assert agent.llm is llm
    assert llm.max_tokens == 4096


def test_explanation_llm_uses_a_dedicated_openrouter_client_when_configured(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "second-key")
    settings = Settings(llm_provider="groq")  # main loop stays on Groq
    llm, used_settings = create_explanation_llm(settings)
    assert used_settings.llm_provider == "openrouter"
    assert llm.model_name == settings.openrouter_model
    assert llm.openai_api_base == "https://openrouter.ai/api/v1"


def test_judge_credentials_do_not_switch_recommendation_provider(monkeypatch):
    monkeypatch.setattr("music_recommender.config.load_dotenv", lambda *args: None)
    for name in (
        "LLM_PROVIDER",
        "OPENROUTER_API_KEY",
        "OPENROUTER_MODEL",
        "OPENROUTER_OUTPUT_METHOD",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-router-key")
    monkeypatch.setenv("OPEN_ROUTER_MODEL", "")
    settings = Settings.from_env()
    assert settings.llm_provider == "groq"
    settings.llm_provider = "openrouter"
    assert settings.api_key() == "test-router-key"
    assert settings.model_name == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert settings.structured_output_method == "function_calling"
    assert "test-router-key" not in settings.model_dump_json()
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    assert Settings.from_env().api_key() == "test-groq-key"


def test_openrouter_client_uses_requested_endpoint_with_bounded_retries():
    settings = Settings(llm_provider="openrouter")
    llm = create_llm(settings, "test-key")
    assert llm.model_name == settings.openrouter_model
    assert llm.openai_api_base == "https://openrouter.ai/api/v1"
    assert llm.max_retries == 2  # bounded backoff for transient rate limits, not unlimited
    assert llm.use_responses_api is False
    chain = explanation_chain(settings, llm)
    assert "parallel_tool_calls" not in chain.first.kwargs
    assert chain.first.kwargs["tool_choice"]["function"]["name"] == "ExplanationDraft"


def test_openrouter_explanations_require_exactly_one_valid_call():
    from langchain_core.messages import AIMessage

    call = {"name": "ExplanationDraft", "args": {"tracks": []}, "id": "one"}
    assert parse_explanation_call(AIMessage(content="", tool_calls=[call])).tracks == []
    for calls in ([], [call, {**call, "id": "two"}], [{**call, "name": "wrong"}]):
        with pytest.raises(ValueError):
            parse_explanation_call(AIMessage(content="", tool_calls=calls))


@pytest.mark.parametrize("repair_succeeds", [True, False])
def test_explanation_repair_is_validated_and_bounded(monkeypatch, repair_succeeds):
    from langchain_core.runnables import RunnableLambda

    from music_recommender.agent import _explain
    from music_recommender.ranking import rank_candidates
    from music_recommender.schemas import (
        Candidate,
        ExplanationDraft,
        RecommendationRequest,
        RetrievalResult,
        Track,
        TrackExplanation,
    )

    retrieval = RetrievalResult(
        query="calm",
        neighbors=[],
        candidates=[
            Candidate(
                track=Track(track_id="t", title="Song", artists=[], tags=[]),
                query_score=0.8,
                neighbor_score=0,
                supporting_users=[],
            )
        ],
    )
    request = RecommendationRequest(message="calm")
    ranked = rank_candidates(retrieval, request, [])

    calls = []

    def explain(messages):
        calls.append(messages)
        tracks = (
            [TrackExplanation(track_id="t", explanation="Fits.", cited_tags=[])]
            if len(calls) == 2 and repair_succeeds
            else []
        )
        return ExplanationDraft(tracks=tracks)

    monkeypatch.setattr(
        "music_recommender.agent.explanation_chain", lambda *args: RunnableLambda(explain)
    )
    explanations, notices = _explain(Settings(llm_provider="openrouter"), object(), request, ranked)
    assert len(calls) == 2
    assert "missing=['t']" in calls[1][-1].content
    assert bool(notices) == (not repair_succeeds)
    assert ranked[0].score == 0.8
    if repair_succeeds:
        assert explanations["t"] == "Fits."


@pytest.mark.parametrize("remaining", [0, 1, 2, 3, 4, 50])
def test_free_allowance_reserves_a_complete_turn(monkeypatch, remaining):
    def fake_urlopen(request, timeout):
        assert request.full_url == "https://openrouter.ai/api/v1/key"
        assert request.get_header("Authorization") == "Bearer test-key"
        return BytesIO(
            json.dumps(
                {
                    "data": {
                        "free_model_daily_requests": {
                            "used": 50 - remaining,
                            "limit": 50,
                            "remaining": remaining,
                        }
                    }
                }
            ).encode()
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    if remaining < 4:
        with pytest.raises(RuntimeError, match="Completed predictions are saved"):
            require_turn_budget("test-key")
    else:
        assert require_turn_budget("test-key")["remaining"] == remaining
