import math
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from music_recommender.evaluate import append_checkpoint, read_checkpoint
from music_recommender.evaluation.judge import JudgeScore
from music_recommender.evaluation.metrics import distinct2, ndcg, summarize
from music_recommender.evaluation.protocol import extract_cases, sample_sessions
from music_recommender.schemas import RecommendationRequest


def session(session_id="s"):
    return {
        "session_id": session_id,
        "user_id": "u",
        "user_profile": {"preferred_language": "English", "user_split": "HIDDEN_SPLIT"},
        "conversation_goal": {"listener_goal": "HIDDEN_GOAL", "specificity": "HH"},
        "goal_progress_assessments": ["HIDDEN_ASSESSMENT"],
        "conversations": [
            {
                "turn_number": number,
                "role": role,
                "content": f"{role}-{number}",
                "thought": "HIDDEN_THOUGHT",
            }
            for number in range(1, 9)
            for role in ("user", "music", "assistant")
        ],
    }


def test_turn_inputs_exclude_current_future_answers_and_hidden_fields():
    cases = extract_cases(session())
    assert len(cases) == 8
    for case in cases:
        assert case.target_track_id == f"music-{case.turn_number}"
        assert case.request.limit == 20
        text = case.request.model_dump_json()
        assert "HIDDEN" not in text
        assert case.request.goal == ""
        assert case.request.specificity is None
        assert case.request.excluded_track_ids == []
        for number in range(case.turn_number, 9):
            assert f"music-{number}" not in text
            assert f"assistant-{number}" not in text
        for number in range(1, case.turn_number):
            assert f"assistant-{number}" in text
            assert f"music-{number}" in text
    assert len(cases[0].request.history) == 0
    assert len(cases[7].request.history) == 14


def test_session_sampling_is_repeatable_and_order_independent():
    rows = [session(str(number)) for number in range(20)]
    chosen = sample_sessions(rows, 10, 42)
    assert chosen == sample_sessions(list(reversed(rows)), 10, 42)
    assert len(chosen) == 10
    assert len(sample_sessions(rows, 0, 42)) == 20
    with pytest.raises(ValueError):
        sample_sessions(rows, 21, 42)
    with pytest.raises(ValueError):
        sample_sessions([rows[0], rows[0]], 1, 42)


def test_malformed_conversations_fail_instead_of_silently_skipping_turns():
    row = session()
    row["conversations"].pop()
    with pytest.raises(ValueError):
        extract_cases(row)
    row = session()
    row["conversations"].append(row["conversations"][0])
    with pytest.raises(ValueError):
        extract_cases(row)


def test_ndcg_cutoffs_and_one_relevant_track():
    ids = [str(i) for i in range(20)]
    assert ndcg("0", ids, 1) == 1
    assert ndcg("1", ids, 1) == 0
    assert ndcg("9", ids, 10) == pytest.approx(1 / math.log2(11))
    assert ndcg("19", ids, 10) == 0
    assert ndcg("19", ids, 20) == pytest.approx(1 / math.log2(21))
    assert ndcg("missing", ids, 20) == 0
    assert ndcg("0", [], 20) == 0
    with pytest.raises(ValueError):
        ndcg("0", ["0", "0"], 20)


def test_distinct2_matches_official_case_whitespace_and_response_boundaries():
    assert distinct2(["A b", "a b", "B c"]) == pytest.approx(2 / 3)
    assert distinct2(["a", "b"]) == 0
    assert distinct2(["a b!", "a b"]) == 1  # punctuation is retained
    assert distinct2([]) == 0


def fixture_predictions():
    cases = extract_cases(session("s1")) + extract_cases(session("s2"))
    predictions = [
        {
            "session_id": case.session_id,
            "user_id": case.user_id,
            "turn_number": case.turn_number,
            "predicted_track_ids": [case.target_track_id] if case.session_id == "s1" else [],
            "predicted_response": "A good song",
            "mode": "groq",
        }
        for case in cases
    ]
    return cases, predictions, {f"music-{i}" for i in range(1, 9)} | {"unindexed"}


def test_full_metrics_and_missing_judge_do_not_renormalize_weights():
    cases, predictions, catalog = fixture_predictions()
    result = summarize(cases, predictions, catalog)
    assert result["ndcg@20"] == 0.5
    assert result["catalog_diversity"] == pytest.approx(8 / 9)
    assert result["lexical_diversity"] == pytest.approx(1 / 16)
    assert result["weighted_non_judge_subtotal_out_of_0.70"] == pytest.approx(
        0.25 + 0.1 * 8 / 9 + 0.1 / 16
    )
    assert result["composite_proxy"] is None
    assert result["official_blind_composite"] is None
    assert result["sessions"] == 2 and result["turns"] == 16
    assert result["ndcg20_session_bootstrap_95ci"]["unit"] == "session"


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "invalid_id", "duplicate_id", "too_many"]
)
def test_invalid_prediction_sets_cannot_inflate_scores(mutation):
    cases, predictions, catalog = fixture_predictions()
    if mutation == "missing":
        predictions.pop()
    elif mutation == "duplicate":
        predictions.append(predictions[0])
    elif mutation == "invalid_id":
        predictions[0]["predicted_track_ids"] = ["invented"]
    elif mutation == "duplicate_id":
        predictions[0]["predicted_track_ids"] *= 2
    elif mutation == "too_many":
        predictions[0]["predicted_track_ids"] = [str(i) for i in range(21)]
        catalog.update(predictions[0]["predicted_track_ids"])
    with pytest.raises(ValueError):
        summarize(cases, predictions, catalog)


def test_judge_normalization_and_composite_are_explicitly_proxies():
    cases, predictions, catalog = fixture_predictions()
    judges = {
        case.key: {"personalization": 5, "explanation_quality": 1, "rationale": "Mixed."}
        for case in cases
    }
    result = summarize(cases, predictions, catalog, judges)
    assert result["llm_judge_proxy"] == 0.5
    assert result["composite_proxy"] == result["weighted_non_judge_subtotal_out_of_0.70"] + 0.15
    assert result["official_blind_composite"] is None
    with pytest.raises(ValidationError):
        JudgeScore(personalization=3.5, explanation_quality=5, rationale="Invalid.")
    with pytest.raises(ValidationError):
        JudgeScore(personalization=True, explanation_quality=5, rationale="Invalid.")
    judges.pop(cases[0].key)
    with pytest.raises(ValueError):
        summarize(cases, predictions, catalog, judges)


def test_checkpoint_is_resumable_and_does_not_ignore_corruption(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    append_checkpoint(path, {"turn_number": 1})
    append_checkpoint(path, {"turn_number": 2})
    assert read_checkpoint(path) == [{"turn_number": 1}, {"turn_number": 2}]
    with path.open("a") as stream:
        stream.write('{"partial":')
    with pytest.raises(ValueError, match="repair before resuming"):
        read_checkpoint(path)


def test_request_supports_top20_but_preserves_interactive_default():
    assert RecommendationRequest(message="music").limit == 5
    assert RecommendationRequest(message="music", limit=20).limit == 20
    with pytest.raises(ValidationError):
        RecommendationRequest(message="music", limit=21)


def test_nemotron_judge_is_separate_and_receives_no_labels(monkeypatch):
    import json
    from io import BytesIO

    from music_recommender.evaluation.judge import judge_response

    def fake_urlopen(request, timeout):
        assert request.full_url == "https://openrouter.ai/api/v1/chat/completions"
        assert request.get_header("Authorization") == "Bearer judge-key"
        payload = json.loads(request.data)
        assert payload["model"] == "nvidia/nemotron-test"
        text = payload["messages"][1]["content"]
        assert "music-1" not in text and "assistant-1" not in text
        assert "ground_truth" not in text and "HIDDEN" not in text
        return BytesIO(
            json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "JudgeScore",
                                            "arguments": json.dumps(
                                                {
                                                    "personalization": 4,
                                                    "explanation_quality": 3,
                                                    "rationale": "Useful explanation.",
                                                }
                                            ),
                                        }
                                    }
                                ]
                            },
                        }
                    ]
                }
            ).encode()
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    score = judge_response(
        extract_cases(session())[0].request,
        "Generated response",
        "nvidia/nemotron-test",
        "judge-key",
    )
    assert score.normalized == 0.625


def test_runner_saves_resumes_and_scores_without_regenerating(monkeypatch, tmp_path):
    import json

    from music_recommender.config import Settings
    from music_recommender.evaluate import run
    from music_recommender.schemas import Recommendation, RecommendationResponse, Track

    data_dir = tmp_path / "catalog"
    data_dir.mkdir()
    (data_dir / "catalog.sqlite3").touch()
    bundle = {
        "sessions": [session()],
        "catalog_ids": [f"music-{i}" for i in range(1, 9)],
        "total_test_sessions": 1,
    }
    monkeypatch.setattr("music_recommender.evaluate.load_inputs", lambda *args: bundle)
    monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls: cls()))

    class Store:
        def __init__(self, settings):
            pass

        def state(self, key):
            return {"ready": True, "scope": {"sample": False}, "identity": {}}.get(key)

        def counts(self):
            return {"tracks": 8}

        def records(self, kind):
            return {item: {} for item in bundle["catalog_ids"]}

    calls = []

    class Agent:
        def invoke(self, state):
            calls.append(state["request"])
            return {
                "response": RecommendationResponse(
                    recommendations=[
                        Recommendation(
                            track=Track(
                                track_id="music-1", title="Song", artists=["Artist"], tags=[]
                            ),
                            score=1,
                            explanation="Matches the request.",
                            supporting_users=[],
                        )
                    ],
                    neighbors=[],
                    notices=[],
                    mode="local",
                )
            }

    monkeypatch.setattr("music_recommender.store.CatalogStore", Store)
    monkeypatch.setattr("music_recommender.agent.build_agent", lambda *args: Agent())
    args = SimpleNamespace(
        data_dir=data_dir,
        output=tmp_path / "run",
        sessions=1,
        seed=42,
        mode="local",
        delay=0,
        score_only=False,
        allow_fallbacks=False,
        judge_model=None,
    )
    run(args)
    assert len(calls) == 8
    run(args)
    assert len(calls) == 8  # No regenerated provider calls on resume.
    args.score_only = True
    run(args)
    scores = json.loads((args.output / "scores.json").read_text())
    assert scores["ndcg@20"] == 0.125
    exported = json.loads((args.output / "predictions.json").read_text())
    assert len(exported) == 8 and "ground_truth_track_id" not in exported[0]
    assert "mode" not in exported[0]  # Official prediction fields only.
    args.score_only = False
    args.allow_fallbacks = True
    with pytest.raises(ValueError, match="configuration/code changed"):
        run(args)
