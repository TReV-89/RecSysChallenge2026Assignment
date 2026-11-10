"""Data/model boundary checks that do not need provider SDKs or external downloads."""

import numpy as np
import pytest
from pydantic import ValidationError

from music_recommender.config import Settings
from music_recommender.embeddings import cosine01, validate_vector
from music_recommender.ingest import ingest, session_record
from music_recommender.schemas import RecommendationRequest, SimilarityWeights
from music_recommender.store import CatalogStore


def test_published_vectors_preserve_values_and_dimensions():
    supplied = [0.25] * 1024
    stored = validate_vector(supplied, 1024)
    assert np.allclose(stored, supplied)
    with pytest.raises(ValueError):
        validate_vector([0.25] * 128, 1024)


def test_missing_and_invalid_embeddings():
    assert validate_vector(None, 1024) is None
    assert validate_vector([], 1024) is None
    assert validate_vector([0] * 1024, 1024) is None
    with pytest.raises(ValueError):
        validate_vector([float("nan")] * 1024, 1024)


def test_cosine_score_scale():
    assert cosine01([1, 0], [1, 0]) == pytest.approx(1)
    assert cosine01([1, 0], [0, 1]) == pytest.approx(0.5)
    assert cosine01([1, 0], [-1, 0]) == pytest.approx(0)


def test_requests_reject_unknown_labels_and_bad_weights():
    with pytest.raises(ValidationError):
        RecommendationRequest(message="music", specificity="invented")
    with pytest.raises(ValidationError):
        SimilarityWeights(profile=0, goal=0, specificity=0)
    with pytest.raises(ValidationError):
        SimilarityWeights(profile=float("nan"))
    with pytest.raises(ValidationError):
        RecommendationRequest(message="music", unexpected="field")


def test_session_projection_excludes_future_conversation_content():
    record = session_record(
        {
            "session_id": "s",
            "user_id": "u",
            "user_profile": {"preferred_musical_culture": "Ambient"},
            "conversation_goal": {"listener_goal": "Relax", "specificity": "LL"},
            "conversations": [{"thought": "private", "content": "future answer"}],
        }
    )
    assert record["profile_text"] == "preferred musical culture: Ambient"
    assert record["specificity"] == "LL"
    assert "conversations" not in record


def test_chroma_stores_separate_spaces_and_filters_seen_ids(tmp_path):
    store = CatalogStore(Settings(data_dir=tmp_path))
    qwen = [1.0] + [0.0] * 1023
    bpr = [1.0] + [0.0] * 127
    store.upsert("tracks", ["a", "b"], [qwen, qwen], [{"track_id": "a"}, {"track_id": "b"}])
    store.upsert("users_cf", ["u"], [bpr], [{"user_id": "u"}])
    assert store.search("tracks", qwen, 10, {"track_id": {"$nin": ["a"]}}) == ["b"]
    assert len(store.vectors("tracks", ["a"])["a"]) == 1024
    assert len(store.vectors("users_cf", ["u"])["u"]) == 128
    store.save_records("tracks", [{"track_id": "a", "tag_list": ["calm", "ambient"]}], "track_id")
    assert store.records("tracks", ["a"])["a"]["tag_list"] == ["calm", "ambient"]


def test_ingestion_with_real_store_and_injected_source(tmp_path):
    qwen = [1.0] + [0.0] * 1023
    bpr = [1.0] + [0.0] * 127
    rows = {
        "sessions": [
            {
                "session_id": "s",
                "user_id": "u",
                "user_profile": {"preferred_musical_culture": "ambient"},
                "conversation_goal": {"listener_goal": "relax", "specificity": "LL"},
            }
        ],
        "tracks": [{"track_id": "t", "track_name": ["Quiet"]}],
        "users": [{"user_id": "u", "country_name": "Rwanda"}],
        "track_vectors": [{"track_id": "t", "metadata-qwen3_embedding_0.6b": qwen, "cf-bpr": bpr}],
        "user_vectors": [{"user_id": "u", "cf-bpr": bpr}],
    }

    class Embedder:
        def documents(self, texts):
            return [qwen for _ in texts]

    settings = Settings(data_dir=tmp_path)
    first = ingest(settings, loader=lambda key, cache: rows[key], embedder=Embedder())
    second = ingest(settings, loader=lambda key, cache: rows[key], embedder=Embedder())
    assert first == second
    assert first["tracks"] == first["profiles"] == first["users_cf"] == 1
    assert CatalogStore(settings).state("ready") is True
