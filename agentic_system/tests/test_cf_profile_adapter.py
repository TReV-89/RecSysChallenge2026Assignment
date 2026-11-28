"""Check the new CF profile correction and centered neighbor calculation."""

import hashlib

import numpy as np
import pytest

from music_recommender.cf_profile_adapter import CultureAdapter, culture_tokens
from music_recommender.config import Settings
from music_recommender.retrieval import Retriever
from music_recommender.schemas import RecommendationRequest
from music_recommender.tools.calculate_user_similarity import find_cf_neighbors


def vector(first=0.0, second=0.0):
    result = np.zeros(128, dtype=np.float32)
    result[:2] = [first, second]
    return result


def test_full_pipeline_defaults_to_official_adapter_and_centered_cosine(monkeypatch):
    monkeypatch.setattr("music_recommender.config.load_dotenv", lambda *args: None)
    for name in (
        "CF_HISTORY_MODE", "CF_NEIGHBOR_SIMILARITY", "CF_PROFILE_ADAPTER_ENABLED",
        "CF_ADAPTER_WEIGHTS_PATH", "CF_ADAPTER_USERS_PATH", "TALKPLAY_CATALOG",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = Settings.from_env(cf_history_mode="official_test")
    assert settings.cf_neighbor_similarity == "population_centered"
    assert settings.cf_adapter_weights_path.name == "culture_adapter_official_weights.npz"
    assert settings.cf_adapter_users_path.name == "culture_adapter_official_users.npz"
    assert settings.cf_adapter_weights_path.exists()
    assert settings.cf_adapter_users_path.exists()

    monkeypatch.setenv("CF_HISTORY_MODE", "internal_validation")
    internal = Settings.from_env()
    assert internal.cf_adapter_weights_path.name == "culture_adapter_final_weights.npz"
    # An explicit scope can select the official artifact independently of .env.
    assert Settings.from_env(cf_history_mode="official_test").cf_adapter_weights_path == (
        settings.cf_adapter_weights_path
    )

    monkeypatch.setenv("CF_PROFILE_ADAPTER_ENABLED", "0")
    baseline = Settings.from_env(cf_history_mode="official_test")
    assert baseline.cf_adapter_weights_path is None
    assert baseline.cf_adapter_users_path is None
    assert baseline.cf_neighbor_similarity == "cosine"


def test_centered_neighbors_use_one_training_population_center():
    query = vector(2, 2)
    neighbors = {"self": query, "a": vector(3, 1), "b": vector(1, 3)}
    center = vector(1, 1)
    result = find_cf_neighbors("self", query, neighbors, center_vector=center)
    assert [user for user, _ in result] == ["a", "b"]
    assert [score for _, score in result] == pytest.approx([2**-0.5, 2**-0.5])
    assert find_cf_neighbors("self", center, neighbors, center_vector=center) == []


def test_adapter_enhances_query_by_route_and_rejects_wrong_catalog(tmp_path, monkeypatch):
    weights = np.zeros((1, 128), dtype=np.float32)
    weights[0, 0] = 0.5
    base = vector(0, 1)
    original = {"a": base.tolist(), "b": vector(1, 0).tolist()}
    original_matrix = np.stack([original["a"], original["b"]]).astype("<f4")
    fingerprint = hashlib.sha256(original_matrix.tobytes()).hexdigest()
    weights_path = tmp_path / "weights.npz"
    users_path = tmp_path / "users.npz"
    np.savez_compressed(
        weights_path, tokens=np.asarray(["afrobeat"]), correction_weights=weights,
        user_to_user_alpha=np.float32(1), user_to_item_alpha=np.float32(0.5),
        maximum_correction_fraction=np.float32(0.5),
    )
    np.savez_compressed(
        users_path, user_ids=np.asarray(["a", "b"]),
        base_user_factors_sha256=np.asarray(fingerprint),
        user_to_user_vectors=np.stack([vector(0.5, 1), vector(1, 0)]),
        user_to_item_vectors=np.stack([vector(0.25, 1), vector(1, 0)]),
    )
    adapter = CultureAdapter(weights_path, users_path)
    profile = "preferred musical culture: Afrobeat; age group: 25-34"
    assert culture_tokens(profile) == {"afrobeat"}
    assert adapter.has_known_culture(profile)
    assert not adapter.has_known_culture("age group: 25-34")
    assert adapter.enhance(base, profile, "user_to_user")[:2] == pytest.approx([0.5, 1])
    assert adapter.enhance(base, profile, "user_to_item")[:2] == pytest.approx([0.25, 1])
    assert adapter.enhance(base, "age group: 25-34", "user_to_user")[:2] == [0, 1]
    assert list(adapter.training_pool(original)) == ["a", "b"]
    with pytest.raises(ValueError, match="different published user factors"):
        adapter.training_pool({"a": vector(1, 0), "b": vector(1, 0)})

    class Store:
        def state(self, key):
            return key in {"ready", "cf_query_vectors_ready", "cf_history_official_test_ready"}

        def records(self, kind, ids):
            return {"query": {"vector": base.tolist()}} if kind == "cf_query_vectors" else {}

        def all_vectors(self, kind):
            return original

        def existing_record_ids(self, kind, ids):
            return set(ids)

    retriever = Retriever(
        Settings(cf_adapter_weights_path=weights_path, cf_adapter_users_path=users_path,
                 cf_neighbor_similarity="population_centered"),
        Store(), None,
    )
    request = RecommendationRequest(message="music", user_id="query", profile=profile)
    assert retriever._query_cf_vector(request, "user_to_user")[:2] == pytest.approx([0.5, 1])
    assert retriever._query_cf_vector(request, "user_to_item")[:2] == pytest.approx([0.25, 1])

    observed = {}

    def capture_neighbors(query_user_id, query_vector, training_vectors, *, center_vector):
        observed["query"] = query_vector
        observed["training"] = training_vectors
        observed["center"] = center_vector
        return []

    monkeypatch.setattr("music_recommender.retrieval.find_cf_neighbors", capture_neighbors)
    retriever.retrieve_cf(request)
    assert observed["query"][:2] == pytest.approx([0.5, 1])
    assert observed["training"]["a"][:2] == pytest.approx([0.5, 1])
    assert observed["center"][:2] == pytest.approx([0.75, 0.5])


def test_neighbor_cache_reapplies_turn_exclusions():
    class Store:
        def state(self, key):
            return key in {"ready", "cf_history_official_test_ready"}

        def vectors(self, kind, ids):
            return {"query": vector(1, 0).tolist()}

        def all_vectors(self, kind):
            return {"neighbor": vector(1, 0).tolist()}

        def records(self, kind, ids):
            if kind == "cf_history_official_test":
                return {"neighbor": {"track_ids": ["t1", "t2"]}} if "neighbor" in ids else {}
            if kind == "tracks":
                return {
                    track: {"track_id": track, "track_name": [track], "artist_name": ["A"]}
                    for track in ids
                }
            return {}

        def existing_record_ids(self, kind, ids):
            return set(ids)

    retriever = Retriever(Settings(candidate_count=2), Store(), None)
    first = retriever.retrieve_cf(RecommendationRequest(
        message="music", user_id="query", excluded_track_ids=["t1"]
    ))
    second = retriever.retrieve_cf(RecommendationRequest(message="music", user_id="query"))
    assert [candidate.track.track_id for candidate in first.candidates] == ["t2"]
    assert [candidate.track.track_id for candidate in second.candidates] == ["t1", "t2"]
