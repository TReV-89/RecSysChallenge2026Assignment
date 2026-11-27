import pytest
from jsonschema import ValidationError, validate

from music_recommender.agent import repair_schema, validate_explanations
from music_recommender.schemas import (
    Candidate,
    ExplanationDraft,
    RankedTrack,
    Track,
    TrackExplanation,
)


def test_repair_schema_rejects_shortened_tags_and_unknown_ids():
    schema = repair_schema([{"track_id": "a", "tags": ["1970s: jazz vocal"]}])
    entry = {
        "track_id": "a",
        "explanation": "Fits the supplied jazz tag.",
        "cited_tags": ["1970s: jazz vocal"],
    }
    validate({"tracks": [entry]}, schema)
    for invalid in ({**entry, "cited_tags": ["jazz vocal"]}, {**entry, "track_id": "unknown"}):
        with pytest.raises(ValidationError):
            validate({"tracks": [invalid]}, schema)
    empty = repair_schema([{"track_id": "a", "tags": []}])
    validate({"tracks": [{**entry, "cited_tags": []}]}, empty)
    with pytest.raises(ValidationError):
        validate({"tracks": [entry]}, empty)


def test_validation_still_rejects_tags_borrowed_from_another_repair_track():
    ranked = [
        RankedTrack(
            candidate=Candidate(
                track=Track(track_id=key, title=key, artists=[], tags=[tag]),
                query_score=0.8,
                neighbor_score=0,
                supporting_users=[],
            ),
            score=0.8,
        )
        for key, tag in [("a", "1970s: jazz vocal"), ("b", "punk")]
    ]
    draft = ExplanationDraft(
        tracks=[
            TrackExplanation(track_id="a", explanation="Fits.", cited_tags=["punk"]),
            TrackExplanation(track_id="b", explanation="Fits.", cited_tags=["punk"]),
        ]
    )
    with pytest.raises(ValueError, match="punk"):
        validate_explanations(draft, ranked)
    draft.tracks[0].cited_tags = ["1970s: jazz vocal"]
    assert set(validate_explanations(draft, ranked)) == {"a", "b"}
