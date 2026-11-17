"""Turn extraction and reproducible session sampling for the public test split."""

import random
from collections import defaultdict
from dataclasses import dataclass

from ..schemas import ChatTurn, RecommendationRequest

DATASET = "talkpl-ai/TalkPlayData-Challenge-Dataset"
CATALOG = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"


@dataclass
class EvaluationCase:
    session_id: str
    user_id: str
    turn_number: int
    target_track_id: str
    request: RecommendationRequest

    @property
    def key(self):
        return self.session_id, self.turn_number


def sample_sessions(rows: list[dict], count: int, seed: int) -> list[dict]:
    """Sample complete sessions uniformly, independent of the dataset's row order.

    A count of zero means all sessions. Keep the selected IDs fixed for model
    comparisons; selecting individual turns would break conversation context.
    """
    ordered = sorted(rows, key=lambda row: row["session_id"])
    if len({row["session_id"] for row in ordered}) != len(ordered):
        raise ValueError("Dataset contains duplicate session IDs.")
    if count < 0 or count > len(ordered):
        raise ValueError(f"Session count must be between 0 and {len(ordered)}.")
    return sorted(
        random.Random(seed).sample(ordered, count or len(ordered)),
        key=lambda row: row["session_id"],
    )


def extract_cases(row: dict) -> list[EvaluationCase]:
    """Evaluate each user turn against its one music target using past context only.

    Teacher-forced history includes only earlier user, music and assistant entries.
    Current/future assistant answers, hidden thoughts, progress assessments and
    session-level conversation goals/specificity never enter the model request.
    """
    turns = defaultdict(dict)
    for entry in row["conversations"]:
        number, role = entry["turn_number"], entry["role"]
        if type(number) is not int or number < 1:
            raise ValueError("Turn numbers must be positive integers.")
        if role not in {"user", "music", "assistant"} or role in turns[number]:
            raise ValueError(f"Unexpected or duplicate role at turn {number}: {role}")
        content = entry["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Missing {role} content at turn {number}.")
        turns[number][role] = content
    if sorted(turns) != list(range(1, 9)):
        raise ValueError("Expected all eight turns of a public test session.")
    profile = row.get("user_profile") or {}
    # Same profile fields as the app; omit split labels and other hidden fields.
    profile_text = "; ".join(
        f"{key.replace('_', ' ')}: {profile[key]}"
        for key in (
            "preferred_musical_culture",
            "preferred_language",
            "age_group",
            "country_name",
            "gender",
        )
        if profile.get(key)
    )
    history, cases = [], []
    for number, turn in sorted(turns.items()):
        if set(turn) != {"user", "music", "assistant"}:
            raise ValueError(f"Turn {number} must have one user, music and assistant entry.")
        request = RecommendationRequest(
            message=turn["user"],
            profile=profile_text,
            user_id=str(row["user_id"]),
            session_id=str(row["session_id"]),
            history=list(history),
            limit=20,
        )
        cases.append(
            EvaluationCase(
                session_id=str(row["session_id"]),
                user_id=str(row["user_id"]),
                turn_number=number,
                target_track_id=turn["music"],
                request=request,
            )
        )
        history.extend(
            [
                ChatTurn(role="user", content=turn["user"]),
                ChatTurn(
                    role="assistant",
                    content=f"Played track ID: {turn['music']}\n{turn['assistant']}",
                ),
            ]
        )
    return cases
