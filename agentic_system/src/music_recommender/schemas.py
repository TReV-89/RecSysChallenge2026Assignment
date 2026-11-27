from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Specificity = Literal["HH", "HL", "LH", "LL"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ChatTurn(StrictModel):
    role: Literal["user", "assistant"]
    content: str


class RecommendationRequest(StrictModel):
    message: str = Field(min_length=1, max_length=6000)
    profile: str = Field(default="", max_length=6000)
    goal: str = Field(default="", max_length=3000)
    specificity: Specificity | None = None
    user_id: str | None = None
    session_id: str | None = None
    reference_track_id: str | None = None
    history: list[ChatTurn] = Field(default_factory=list)
    excluded_track_ids: list[str] = Field(default_factory=list)
    limit: int = Field(default=5, ge=1, le=20)


class Neighbor(StrictModel):
    user_id: str
    score: float
    source: Literal["cf_bpr"] = "cf_bpr"


class Track(StrictModel):
    track_id: str
    title: str
    artists: list[str]
    tags: list[str]


class SupportingUser(StrictModel):
    user_id: str
    similarity: float
    evidence_type: Literal["positive_conversation_feedback"] = "positive_conversation_feedback"


class Candidate(StrictModel):
    track: Track
    query_score: float
    neighbor_score: float
    supporting_users: list[SupportingUser]
    source_ranks: dict[str, int] = Field(default_factory=dict)
    fusion_score: float | None = None


class RetrievalResult(StrictModel):
    query: str
    neighbors: list[Neighbor]
    candidates: list[Candidate]


class RankedTrack(StrictModel):
    candidate: Candidate
    score: float


class Recommendation(StrictModel):
    track: Track
    score: float
    explanation: str
    supporting_users: list[SupportingUser]


class RecommendationResponse(StrictModel):
    kind: Literal["chat", "recommendations"] = "recommendations"
    message: str = ""
    recommendations: list[Recommendation]
    neighbors: list[Neighbor]
    notices: list[str]
    mode: Literal["groq", "openrouter", "fallback"]


class TrackExplanation(StrictModel):
    track_id: str
    explanation: str
    cited_tags: list[str]


class ExplanationDraft(StrictModel):
    tracks: list[TrackExplanation]
