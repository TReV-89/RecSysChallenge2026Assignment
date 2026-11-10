from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Specificity = Literal["HH", "HL", "LH", "LL"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class SimilarityWeights(StrictModel):
    profile: float = Field(default=0.5, ge=0, le=1)
    goal: float = Field(default=0.35, ge=0, le=1)
    specificity: float = Field(default=0.15, ge=0, le=1)

    @model_validator(mode="after")
    def nonzero(self):
        if self.profile + self.goal + self.specificity <= 0:
            raise ValueError("At least one similarity weight must be positive.")
        return self


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
    history: list[ChatTurn] = Field(default_factory=list)
    excluded_track_ids: list[str] = Field(default_factory=list)
    weights: SimilarityWeights = Field(default_factory=SimilarityWeights)
    query_weight: float = Field(default=0.75, ge=0, le=1)
    limit: int = Field(default=5, ge=1, le=10)


class Neighbor(StrictModel):
    user_id: str
    session_id: str
    score: float
    profile_score: float | None
    goal_score: float | None
    specificity_score: float | None


class Track(StrictModel):
    track_id: str
    title: str
    artists: list[str]
    tags: list[str]


class SupportingUser(StrictModel):
    user_id: str
    similarity: float
    track_affinity: float
    evidence_type: Literal["semantic_affinity"] = "semantic_affinity"


class Candidate(StrictModel):
    track: Track
    query_score: float
    neighbor_score: float
    supporting_users: list[SupportingUser]


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
    recommendations: list[Recommendation]
    neighbors: list[Neighbor]
    notices: list[str]
    mode: Literal["groq", "local", "fallback"]


class SearchArguments(StrictModel):
    query: str = Field(
        min_length=1,
        max_length=2000,
        description="Music search incorporating current request and relevant prior preferences.",
    )


class RankingArguments(StrictModel):
    exclude_track_ids: list[str] = Field(
        description="Candidate IDs clearly conflicting with the user's explicit preferences. Empty if none."
    )


class TrackExplanation(StrictModel):
    track_id: str
    explanation: str
    cited_tags: list[str]


class ExplanationDraft(StrictModel):
    tracks: list[TrackExplanation]
