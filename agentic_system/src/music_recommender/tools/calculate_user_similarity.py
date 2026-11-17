"""Similarity calculation used internally by catalog retrieval."""

from langchain_core.tools import tool

from ..schemas import SimilarityWeights


def weighted_similarity(profile, goal, specificity, weights: SimilarityWeights) -> float:
    pairs = [(profile, weights.profile), (goal, weights.goal), (specificity, weights.specificity)]
    denominator = sum(weight for score, weight in pairs if score is not None)
    if denominator == 0:
        return 0.0
    return sum(score * weight for score, weight in pairs if score is not None) / denominator


@tool
def calculate_user_similarity(
    profile_score: float | None,
    goal_score: float | None,
    specificity_score: float | None,
    weights: SimilarityWeights,
) -> float:
    """Combine profile, goal and specificity similarities for a training listener.

    Args:
        profile_score: Profile similarity in [0, 1], or None when unavailable.
        goal_score: Goal similarity in [0, 1], or None when unavailable.
        specificity_score: Exact label match (1 or 0), or None when unspecified.
        weights: Nonnegative profile, goal and specificity weights with at least
            one positive value. Remaining weights are renormalized when a score
            is missing. Components outside [0, 1] raise ValueError.

    Returns:
        A float in [0, 1]: the weighted mean of available components, or 0.0
        when none has positive weight. This is a relative similarity score,
        not a probability or evidence of listening. Retrieval calls this tool
        internally using computed scores; the main agent need not call it.
    """
    for value in (profile_score, goal_score, specificity_score):
        if value is not None and not 0 <= value <= 1:
            raise ValueError("Similarity components must be between zero and one.")
    return weighted_similarity(profile_score, goal_score, specificity_score, weights)
