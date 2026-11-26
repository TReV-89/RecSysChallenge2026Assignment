"""Agent tool for candidate tracks supported by similar training users."""

from typing import Annotated

from langchain_core.tools import InjectedToolArg, tool

from ..schemas import RecommendationRequest, RetrievalResult


@tool
def retrieve_user_to_user_candidates(
    request: Annotated[RecommendationRequest, InjectedToolArg],
    retriever: Annotated[object, InjectedToolArg],
) -> RetrievalResult:
    """Find tracks supported by similar users' eligible training feedback.

    Call for a recommendation request when the application has a user ID and
    a CF-BPR vector for that listener. The application supplies the listener
    identity and data; do not guess or substitute a user ID. Similarity uses
    the prepared CF-BPR user vectors. A track receives support only from
    eligible positive feedback in the training conversations.

    Args:
        request: Application-provided listener identity and excluded tracks.
        retriever: Application-provided CF retriever and prepared data access.
            The model cannot set either argument.

    Returns:
        RetrievalResult containing similar training users and catalog-backed
        candidate tracks ordered by their neighbor support. Each candidate
        has a user-to-user source rank and bounded supporting-user evidence.
        The result is empty when the listener has no usable CF-BPR vector or
        no eligible neighbor tracks. Feedback is not a recorded listen or like.
    """
    return retriever.retrieve_cf(request)
