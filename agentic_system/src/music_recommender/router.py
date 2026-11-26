import logging
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Header, HTTPException, Request, Response

from .schemas import RecommendationRequest, RecommendationResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", response_model=RecommendationResponse)
@router.post("/recommendations", response_model=RecommendationResponse)
def recommend(
    payload: RecommendationRequest,
    request: Request,
    response: Response,
    conversation_id: Annotated[
        str | None, Header(alias="X-Conversation-ID", min_length=1, max_length=128)
    ] = None,
) -> RecommendationResponse:
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="Recommendation service is not ready.")

    thread_id = conversation_id or str(uuid4())
    response.headers["X-Conversation-ID"] = thread_id
    try:
        result = agent.invoke(
            {"request": payload, "notices": []},
            config={"configurable": {"thread_id": thread_id}},
        )
    except Exception as exc:
        logger.exception("Recommendation request failed")
        raise HTTPException(status_code=500, detail="Recommendation failed.") from exc

    return RecommendationResponse.model_validate(result["response"])
