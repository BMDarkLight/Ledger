"""`POST /v1/route`: the routing decision on its own, with no generation spent."""

from fastapi import APIRouter

from api.schemas import RouteDecision, RouteRequest
from api.services import router as routing

router = APIRouter(prefix="/v1", tags=["routing"])


@router.post("/route", response_model=RouteDecision)
def route(request: RouteRequest) -> RouteDecision:
    return routing.decide(request.question)
