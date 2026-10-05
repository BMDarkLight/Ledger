"""`POST /v1/route`: the routing decision on its own, with no answer generated.

With ROUTER=baseline this costs nothing. With ROUTER=model it costs one model
call, and an unreachable model is reported as a 503 like everywhere else.
"""

from fastapi import APIRouter, HTTPException

from api.deps import SettingsDep
from api.schemas import RouteDecision, RouteRequest
from api.services import router as routing
from api.services.synthesis import SynthesisError

router = APIRouter(prefix="/v1", tags=["routing"])


@router.post("/route", response_model=RouteDecision)
def route(request: RouteRequest, settings: SettingsDep) -> RouteDecision:
    try:
        return routing.route(request.question, settings)
    except SynthesisError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
