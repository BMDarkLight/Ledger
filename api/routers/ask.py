"""`POST /v1/ask` and `POST /v1/ask/receipts` — the full pipeline."""

from fastapi import APIRouter, HTTPException

from api.config import Settings
from api.deps import SettingsDep
from api.schemas import (
    AnswerStatus,
    AskRequest,
    AskResponse,
    ReceiptedAskResponse,
    Route,
)
from api.services import receipts as receipts_service
from api.services import retrieval, synthesis, tools
from api.services import router as routing
from api.services.retrieval import RetrievalUnavailable

router = APIRouter(prefix="/v1", tags=["ask"])


def _run_pipeline(request: AskRequest, settings: Settings) -> ReceiptedAskResponse:
    decision = routing.decide(request.question)

    if decision.route is Route.REFUSE:
        return ReceiptedAskResponse(
            answer=receipts_service.REFUSAL_TEXT,
            status=AnswerStatus.REFUSED,
            route=decision,
            receipts=[],
            claims=[],
            citation_coverage=0.0,
        )

    collected = []
    if decision.route in (Route.RETRIEVE, Route.RETRIEVE_THEN_TOOL):
        collected += retrieval.search(request.question, settings, request.top_k)
    if decision.route in (Route.TOOL, Route.RETRIEVE_THEN_TOOL):
        collected += tools.run_tools(
            [(name, request.question) for name in decision.tools], settings
        )

    if not collected:
        return ReceiptedAskResponse(
            answer=receipts_service.REFUSAL_TEXT,
            status=AnswerStatus.REFUSED,
            route=decision,
            receipts=[],
            claims=[],
            citation_coverage=0.0,
        )

    answer = synthesis.synthesize(request.question, collected, settings)
    if answer.strip() == synthesis.INSUFFICIENT:
        return ReceiptedAskResponse(
            answer=receipts_service.REFUSAL_TEXT,
            status=AnswerStatus.REFUSED,
            route=decision,
            receipts=collected,
            claims=[],
            citation_coverage=0.0,
        )

    claims = receipts_service.extract_claims(answer, collected)
    return ReceiptedAskResponse(
        answer=answer,
        status=receipts_service.decide_status(claims, collected, settings.citation_coverage_floor),
        route=decision,
        receipts=collected,
        claims=claims,
        citation_coverage=receipts_service.citation_coverage(claims),
    )


def _guarded(request: AskRequest, settings: Settings) -> ReceiptedAskResponse:
    """Run the pipeline, translating the two ways it can legitimately not answer.

    Both are deliberate: an unbuilt stage and an unreachable store must surface
    as themselves, never as a degraded answer.
    """
    try:
        return _run_pipeline(request, settings)
    except RetrievalUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, settings: SettingsDep) -> AskResponse:
    result = _guarded(request, settings)
    return AskResponse(**result.model_dump(exclude={"claims"}))


@router.post("/ask/receipts", response_model=ReceiptedAskResponse)
def ask_with_receipts(request: AskRequest, settings: SettingsDep) -> ReceiptedAskResponse:
    return _guarded(request, settings)
