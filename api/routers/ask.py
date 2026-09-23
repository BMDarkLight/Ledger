"""`POST /v1/ask` and `POST /v1/ask/receipts`: the full pipeline."""

from fastapi import APIRouter, HTTPException

from api.config import Settings
from api.deps import SettingsDep
from api.schemas import (
    AnswerStatus,
    AskRequest,
    AskResponse,
    Receipt,
    ReceiptedAskResponse,
    Route,
)
from api.services import receipts as receipts_service
from api.services import retrieval, synthesis, tools
from api.services import router as routing
from api.services.retrieval import RetrievalUnavailable
from api.services.synthesis import SynthesisError

router = APIRouter(prefix="/v1", tags=["ask"])


def _refusal(decision, collected: list[Receipt]) -> ReceiptedAskResponse:
    return ReceiptedAskResponse(
        answer=receipts_service.REFUSAL_TEXT,
        status=AnswerStatus.REFUSED,
        route=decision,
        receipts=collected,
        claims=[],
        citation_coverage=0.0,
    )


def _tool_argument(name: str, question: str, evidence: list[Receipt], settings: Settings) -> str:
    """Work out what to pass a tool.

    Questions that write their own numbers out are handled without a model call.
    A chained question is not: the figure it needs was in the corpus, so the
    argument can only be written once retrieval has run.
    """
    try:
        return tools.argument_for(name, question)
    except tools.ToolError:
        if not settings.llm_configured:
            raise
        planned = synthesis.plan_tool_call(question, name, tools.describe(name), evidence, settings)
        if planned is None:
            raise
        return planned


def run_pipeline(request: AskRequest, settings: Settings) -> ReceiptedAskResponse:
    """Route, gather evidence, generate, then check the generation against it."""
    decision = routing.decide(request.question)

    if decision.route is Route.REFUSE:
        return _refusal(decision, [])

    collected: list[Receipt] = []
    if decision.route in (Route.RETRIEVE, Route.RETRIEVE_THEN_TOOL):
        collected += retrieval.search(request.question, settings, request.top_k)
    if decision.route in (Route.TOOL, Route.RETRIEVE_THEN_TOOL):
        collected += tools.run_tools(
            [
                (name, _tool_argument(name, request.question, collected, settings))
                for name in decision.tools
            ],
            settings,
        )

    if not collected:
        return _refusal(decision, [])

    answer = synthesis.synthesize(request.question, collected, settings)
    if answer.strip() == synthesis.INSUFFICIENT:
        return _refusal(decision, collected)

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
    """Run the pipeline, translating the ways it can legitimately not answer.

    An unbuilt stage, an unreachable store and a tool that cannot run all have
    to surface as themselves. None of them may come back as a degraded answer.
    """
    try:
        return run_pipeline(request, settings)
    except (RetrievalUnavailable, SynthesisError, tools.ToolError) as exc:
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
