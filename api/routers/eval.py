"""`POST /v1/eval/run`: run the golden set on demand.

The run is synchronous. In full mode it makes one generation call per case, so
a whole-set run takes minutes. `categories` and `limit` are there to keep an
exploratory run short.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.deps import SettingsDep
from eval.golden_set import Category, load_golden_set
from eval.run_golden_set import MODES, caption, choose_mode, run

router = APIRouter(prefix="/v1/eval", tags=["eval"])


class EvalRequest(BaseModel):
    categories: list[Category] | None = Field(
        default=None, description="Restrict the run to these categories. None means all."
    )
    limit: int | None = Field(default=None, ge=1)
    mode: str | None = Field(
        default=None,
        description=f"One of {', '.join(MODES)}. None picks the richest mode available.",
    )


class EvalSummary(BaseModel):
    mode: str
    cases_run: int
    routing_accuracy: float
    refusal_accuracy: float | None = None
    citation_coverage: float | None = None
    scorecard_markdown: str


@router.post("/run", response_model=EvalSummary)
def run_eval(request: EvalRequest, settings: SettingsDep) -> EvalSummary:
    cases = load_golden_set()
    if request.categories:
        wanted = set(request.categories)
        cases = [c for c in cases if c.category in wanted]
    if request.limit:
        cases = cases[: request.limit]
    if not cases:
        raise HTTPException(status_code=400, detail="no cases match that filter")

    mode = request.mode or choose_mode(settings, None)
    if mode not in MODES:
        raise HTTPException(status_code=400, detail=f"unknown mode: {mode!r}")

    try:
        card = run(mode, settings, cases)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return EvalSummary(
        mode=mode,
        cases_run=card.cases_run,
        routing_accuracy=card.routing_accuracy,
        refusal_accuracy=card.refusal_accuracy,
        citation_coverage=card.citation_coverage,
        scorecard_markdown=card.to_markdown(caption(mode, datetime.now(timezone.utc))),
    )
