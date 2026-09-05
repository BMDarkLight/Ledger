"""`POST /v1/eval/run` — run the golden set on demand."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from eval.golden_set import Category, load_golden_set

router = APIRouter(prefix="/v1/eval", tags=["eval"])


class EvalRequest(BaseModel):
    categories: list[Category] | None = Field(
        default=None, description="Restrict the run to these categories. None means all."
    )
    limit: int | None = Field(default=None, ge=1)


class EvalSummary(BaseModel):
    cases_run: int
    routing_accuracy: float
    refusal_accuracy: float
    citation_coverage: float | None = None
    scorecard_markdown: str


@router.post("/run", response_model=EvalSummary)
def run(request: EvalRequest) -> EvalSummary:
    cases = load_golden_set()
    if request.categories:
        wanted = set(request.categories)
        cases = [c for c in cases if c.category in wanted]
    if request.limit:
        cases = cases[: request.limit]
    raise HTTPException(
        status_code=501,
        detail=(
            f"Phase 1: wire the harness to the live pipeline. "
            f"{len(cases)} cases are loaded and ready."
        ),
    )
