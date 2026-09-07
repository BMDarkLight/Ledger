"""Run the golden set and write eval/scorecard.md.

Two modes, chosen automatically:

  routing-only  No LLM_API_KEY, or --routing-only. Grades the router alone,
                which is deterministic and needs nothing but this repo. This is
                what CI runs on every PR.
  full          Runs the whole pipeline per case. Lands with Phase 1.

Usage:
    python -m eval.run_golden_set [--routing-only] [--out eval/scorecard.md]
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from api.config import get_settings
from api.services import router as routing
from eval.golden_set import load_golden_set
from eval.metrics import CaseResult, Scorecard


def _grade_behavior(case, chosen_route: str) -> bool | None:
    """Whether this run can see the case behave, and whether it behaved.

    Returns None when the answer is "we cannot tell from here". Most adversarial
    cases are supposed to be refused by the receipts gate after retrieval comes
    back empty-handed, which a routing-only run never reaches. Scoring those as
    passes is how a scorecard ends up reporting 100% refusal accuracy without a
    single refusal having happened.
    """
    refused = chosen_route == "refuse"
    if case.expected_route == "refuse":
        return refused
    if refused:
        # Observable failure: it declined something it was supposed to attempt.
        return False
    return None


def _grade_tool_choice(case, chosen_tools: list[str]) -> bool | None:
    """Routing accuracy hides this: the right route with the wrong tool."""
    if case.expected_tools:
        return set(chosen_tools) == set(case.expected_tools)
    if chosen_tools:
        return False
    return None


def run_routing_only() -> Scorecard:
    card = Scorecard()
    for case in load_golden_set():
        decision = routing.decide(case.question)
        routed = decision.route.value == case.expected_route
        behaved = _grade_behavior(case, decision.route.value)
        tool_ok = _grade_tool_choice(case, decision.tools)

        if not routed:
            error = f"routed `{decision.route.value}`, expected `{case.expected_route}`"
        elif behaved is False:
            error = "refused a question it should have attempted"
        elif tool_ok is False:
            error = f"chose {decision.tools or '[]'}, expected {case.expected_tools}"
        else:
            error = None

        card.results.append(
            CaseResult(
                case_id=case.id,
                category=case.category.value,
                routed_correctly=routed,
                behaved_correctly=behaved,
                tool_choice_correct=tool_ok,
                error=error,
            )
        )
    return card


def run_full() -> Scorecard:
    raise NotImplementedError(
        "Phase 1: run each case through /v1/ask/receipts and grade recall, "
        "citation coverage and faithfulness."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routing-only", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("eval/scorecard.md"))
    args = parser.parse_args(argv)

    settings = get_settings()
    routing_only = args.routing_only or not settings.llm_configured

    if routing_only:
        card = run_routing_only()
        mode = (
            "routing-only (deterministic baseline router, no LLM calls). "
            "Read this number with suspicion: the baseline's patterns were written "
            "against these same questions, so it is fitted to them and its score is "
            "an upper bound, not a generalization estimate."
        )
    else:
        card = run_full()
        mode = "full pipeline"

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = f"_Mode: {mode} · generated {stamp} by `python -m eval.run_golden_set`._"
    args.out.write_text(card.to_markdown(header), encoding="utf-8")

    print(card.to_markdown(header))
    print(f"→ wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
