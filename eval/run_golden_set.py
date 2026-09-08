"""Run the golden set and write eval/scorecard.md.

Three modes, picked automatically by what the environment can actually support:

  routing    Grades the deterministic baseline router alone. Needs nothing but
             this repo — no corpus, no models, no key.
  retrieval  Also ingests the corpus and measures recall@k before and after
             reranking. Needs the corpus on disk; embeddings run locally, so
             still no API key. This is what CI runs on every pull request.
  full       Also runs synthesis and grades citation coverage. Needs LLM_API_KEY.

Usage:
    python -m eval.run_golden_set                    # best mode available
    python -m eval.run_golden_set --mode routing
    python -m eval.run_golden_set --out /tmp/card.md
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from api.config import Settings, get_settings
from api.services import router as routing
from eval.golden_set import Case, load_golden_set
from eval.metrics import CaseResult, Scorecard, recall_at_k

MODES = ("routing", "retrieval", "full")

MODE_CAPTIONS = {
    "routing": (
        "routing-only (deterministic baseline router, no corpus, no LLM calls). "
        "Read this number with suspicion: the baseline's patterns were written "
        "against these same questions, so it is fitted to them and its score is "
        "an upper bound, not a generalization estimate."
    ),
    "retrieval": (
        "retrieval (local ONNX embeddings, no LLM calls). Routing numbers carry the "
        "same fitted-baseline caveat as routing-only mode. Citation coverage still "
        "needs synthesis, so it stays unmeasured here."
    ),
    "full": "full pipeline (retrieval, tools and synthesis).",
}


def _grade_behavior(case: Case, chosen_route: str) -> bool | None:
    """Whether this run can see the case behave, and whether it behaved.

    Returns None when the answer is "we cannot tell from here". Most adversarial
    cases are supposed to be refused by the receipts gate after retrieval comes
    back empty-handed, which a run without synthesis never reaches. Scoring those
    as passes is how a scorecard ends up reporting 100% refusal accuracy without
    a single refusal having happened.
    """
    refused = chosen_route == "refuse"
    if case.expected_route == "refuse":
        return refused
    if refused:
        # Observable failure: it declined something it was supposed to attempt.
        return False
    return None


def _grade_tool_choice(case: Case, chosen_tools: list[str]) -> bool | None:
    """Routing accuracy hides this: the right route with the wrong tool."""
    if case.expected_tools:
        return set(chosen_tools) == set(case.expected_tools)
    if chosen_tools:
        return False
    return None


def _grade_routing(case: Case) -> tuple[CaseResult, list[str]]:
    """Everything gradeable without touching the corpus."""
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

    return (
        CaseResult(
            case_id=case.id,
            category=case.category.value,
            routed_correctly=routed,
            behaved_correctly=behaved,
            tool_choice_correct=tool_ok,
            error=error,
        ),
        decision.tools,
    )


def run_routing() -> Scorecard:
    card = Scorecard()
    for case in load_golden_set():
        result, _ = _grade_routing(case)
        card.results.append(result)
    return card


def run_retrieval(settings: Settings) -> Scorecard:
    """Routing grading, plus recall measured at both retrieval stages."""
    from api.services import retrieval
    from scripts.ingest_corpus import ingest_corpus

    try:
        ingest_corpus(settings)
    except Exception as exc:  # model download, or an unreachable vector store
        raise RuntimeError(
            f"retrieval mode could not start: {exc}\n"
            "Embedding models are downloaded from Hugging Face on first use, and "
            "unauthenticated downloads are rate limited — set HF_TOKEN, or set "
            "MODEL_CACHE_DIR to a warmed cache. To skip retrieval entirely, run "
            "with --mode routing."
        ) from exc

    card = Scorecard()
    for case in load_golden_set():
        result, _ = _grade_routing(case)

        if case.expected_sources:
            hits = retrieval.dense_search(case.question, settings, settings.retrieval_top_k)
            result.recall = recall_at_k(case.expected_sources, [h["doc_id"] for h in hits])

            ranked = retrieval.rerank(case.question, hits, settings)
            result.recall_after_rerank = recall_at_k(
                case.expected_sources, [h["doc_id"] for h in ranked]
            )

            if result.error is None and result.recall_after_rerank == 0.0:
                found = (
                    "nothing" if result.recall == 0.0 else "dense search had it, rerank dropped it"
                )
                result.error = f"expected {case.expected_sources} not retrieved — {found}"

        card.results.append(result)
    return card


def run_full(settings: Settings) -> Scorecard:
    raise NotImplementedError(
        "Phase 1b: run each case through /v1/ask/receipts and grade citation "
        "coverage and faithfulness."
    )


def choose_mode(settings: Settings, requested: str | None) -> str:
    """Pick the richest mode the environment can actually support."""
    if requested:
        return requested
    from scripts.fetch_corpus import corpus_path, referenced_pep_ids

    corpus_ready = all(corpus_path(d).exists() for d in referenced_pep_ids())
    if settings.llm_configured and corpus_ready:
        return "full"
    return "retrieval" if corpus_ready else "routing"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, default=None)
    parser.add_argument("--out", type=Path, default=Path("eval/scorecard.md"))
    args = parser.parse_args(argv)

    settings = get_settings()
    mode = choose_mode(settings, args.mode)

    if mode == "routing":
        card = run_routing()
    elif mode == "retrieval":
        card = run_retrieval(settings)
    else:
        card = run_full(settings)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = (
        f"_Mode: {MODE_CAPTIONS[mode]} · generated {stamp} by `python -m eval.run_golden_set`._"
    )
    args.out.write_text(card.to_markdown(header), encoding="utf-8")

    print(card.to_markdown(header))
    print(f"→ wrote {args.out} (mode: {mode})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
