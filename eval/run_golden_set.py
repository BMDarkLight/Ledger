"""Run the golden set and write eval/scorecard.md.

Three modes, picked automatically by what the environment can support:

  routing    Grades the deterministic baseline router alone. Needs nothing but
             this repo: no corpus, no models, no key.
  retrieval  Also ingests the corpus and measures recall@k before and after
             reranking. Needs the corpus on disk. Embeddings run locally, so
             still no API key. This is what CI runs on every pull request.
  full       Also generates answers and grades citation coverage, observed
             refusals and expected-answer match. Needs LLM_API_KEY.

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
from api.schemas import AnswerStatus, AskRequest, ReceiptedAskResponse
from api.services import router as routing
from api.services.receipts import strip_tags
from eval.golden_set import Behavior, Case, load_golden_set
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
    "full": (
        "full pipeline (retrieval, tools and synthesis). Routing numbers carry the "
        "same fitted-baseline caveat as routing-only mode. Refusals are observed "
        "here rather than inferred from the route, so every adversarial case is "
        "graded. A case listed below with a tool error reached the tool stage and "
        "could not run it, which is a failure of the tool rather than of the answer."
    ),
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
        chosen = decision.tools or []
        error = f"chose {chosen}, expected {case.expected_tools}"
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


def _grade_answer(case: Case, result: CaseResult, answer: ReceiptedAskResponse) -> None:
    """Grade what only a generated answer can show: refusal, coverage, content."""
    refused = answer.status is AnswerStatus.REFUSED
    should_refuse = case.expected_behavior is Behavior.REFUSE
    result.behaved_correctly = refused is should_refuse

    if refused:
        if not should_refuse:
            result.error = result.error or "refused a question it should have answered"
        return

    if should_refuse:
        result.error = result.error or "answered a question it should have refused"

    result.citation_coverage = answer.citation_coverage

    if case.answer_contains:
        prose = strip_tags(answer.answer).lower()
        missing = [n for n in case.answer_contains if n.lower() not in prose]
        result.substrings_hit = 1 - len(missing) / len(case.answer_contains)
        if missing and not result.error:
            result.error = f"answer is missing {missing}"

    if result.error is None and answer.status is AnswerStatus.UNVERIFIED:
        uncited = sum(1 for c in answer.claims if not c.supported)
        result.error = f"{uncited} claim(s) returned without a receipt"


def _measure_recall(case: Case, result: CaseResult, settings: Settings) -> None:
    """Recall at both retrieval stages, so a rerank drop is visible as one."""
    from api.services import retrieval

    if not case.expected_sources:
        return

    hits = retrieval.dense_search(case.question, settings, settings.retrieval_top_k)
    result.recall = recall_at_k(case.expected_sources, [h["doc_id"] for h in hits])

    ranked = retrieval.rerank(case.question, hits, settings)
    result.recall_after_rerank = recall_at_k(case.expected_sources, [h["doc_id"] for h in ranked])

    if result.error is None and result.recall_after_rerank == 0.0:
        found = "nothing" if result.recall == 0.0 else "dense search had it, rerank dropped it"
        result.error = f"expected {case.expected_sources} not retrieved: {found}"


def _ingest(settings: Settings) -> None:
    from scripts.ingest_corpus import ingest_corpus

    try:
        ingest_corpus(settings)
    except Exception as exc:  # model download, or an unreachable vector store
        raise RuntimeError(
            f"this mode could not start: {exc}\n"
            "Embedding models are downloaded from Hugging Face on first use, and "
            "unauthenticated downloads are rate limited. Set HF_TOKEN, or set "
            "MODEL_CACHE_DIR to a warmed cache. To skip retrieval entirely, run "
            "with --mode routing."
        ) from exc


def _answer_case(case: Case, result: CaseResult, settings: Settings) -> None:
    """Put one case through the pipeline and grade what comes back.

    A case that cannot reach an answer records why and keeps its routing score.
    One unbuilt tool must not take the other thirty cases down with it.
    """
    from api.routers.ask import run_pipeline
    from api.services.retrieval import RetrievalUnavailable
    from api.services.synthesis import SynthesisError
    from api.services.tools import ToolError

    try:
        answer = run_pipeline(AskRequest(question=case.question), settings)
    except (RetrievalUnavailable, SynthesisError, ToolError, NotImplementedError) as exc:
        result.error = result.error or f"{type(exc).__name__}: {exc}"
    else:
        _grade_answer(case, result, answer)


def run(mode: str, settings: Settings, cases: list[Case] | None = None) -> Scorecard:
    """Grade `cases` (the whole golden set by default) at the given mode."""
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode!r}")

    cases = load_golden_set() if cases is None else cases
    if mode != "routing":
        _ingest(settings)

    card = Scorecard()
    for case in cases:
        result, _ = _grade_routing(case)
        if mode != "routing":
            _measure_recall(case, result, settings)
        if mode == "full":
            _answer_case(case, result, settings)
        card.results.append(result)
    return card


def caption(mode: str, at: datetime) -> str:
    stamp = at.strftime("%Y-%m-%d %H:%M UTC")
    return f"_Mode: {MODE_CAPTIONS[mode]} Generated {stamp} by `python -m eval.run_golden_set`._"


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

    card = run(mode, settings)
    header = caption(mode, datetime.now(timezone.utc))
    args.out.write_text(card.to_markdown(header), encoding="utf-8")

    print(card.to_markdown(header))
    print(f"wrote {args.out} (mode: {mode})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
