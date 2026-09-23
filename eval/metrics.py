"""Scoring functions for the golden set. Pure, with no network and no API key."""

from dataclasses import dataclass, field


def recall_at_k(expected: list[str], retrieved: list[str]) -> float | None:
    """Share of expected source documents present in the retrieved set.

    Returns None for cases that expect no sources, so they can be excluded from
    the average instead of scoring a misleading 0.0 or 1.0.
    """
    if not expected:
        return None
    hit = sum(1 for doc in expected if doc in retrieved)
    return hit / len(expected)


def mean(values: list[float | None]) -> float:
    usable = [v for v in values if v is not None]
    return sum(usable) / len(usable) if usable else 0.0


def mean_or_none(values: list[float | None]) -> float | None:
    """Like `mean`, but keeps "scored zero" distinct from "never measured"."""
    usable = [v for v in values if v is not None]
    return sum(usable) / len(usable) if usable else None


def _why(result: "CaseResult") -> str:
    if result.error:
        return result.error
    if not result.routed_correctly:
        return "wrong route"
    if result.behaved_correctly is False:
        return "wrong behavior"
    return "wrong tool"


def _pct(value: float | None) -> str:
    """A metric nobody measured is reported as such, never as a zero."""
    return f"{value:.1%}" if value is not None else "_not measured_"


@dataclass
class CaseResult:
    case_id: str
    category: str
    routed_correctly: bool

    behaved_correctly: bool | None = None
    """None when this run could not observe the behavior at all.

    Most refusals are decided by the receipts gate rather than the router, so a
    routing-only run cannot see them. Scoring those as passes would report a
    refusal accuracy that no refusal produced.
    """

    tool_choice_correct: bool | None = None
    """None when the case expects no tool and none was chosen."""

    recall: float | None = None
    """Expected documents present in the dense top-k, before reranking."""

    recall_after_rerank: float | None = None
    """The same, after the cross-encoder cut it to top-n. A drop here means the
    reranker threw away a document the vector search had already found."""

    citation_coverage: float | None = None
    """Share of factual sentences carrying a valid receipt. Needs synthesis, and
    stays None for a refusal, which has no factual sentences to cover."""

    substrings_hit: float | None = None
    """Share of the case's `answer_contains` strings found in the answer."""

    error: str | None = None


@dataclass
class Scorecard:
    results: list[CaseResult] = field(default_factory=list)

    @property
    def cases_run(self) -> int:
        return len(self.results)

    @property
    def routing_accuracy(self) -> float:
        return mean([1.0 if r.routed_correctly else 0.0 for r in self.results])

    @property
    def refusal_accuracy(self) -> float | None:
        graded = [
            r
            for r in self.results
            if r.category == "adversarial" and r.behaved_correctly is not None
        ]
        return mean_or_none([1.0 if r.behaved_correctly else 0.0 for r in graded])

    @property
    def refusals_exercised(self) -> tuple[int, int]:
        """(observed, total) adversarial cases, the caveat next to the accuracy."""
        adversarial = [r for r in self.results if r.category == "adversarial"]
        observed = [r for r in adversarial if r.behaved_correctly is not None]
        return len(observed), len(adversarial)

    @property
    def tool_accuracy(self) -> float | None:
        """Did the router pick the right tool, not merely the right route."""
        graded = [r for r in self.results if r.tool_choice_correct is not None]
        return mean_or_none([1.0 if r.tool_choice_correct else 0.0 for r in graded])

    @property
    def recall(self) -> float | None:
        return mean_or_none([r.recall for r in self.results])

    @property
    def recall_after_rerank(self) -> float | None:
        return mean_or_none([r.recall_after_rerank for r in self.results])

    @property
    def citation_coverage(self) -> float | None:
        return mean_or_none([r.citation_coverage for r in self.results])

    @property
    def answer_match(self) -> float | None:
        """How much of each expected answer turned up in the prose.

        Coverage only says that the claims were cited. This says whether the
        question was answered.
        """
        return mean_or_none([r.substrings_hit for r in self.results])

    @property
    def cases_errored(self) -> int:
        return sum(1 for r in self.results if r.error and r.routed_correctly)

    def by_category(self) -> dict[str, float]:
        buckets: dict[str, list[float]] = {}
        for r in self.results:
            buckets.setdefault(r.category, []).append(1.0 if r.routed_correctly else 0.0)
        return {k: mean(v) for k, v in sorted(buckets.items())}

    def failures(self) -> list[CaseResult]:
        return [
            r
            for r in self.results
            if not r.routed_correctly
            or r.behaved_correctly is False
            or r.tool_choice_correct is False
            or r.error is not None
        ]

    def to_markdown(self, header: str = "") -> str:
        observed, total = self.refusals_exercised
        lines = [
            "# Ledger scorecard",
            "",
            header or "_Generated by `python -m eval.run_golden_set`._",
            "",
            "| Metric | Score |",
            "|---|---|",
            f"| Cases run | {self.cases_run} |",
            f"| Routing accuracy | {self.routing_accuracy:.1%} |",
            f"| Refusal accuracy (adversarial) | {_pct(self.refusal_accuracy)}"
            f" ({observed} of {total} cases observed) |",
            f"| Tool selection accuracy | {_pct(self.tool_accuracy)} |",
            f"| Retrieval recall@k (dense) | {_pct(self.recall)} |",
            f"| Recall after rerank | {_pct(self.recall_after_rerank)} |",
            f"| Citation coverage | {_pct(self.citation_coverage)} |",
            f"| Expected answer match | {_pct(self.answer_match)} |",
            "",
            "## Routing accuracy by category",
            "",
            "| Category | Score |",
            "|---|---|",
        ]
        lines += [f"| `{k}` | {v:.1%} |" for k, v in self.by_category().items()]
        failures = self.failures()
        if failures:
            lines += ["", "## Failing cases", "", "| Case | Category | Why |", "|---|---|---|"]
            lines += [f"| `{r.case_id}` | {r.category} | {_why(r)} |" for r in failures]
        return "\n".join(lines) + "\n"
