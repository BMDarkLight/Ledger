"""Scoring functions — graded before they are used to grade anything else."""

from eval.metrics import CaseResult, Scorecard, recall_at_k


def test_recall_at_k_is_none_when_no_sources_are_expected():
    assert recall_at_k([], ["pep-0008"]) is None


def test_recall_at_k_is_partial_when_one_of_two_sources_lands():
    assert recall_at_k(["pep-0008", "pep-0572"], ["pep-0008", "pep-0020"]) == 0.5


def test_scorecard_reports_routing_and_refusal_separately():
    card = Scorecard(
        results=[
            CaseResult("R001", "retrieval", routed_correctly=True, behaved_correctly=True),
            CaseResult("A001", "adversarial", routed_correctly=True, behaved_correctly=False),
        ]
    )
    assert card.routing_accuracy == 1.0
    assert card.refusal_accuracy == 0.0, "adversarial behavior is graded on its own"


def test_markdown_lists_failing_cases():
    card = Scorecard(
        results=[
            CaseResult(
                "A004",
                "adversarial",
                routed_correctly=False,
                behaved_correctly=False,
                error="routed `retrieve`, expected `refuse`",
            )
        ]
    )
    rendered = card.to_markdown()
    assert "Failing cases" in rendered
    assert "A004" in rendered


def test_unmeasured_metrics_are_not_reported_as_zero():
    """Routing-only runs measure no recall. Rendering that as 0.0% would be a lie."""
    card = Scorecard(
        results=[CaseResult("R001", "retrieval", routed_correctly=True, behaved_correctly=True)]
    )
    assert card.recall is None
    assert "_not measured_" in card.to_markdown()
