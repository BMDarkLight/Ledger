"""Scoring functions, graded before they are used to grade anything else."""

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


def test_unobserved_refusals_are_excluded_not_counted_as_passes():
    """The bug this guards: 100% refusal accuracy with zero refusals observed.

    Most adversarial cases are refused by the receipts gate, which a routing-only
    run never reaches. Those must not be scored at all.
    """
    card = Scorecard(
        results=[
            CaseResult("A001", "adversarial", routed_correctly=True, behaved_correctly=None),
            CaseResult("A004", "adversarial", routed_correctly=True, behaved_correctly=True),
        ]
    )
    assert card.refusals_exercised == (1, 2)
    assert card.refusal_accuracy == 1.0
    assert "(1 of 2 cases observed)" in card.to_markdown()


def test_refusal_accuracy_is_unmeasured_when_nothing_was_observed():
    card = Scorecard(
        results=[CaseResult("A001", "adversarial", routed_correctly=True, behaved_correctly=None)]
    )
    assert card.refusal_accuracy is None


def test_tool_choice_failures_are_visible_even_when_the_route_is_right():
    card = Scorecard(
        results=[
            CaseResult("T006", "tool", routed_correctly=True, tool_choice_correct=False),
            CaseResult("T002", "tool", routed_correctly=True, tool_choice_correct=True),
        ]
    )
    assert card.routing_accuracy == 1.0, "routing alone would call this a clean sweep"
    assert card.tool_accuracy == 0.5
    assert "T006" in card.to_markdown()


def test_expected_answer_match_is_averaged_only_over_cases_that_asserted_one():
    card = Scorecard(
        results=[
            CaseResult("R001", "retrieval", routed_correctly=True, substrings_hit=1.0),
            CaseResult("R002", "retrieval", routed_correctly=True, substrings_hit=0.5),
            CaseResult("A004", "adversarial", routed_correctly=True),
        ]
    )
    assert card.answer_match == 0.75


def test_a_case_that_could_not_run_is_listed_rather_than_averaged_away():
    """A tool that will not start is a result. Dropping it flatters the run."""
    card = Scorecard(
        results=[
            CaseResult(
                "T006",
                "tool",
                routed_correctly=True,
                tool_choice_correct=True,
                error="NotImplementedError: Phase 2: sandboxed code execution.",
            )
        ]
    )
    assert card.citation_coverage is None
    assert "T006" in card.to_markdown()
    assert card.cases_errored == 1
