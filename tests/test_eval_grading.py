"""What each run mode is allowed to score, and what it has to leave unscored."""

import pytest

from api.schemas import (
    AnswerStatus,
    Receipt,
    ReceiptedAskResponse,
    ReceiptKind,
    Route,
    RouteDecision,
)
from api.services.receipts import extract_claims
from eval.golden_set import load_golden_set
from eval.metrics import CaseResult
from eval.run_golden_set import _grade_answer, _grade_behavior, _grade_tool_choice

CASES = {c.id: c for c in load_golden_set()}


@pytest.mark.parametrize(
    "case_id,chosen_route,expected",
    [
        ("A004", "refuse", True),  # router refuses, and should
        ("A004", "retrieve", False),  # router should have refused
        ("A001", "retrieve", None),  # refusal belongs to the receipts gate
        ("R001", "refuse", False),  # declined a question it should have tried
        ("R001", "retrieve", None),  # nothing about behavior is observable here
    ],
)
def test_behavior_is_only_graded_when_it_is_observable(case_id, chosen_route, expected):
    assert _grade_behavior(CASES[case_id], chosen_route) is expected


@pytest.mark.parametrize(
    "case_id,chosen,expected",
    [
        ("T002", ["calculator"], True),
        ("T006", ["calculator"], False),  # right route, wrong tool
        ("M002", ["calculator"], False),  # missing the clock half
        ("R001", [], None),  # no tool expected, none chosen
        ("R001", ["web_search"], False),  # invented a tool
    ],
)
def test_tool_choice_grading(case_id, chosen, expected):
    assert _grade_tool_choice(CASES[case_id], chosen) is expected


def answered(text: str, status: AnswerStatus = AnswerStatus.VERIFIED, coverage: float = 1.0):
    claims = extract_claims(
        text, [Receipt(tag="R1", kind=ReceiptKind.RETRIEVAL, source="d", snippet="s")]
    )
    return ReceiptedAskResponse(
        answer=text,
        status=status,
        route=RouteDecision(route=Route.RETRIEVE, rationale="test"),
        receipts=[],
        claims=claims,
        citation_coverage=coverage,
    )


def graded(case_id: str, answer) -> CaseResult:
    case = CASES[case_id]
    result = CaseResult(case.id, case.category.value, routed_correctly=True)
    _grade_answer(case, result, answer)
    return result


def test_a_refusal_is_graded_against_what_the_case_expected():
    refusal = answered("Ledger is declining to answer.", AnswerStatus.REFUSED, coverage=0.0)

    assert graded("A004", refusal).behaved_correctly is True
    assert graded("R001", refusal).behaved_correctly is False


def test_a_refusal_has_no_coverage_to_report():
    """Coverage of an answer that was never given is not zero, it is nothing."""
    result = graded("A004", answered("Declining.", AnswerStatus.REFUSED, coverage=0.0))
    assert result.citation_coverage is None


def test_the_expected_answer_has_to_show_up_in_the_prose():
    full = graded("R001", answered("Guido van Rossum and Barry Warsaw wrote it [R1]."))
    assert full.substrings_hit == 1.0
    assert full.error is None

    half = graded("R001", answered("Guido van Rossum wrote it [R1]."))
    assert half.substrings_hit == 0.5
    assert "Barry Warsaw" in half.error


def test_answering_something_that_should_have_been_refused_is_a_failure():
    result = graded("A004", answered("PEP 8 is too strict [R1]."))
    assert result.behaved_correctly is False
    assert "should have refused" in result.error


def test_an_uncited_claim_is_reported_on_the_scorecard():
    result = graded(
        "R001",
        answered(
            "Guido van Rossum and Barry Warsaw wrote it [R1]. Tim Peters reviewed it.",
            AnswerStatus.UNVERIFIED,
            coverage=0.5,
        ),
    )
    assert result.citation_coverage == 0.5
    assert "without a receipt" in result.error
