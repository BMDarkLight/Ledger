"""How a routing-only run decides what it is and isn't allowed to score."""

import pytest

from eval.golden_set import load_golden_set
from eval.run_golden_set import _grade_behavior, _grade_tool_choice

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
