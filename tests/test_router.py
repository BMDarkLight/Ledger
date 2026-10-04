"""The deterministic baseline router, the floor Phase 2 has to beat."""

import pytest

from api.schemas import Route
from api.services import router as routing


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Who are the authors of PEP 8?", Route.RETRIEVE),
        ("What is 1327 * 4519?", Route.TOOL),
        ("What is the current price of Bitcoin in USD?", Route.TOOL),
        ("In your opinion, is PEP 8 too strict?", Route.REFUSE),
        ("Will Python ever remove the GIL entirely?", Route.REFUSE),
        (
            "PEP 8 recommends a maximum line length. How many characters over that "
            "limit is a 100-character line?",
            Route.RETRIEVE_THEN_TOOL,
        ),
    ],
)
def test_baseline_routing(question, expected):
    assert routing.decide(question).route is expected


def test_every_decision_carries_a_rationale():
    for question in ("Who wrote PEP 20?", "What is 2 + 2?", "Should I use PEP 8?"):
        assert routing.decide(question).rationale.strip()


# Held out: none of these are in the golden set. The baseline is fitted to that
# set, so its rules only count as fixes if they also hold on questions like these.
@pytest.mark.parametrize(
    "question,route,tools",
    [
        # beyond the calculator's operators -> real code
        ("What is the cube root of 27?", Route.TOOL, ["code_exec"]),
        ("What is 12 factorial?", Route.TOOL, ["code_exec"]),
        ("What is 2 ** 10?", Route.TOOL, ["calculator"]),
        # elapsed time to now needs the clock as well as arithmetic
        ("How many days ago was PEP 8 created?", Route.RETRIEVE_THEN_TOOL, ["clock", "calculator"]),
        ("How long ago was PEP 20 written?", Route.RETRIEVE_THEN_TOOL, ["clock", "calculator"]),
        # a count with operands is arithmetic; one without is a fact about the world
        ("How many minutes are in 3 hours?", Route.TOOL, ["calculator"]),
        (
            "How many PEPs have been rejected since PEP 572?",
            Route.RETRIEVE_THEN_TOOL,
            ["web_search"],
        ),
        ("How many moons does Jupiter have?", Route.TOOL, ["web_search"]),
        # a PEP number is a name, not an operand; the span is what makes it arithmetic
        ("How many years separate PEP 8 and PEP 20?", Route.RETRIEVE_THEN_TOOL, ["calculator"]),
    ],
)
def test_routing_rules_generalize_beyond_the_golden_set(question, route, tools):
    decision = routing.decide(question)
    assert (decision.route, decision.tools) == (route, tools)
