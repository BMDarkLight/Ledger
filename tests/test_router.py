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
