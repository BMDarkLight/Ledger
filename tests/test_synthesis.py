"""The generation step, exercised without a network or an API key."""

import pytest

from api.config import Settings
from api.schemas import Receipt, ReceiptKind
from api.services import synthesis


def retrieval_receipt(tag: str = "R1", snippet: str = "Author: Guido van Rossum") -> Receipt:
    return Receipt(tag=tag, kind=ReceiptKind.RETRIEVAL, source="pep-0008", snippet=snippet)


@pytest.fixture
def configured() -> Settings:
    return Settings(llm_api_key="test-key")


def reply_with(monkeypatch, text: str) -> list[list[dict[str, str]]]:
    """Replace the model call with a fixed reply, recording what was sent."""
    sent: list[list[dict[str, str]]] = []

    def fake(messages, settings):
        sent.append(messages)
        return text

    monkeypatch.setattr(synthesis, "_complete", fake)
    return sent


def test_receipts_are_rendered_with_their_tags():
    block = synthesis.format_receipts([retrieval_receipt("R2")])
    assert "[R2]" in block
    assert "pep-0008" in block
    assert "Guido van Rossum" in block


def test_the_prompt_carries_the_question_and_the_receipts():
    messages = synthesis.build_messages("Who wrote PEP 8?", [retrieval_receipt()])
    assert messages[0]["role"] == "system"
    assert "Who wrote PEP 8?" in messages[1]["content"]
    assert "[R1]" in messages[1]["content"]


def test_no_receipts_means_no_model_call(configured):
    """Nothing to cite, so there is nothing to generate from."""
    assert synthesis.synthesize("Who wrote PEP 8?", [], configured) == synthesis.INSUFFICIENT


def test_a_missing_key_is_reported_rather_than_worked_around(settings):
    with pytest.raises(synthesis.SynthesisError, match="LLM_API_KEY"):
        synthesis.synthesize("Who wrote PEP 8?", [retrieval_receipt()], settings)


def test_the_completion_comes_back_with_its_tags_intact(monkeypatch, configured):
    reply_with(monkeypatch, "PEP 8 was written by Guido van Rossum [R1].\n")
    answer = synthesis.synthesize("Who wrote PEP 8?", [retrieval_receipt()], configured)
    assert answer == "PEP 8 was written by Guido van Rossum [R1]."


def test_a_model_that_declines_is_passed_through_as_insufficient(monkeypatch, configured):
    reply_with(monkeypatch, "INSUFFICIENT_EVIDENCE")
    answer = synthesis.synthesize("What is PEP 9999 about?", [retrieval_receipt()], configured)
    assert answer == synthesis.INSUFFICIENT


def test_an_empty_completion_is_an_error_not_an_answer(monkeypatch, configured):
    reply_with(monkeypatch, "   ")
    with pytest.raises(synthesis.SynthesisError, match="empty"):
        synthesis.synthesize("Who wrote PEP 8?", [retrieval_receipt()], configured)


def test_a_planned_tool_argument_is_a_single_line(monkeypatch, configured):
    sent = reply_with(monkeypatch, "100 - 79\nthat is the difference")
    argument = synthesis.plan_tool_call(
        "How far over the limit is a 100-character line?",
        "calculator",
        "Evaluate an arithmetic expression.",
        [retrieval_receipt(snippet="Limit lines to 79 characters.")],
        configured,
    )
    assert argument == "100 - 79"
    assert "79 characters" in sent[0][1]["content"], "the evidence must reach the planner"


def test_a_planner_that_declines_returns_nothing_to_call(monkeypatch, configured):
    reply_with(monkeypatch, "NONE")
    assert (
        synthesis.plan_tool_call(
            "Who wrote PEP 8?", "calculator", "Evaluate an arithmetic expression.", [], configured
        )
        is None
    )
