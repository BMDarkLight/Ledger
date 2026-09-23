"""End to end through the pipeline, with retrieval and the model stubbed out.

These cover the seam the project turns on: whatever the model writes, a claim is
only reported as verified if the receipt it cites exists.
"""

import pytest

from api.config import Settings
from api.routers.ask import run_pipeline
from api.schemas import AnswerStatus, AskRequest, Receipt, ReceiptKind
from api.services import retrieval, synthesis


@pytest.fixture
def configured() -> Settings:
    return Settings(llm_api_key="test-key")


@pytest.fixture
def corpus(monkeypatch):
    """One retrieval receipt for any question."""
    receipt = Receipt(
        tag="R1",
        kind=ReceiptKind.RETRIEVAL,
        source="pep-0008",
        snippet="Limit all lines to a maximum of 79 characters.",
        score=0.9,
    )
    monkeypatch.setattr(retrieval, "search", lambda q, s, k=None: [receipt])
    return receipt


@pytest.fixture
def model(monkeypatch):
    """Let a test choose what the model says, one reply per call."""

    replies: list[str] = []

    def fake(messages, settings):
        return replies.pop(0) if replies else ""

    monkeypatch.setattr(synthesis, "_complete", fake)
    return replies


def test_a_fully_cited_answer_comes_back_verified(corpus, model, configured):
    model.append("PEP 8 limits lines to 79 characters [R1].")
    result = run_pipeline(AskRequest(question="What line length does PEP 8 recommend?"), configured)

    assert result.status is AnswerStatus.VERIFIED
    assert result.citation_coverage == 1.0
    assert [r.tag for r in result.receipts] == ["R1"]


def test_an_uncited_claim_drags_the_answer_down_to_unverified(corpus, model, configured):
    model.append("PEP 8 limits lines to 79 characters [R1]. Guido later regretted the rule.")
    result = run_pipeline(AskRequest(question="What line length does PEP 8 recommend?"), configured)

    assert result.status is AnswerStatus.UNVERIFIED
    assert result.citation_coverage == 0.5
    assert [c.supported for c in result.claims] == [True, False]


def test_a_tag_pointing_at_nothing_does_not_count_as_support(corpus, model, configured):
    """The failure mode that matters: a citation that looks real and is not."""
    model.append("PEP 8 limits lines to 79 characters [R4].")
    result = run_pipeline(AskRequest(question="What line length does PEP 8 recommend?"), configured)

    assert result.status is AnswerStatus.UNVERIFIED
    assert result.claims[0].receipt_tags == []


def test_a_model_that_declines_produces_a_refusal(corpus, model, configured):
    model.append(synthesis.INSUFFICIENT)
    result = run_pipeline(AskRequest(question="What does PEP 9999 say?"), configured)

    assert result.status is AnswerStatus.REFUSED
    assert result.receipts, "the evidence that failed to answer is still reported"


def test_an_opinion_question_never_reaches_retrieval_or_the_model(configured):
    result = run_pipeline(AskRequest(question="In your opinion, is PEP 8 too strict?"), configured)

    assert result.status is AnswerStatus.REFUSED
    assert result.receipts == []


def test_a_chained_question_computes_from_the_retrieved_figure(corpus, model, configured):
    """The multi-hop path: the number to compute with is in the receipts, so the
    expression has to be written after retrieval rather than read off the question."""
    model.append("100 - 79")
    model.append("A 100-character line is 21 characters over the PEP 8 limit [R1][T1].")

    result = run_pipeline(
        AskRequest(
            question=(
                "PEP 8 recommends a maximum line length. How many characters over "
                "that limit is a 100-character line?"
            )
        ),
        configured,
    )

    assert result.status is AnswerStatus.VERIFIED
    tool_receipt = next(r for r in result.receipts if r.kind is ReceiptKind.TOOL)
    assert tool_receipt.tag == "T1"
    assert tool_receipt.snippet == "21"
    assert tool_receipt.metadata["argument"] == "100 - 79"


def test_arithmetic_already_in_the_question_needs_no_planning_call(model, configured):
    model.append("1327 multiplied by 4519 is 5996713 [T1].")
    result = run_pipeline(AskRequest(question="What is 1327 * 4519?"), configured)

    assert result.status is AnswerStatus.VERIFIED
    assert result.receipts[0].snippet == "5996713"
    assert not model, "only the synthesis call should have been made"
