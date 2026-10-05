"""The faithfulness judge, with the model replaced by scripted verdicts."""

import json

import pytest

from api.config import Settings
from api.schemas import Claim, Receipt, ReceiptKind
from api.services import synthesis
from eval import judge
from eval.metrics import CaseResult, Scorecard

RECEIPTS = [
    Receipt(tag="R1", kind=ReceiptKind.RETRIEVAL, source="pep-0008", snippet="Limit lines to 79."),
    Receipt(tag="R2", kind=ReceiptKind.RETRIEVAL, source="pep-0020", snippet="Beautiful > ugly."),
    Receipt(tag="R3", kind=ReceiptKind.RETRIEVAL, source="pep-0257", snippet="Not cited at all."),
]
CLAIMS = [
    Claim(text="PEP 8 limits lines to 79 characters.", receipt_tags=["R1"], supported=True),
    Claim(text="Guido regrets the limit.", receipt_tags=["R1"], supported=True),
    Claim(text="Nothing backs this one.", receipt_tags=[], supported=False),
]


def verdicts(*flags: bool) -> str:
    return json.dumps(
        {"verdicts": [{"claim": i, "supported": f, "reason": "r"} for i, f in enumerate(flags, 1)]}
    )


@pytest.fixture
def model(monkeypatch):
    calls: list[tuple[list[dict], Settings]] = []
    replies: list[str] = []

    def fake(messages, settings):
        calls.append((messages, settings))
        return replies.pop(0)

    monkeypatch.setattr(synthesis, "_complete", fake)
    return replies, calls


@pytest.fixture
def keyed() -> Settings:
    return Settings(llm_api_key="k", llm_model="writer")


def test_only_cited_claims_are_judged_against_only_their_receipts(model, keyed):
    replies, calls = model
    replies.append(verdicts(True, False))
    result = judge.judge(CLAIMS, RECEIPTS, keyed)

    assert [v.supported for v in result] == [True, False]
    prompt = calls[0][0][1]["content"]
    assert "Nothing backs this one" not in prompt, "uncited claims are coverage's job"
    assert "Limit lines to 79." in prompt
    assert "Not cited at all." not in prompt, "the judge sees cited evidence only"


def test_a_separate_judge_model_is_used_when_configured(model):
    replies, calls = model
    replies.append(verdicts(True, True))
    judge.judge(CLAIMS, RECEIPTS, Settings(llm_api_key="k", llm_model="writer", judge_model="j"))
    assert calls[0][1].llm_model == "j"


def test_nothing_cited_means_no_call_and_nothing_to_score(model, keyed):
    _, calls = model
    assert judge.judge(CLAIMS[2:], RECEIPTS, keyed) == []
    assert calls == []
    assert judge.faithfulness([]) is None


@pytest.mark.parametrize(
    "reply",
    [
        "They look fine to me.",
        verdicts(True),  # one verdict short
        verdicts(True, True, True),  # one too many
        json.dumps({"verdicts": [{"claim": 1, "supported": "yes"}, {"claim": 2, "supported": 1}]}),
    ],
)
def test_an_unusable_judgement_is_none_not_a_score(model, keyed, reply):
    replies, _ = model
    replies.append(reply)
    assert judge.judge(CLAIMS, RECEIPTS, keyed) is None


def test_faithfulness_is_the_supported_share():
    vs = [judge.Verdict("a", True, ""), judge.Verdict("b", False, "")]
    assert judge.faithfulness(vs) == 0.5


def test_scorecard_reports_faithfulness_and_unmeasured_stays_unmeasured():
    card = Scorecard(
        results=[
            CaseResult("R001", "retrieval", routed_correctly=True, faithfulness=0.5),
            CaseResult("R002", "retrieval", routed_correctly=True, faithfulness=None),
        ]
    )
    assert card.faithfulness == 0.5
    assert "| Faithfulness (judged) | 50.0% |" in card.to_markdown()
    assert "| Faithfulness (judged) | _not measured_ |" in Scorecard().to_markdown()


def test_the_eval_records_an_unsupported_claim_as_the_cases_failure(model, keyed):
    from api.schemas import AnswerStatus, ReceiptedAskResponse, Route, RouteDecision
    from eval.run_golden_set import _judge_answer

    replies, _ = model
    replies.append(verdicts(True, False))
    answer = ReceiptedAskResponse(
        answer="...",
        status=AnswerStatus.VERIFIED,
        route=RouteDecision(route=Route.RETRIEVE, rationale="t"),
        receipts=RECEIPTS,
        claims=CLAIMS[:2],
        citation_coverage=1.0,
    )
    result = CaseResult("R002", "retrieval", routed_correctly=True)
    _judge_answer(result, answer, keyed)

    assert result.faithfulness == 0.5
    assert "Guido regrets the limit." in result.error


def test_an_unreachable_judge_leaves_the_case_unmeasured(monkeypatch, keyed):
    from api.schemas import AnswerStatus, ReceiptedAskResponse, Route, RouteDecision
    from eval.run_golden_set import _judge_answer

    def down(messages, settings):
        raise synthesis.SynthesisError("unreachable")

    monkeypatch.setattr(synthesis, "_complete", down)
    answer = ReceiptedAskResponse(
        answer="...",
        status=AnswerStatus.VERIFIED,
        route=RouteDecision(route=Route.RETRIEVE, rationale="t"),
        receipts=RECEIPTS,
        claims=CLAIMS[:1],
        citation_coverage=1.0,
    )
    result = CaseResult("R002", "retrieval", routed_correctly=True)
    _judge_answer(result, answer, keyed)
    assert (result.faithfulness, result.error) == (None, None)
