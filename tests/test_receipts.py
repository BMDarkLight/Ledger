"""The one rule. These tests are why the receipts module has no dependencies."""

import pytest

from api.schemas import AnswerStatus, Receipt, ReceiptKind
from api.services import receipts


def make_receipt(tag: str = "R1") -> Receipt:
    return Receipt(tag=tag, kind=ReceiptKind.RETRIEVAL, source="pep-0008", snippet="...")


def test_tags_are_extracted_in_order_without_duplicates():
    assert receipts.extract_tags("a [R2] b [R1] c [R2]") == ["R2", "R1"]


def test_strip_tags_leaves_clean_prose():
    assert (
        receipts.strip_tags("Line length is 79 characters [R1].") == "Line length is 79 characters."
    )


@pytest.mark.parametrize(
    "sentence,factual",
    [
        ("PEP 8 recommends 79 characters [R1].", True),
        ("Here is what the corpus says:", False),
        ("The corpus does not contain an answer.", False),
        ("Yes.", False),
    ],
)
def test_only_load_bearing_sentences_need_receipts(sentence, factual):
    assert receipts.looks_factual(sentence) is factual


def test_a_claim_citing_an_unknown_tag_is_not_supported():
    """A fabricated citation is worse than a missing one, so it must not count."""
    claims = receipts.extract_claims("PEP 8 was written in 2001 [R7].", [make_receipt("R1")])
    assert len(claims) == 1
    assert claims[0].supported is False
    assert claims[0].receipt_tags == []


def test_coverage_is_the_share_of_supported_claims():
    answer = "PEP 8 recommends 79 characters [R1]. It was adopted widely across the ecosystem."
    claims = receipts.extract_claims(answer, [make_receipt("R1")])
    assert len(claims) == 2
    assert receipts.citation_coverage(claims) == 0.5


def test_no_receipts_at_all_means_refusal():
    assert receipts.decide_status([], [], floor=1.0) is AnswerStatus.REFUSED


def test_partial_coverage_is_flagged_unverified_not_returned_as_fact():
    answer = "PEP 8 recommends 79 characters [R1]. Guido regrets it most of all."
    available = [make_receipt("R1")]
    claims = receipts.extract_claims(answer, available)
    assert receipts.decide_status(claims, available, floor=1.0) is AnswerStatus.UNVERIFIED


def test_full_coverage_is_verified():
    answer = "PEP 8 recommends 79 characters [R1]."
    available = [make_receipt("R1")]
    claims = receipts.extract_claims(answer, available)
    assert receipts.decide_status(claims, available, floor=1.0) is AnswerStatus.VERIFIED
