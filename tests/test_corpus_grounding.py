"""Does the golden set describe the corpus we actually have?

An eval set whose expected answers were written from memory grades every
downstream metric against a guess. These tests close that gap by checking each
expectation against the fetched source text.

Marked `corpus` and deselected by default, because they need the corpus on disk:

    python -m scripts.fetch_corpus && pytest -m corpus
"""

import pytest

from eval.golden_set import Category, load_golden_set
from scripts.fetch_corpus import corpus_path, load_document, referenced_pep_ids

pytestmark = pytest.mark.corpus


def _require_corpus() -> None:
    missing = [d for d in referenced_pep_ids() if not corpus_path(d).exists()]
    if missing:
        pytest.skip(f"corpus not fetched ({len(missing)} missing) — run scripts.fetch_corpus")


@pytest.fixture(autouse=True)
def _corpus_present():
    _require_corpus()


def test_every_referenced_document_was_fetched():
    for doc_id in referenced_pep_ids():
        assert corpus_path(doc_id).exists(), f"{doc_id} is referenced but not in the corpus"


def test_each_document_is_the_pep_its_filename_claims():
    """Guards against a redirect or a renumbering silently swapping a document."""
    for doc_id in referenced_pep_ids():
        number = int(doc_id.removeprefix("pep-"))
        header = load_document(doc_id).splitlines()[0]
        assert header == f"PEP: {number}", f"{doc_id} starts with {header!r}"


def test_expected_sources_are_documents_we_actually_have():
    for case in load_golden_set():
        for doc_id in case.expected_sources:
            assert corpus_path(doc_id).exists(), f"{case.id} cites missing document {doc_id}"


def test_corpus_backed_answers_actually_appear_in_the_corpus():
    """The core check: an expectation that no document supports is not ground truth.

    Only applies to cases that expect no tool. A case that calls a tool has a
    *derived* expected answer — `100 - 79 = 21` is correct precisely because it
    is nowhere in PEP 8 — so grounding it in document text would be wrong.
    """
    unfounded = []
    for case in load_golden_set():
        if case.expected_tools or not case.answer_contains:
            continue
        assert case.expected_sources, f"{case.id} asserts an answer but cites no source"
        haystack = "\n".join(load_document(d) for d in case.expected_sources).lower()
        unfounded += [
            f"{case.id}: {needle!r} not found in {case.expected_sources}"
            for needle in case.answer_contains
            if needle.lower() not in haystack
        ]
    assert not unfounded, "expected answers with no support in the corpus:\n" + "\n".join(unfounded)


def test_tool_and_multi_hop_answers_are_derived_not_quoted():
    """The converse. If a computed answer is already sitting in a document,
    the case is not testing a tool call — it is testing retrieval by accident."""
    for case in load_golden_set():
        if case.category not in (Category.TOOL, Category.MULTI_HOP):
            continue
        if not (case.answer_contains and case.expected_sources):
            continue
        haystack = "\n".join(load_document(d) for d in case.expected_sources).lower()
        for needle in case.answer_contains:
            assert needle.lower() not in haystack, (
                f"{case.id} expects {needle!r}, which is already in {case.expected_sources} — "
                "the tool call is not actually required"
            )
