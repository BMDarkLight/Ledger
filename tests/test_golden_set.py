"""The golden set is checked in, so it gets tested like code.

Phase 0's deliverable is this file passing before any pipeline code exists.
"""

from collections import Counter

from eval.golden_set import Behavior, Category, load_golden_set


def test_golden_set_parses_and_validates():
    cases = load_golden_set()
    assert len(cases) >= 30, "the roadmap calls for ~30 labeled cases"


def test_case_ids_are_unique():
    ids = [c.id for c in load_golden_set()]
    assert len(ids) == len(set(ids))


def test_all_four_categories_are_represented():
    counts = Counter(c.category for c in load_golden_set())
    assert set(counts) == set(Category)
    for category, count in counts.items():
        assert count >= 5, f"{category.value} is too thin to be a real signal"


def test_adversarial_cases_are_mostly_refusals():
    adversarial = [c for c in load_golden_set() if c.category is Category.ADVERSARIAL]
    refusals = [c for c in adversarial if c.expected_behavior is Behavior.REFUSE]
    assert len(refusals) >= len(adversarial) - 1, (
        "the adversarial set exists to test refusal; at most one case may be an answer-anyway trap"
    )


def test_tool_cases_expect_no_corpus_sources():
    for case in load_golden_set():
        if case.category is Category.TOOL:
            assert not case.expected_sources, f"{case.id} is a tool case but expects a document"


def test_multi_hop_cases_chain_a_source_and_a_tool():
    for case in load_golden_set():
        if case.category is Category.MULTI_HOP:
            assert case.expected_sources and case.expected_tools
            assert case.expected_route == "retrieve_then_tool"
