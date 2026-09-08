"""Chunk boundaries decide what Ledger can cite, so they get tested directly."""

from api.services.chunking import (
    METADATA_SECTION,
    Chunk,
    chunk_document,
    split_sections,
)

RST = """\
PEP: 20
Title: The Zen of Python
Author: Tim Peters

Abstract
========

Long time Pythoneer Tim Peters channels the BDFL's guiding principles.

The Zen of Python
=================

Beautiful is better than ugly.
Explicit is better than implicit.

Copyright
=========

This document has been placed in the public domain.
"""


def test_preamble_becomes_its_own_metadata_section():
    sections = split_sections(RST)
    assert sections[0][0] == METADATA_SECTION
    assert "Author: Tim Peters" in sections[0][1]


def test_section_titles_are_recovered_from_rst_underlines():
    assert [title for title, _ in split_sections(RST)] == [
        METADATA_SECTION,
        "Abstract",
        "The Zen of Python",
        "Copyright",
    ]


def test_a_line_of_dashes_is_not_a_header_without_a_title_above_it():
    assert [t for t, _ in split_sections("=====\n\nbody text here\n")] == [METADATA_SECTION]


def test_chunks_carry_their_section_for_context():
    chunks = chunk_document("pep-0020", RST)
    zen = next(c for c in chunks if "Beautiful" in c.text)
    assert zen.section == "The Zen of Python"
    assert zen.text.startswith("The Zen of Python"), "section title is prepended for embedding"


def test_metadata_chunk_is_not_prefixed_with_its_own_title():
    first = chunk_document("pep-0020", RST)[0]
    assert first.section == METADATA_SECTION
    assert first.text.startswith("PEP: 20")


def test_chunk_ids_are_stable_and_addressable():
    chunks = chunk_document("pep-0020", RST)
    assert [c.chunk_id for c in chunks[:2]] == ["pep-0020#0", "pep-0020#1"]
    assert all(c.doc_id == "pep-0020" for c in chunks)


def test_long_sections_are_split_with_paragraph_overlap():
    body = "Section\n=======\n\n" + "\n\n".join(f"Paragraph number {i}." * 8 for i in range(12))
    chunks = chunk_document("doc", body, max_chars=400, overlap=1)
    assert len(chunks) > 1
    # The last paragraph of one window opens the next, so a fact at a boundary
    # is reachable from both sides.
    first_tail = chunks[0].text.rstrip().split("\n\n")[-1]
    assert first_tail in chunks[1].text


def test_an_oversized_paragraph_is_kept_whole_rather_than_cut_mid_line():
    giant = "x" * 3000
    chunks = chunk_document("doc", f"Code\n====\n\n{giant}\n", max_chars=400)
    assert any(giant in c.text for c in chunks), "a long code block must not be sliced"


def test_empty_sections_are_dropped():
    assert chunk_document("doc", "Header\n======\n\n\nOther\n=====\n\nreal body\n") == [
        Chunk(doc_id="doc", ordinal=0, section="Other", text="Other\n\nreal body")
    ]
