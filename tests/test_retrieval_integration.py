"""End-to-end retrieval against the real corpus, in an in-process vector store.

Needs the corpus on disk and downloads ONNX models on first run, so it is marked
`retrieval` and deselected by default:

    python -m scripts.fetch_corpus && pytest -m retrieval
"""

import pytest

from api.config import Settings
from api.schemas import ReceiptKind
from api.services import retrieval
from scripts.fetch_corpus import corpus_path, referenced_pep_ids
from scripts.ingest_corpus import ingest_corpus

pytestmark = pytest.mark.retrieval


@pytest.fixture(scope="module")
def indexed() -> Settings:
    missing = [d for d in referenced_pep_ids() if not corpus_path(d).exists()]
    if missing:
        pytest.skip(f"corpus not fetched ({len(missing)} missing)")
    settings = Settings(qdrant_url=retrieval.IN_MEMORY, qdrant_collection="ledger_test")
    ingest_corpus(settings)
    return settings


def test_ingestion_indexes_every_document(indexed):
    client = retrieval.get_client(indexed)
    count = client.count(indexed.qdrant_collection).count
    assert count > 300, f"expected the whole corpus, got {count} chunks"


def test_reingesting_replaces_rather_than_duplicates(indexed):
    client = retrieval.get_client(indexed)
    before = client.count(indexed.qdrant_collection).count
    ingest_corpus(indexed)
    assert client.count(indexed.qdrant_collection).count == before


def test_search_returns_tagged_retrieval_receipts(indexed):
    receipts = retrieval.search("Who are the authors of PEP 8?", indexed)
    assert receipts, "a question the corpus answers must return evidence"
    assert [r.tag for r in receipts] == [f"R{i}" for i in range(1, len(receipts) + 1)]
    assert all(r.kind is ReceiptKind.RETRIEVAL for r in receipts)
    assert all(r.metadata.get("chunk_id") for r in receipts), "receipts must be addressable"


def test_search_finds_the_document_the_golden_set_expects(indexed):
    receipts = retrieval.search("Who are the authors of PEP 8?", indexed)
    assert "pep-0008" in {r.source for r in receipts}


def test_rerank_narrows_to_the_configured_top_n(indexed):
    hits = retrieval.candidate_search("maximum line length", indexed, indexed.retrieval_top_k)
    assert len(hits) == indexed.retrieval_top_k
    ranked = retrieval.rerank("maximum line length", hits, indexed)
    assert len(ranked) == indexed.rerank_top_n
    assert ranked == sorted(ranked, key=lambda h: h["score"], reverse=True)


def test_querying_a_collection_that_does_not_exist_is_an_explicit_failure():
    empty = Settings(qdrant_url=retrieval.IN_MEMORY, qdrant_collection="never_created")
    with pytest.raises(retrieval.RetrievalUnavailable, match="ingest documents first"):
        retrieval.candidate_search("anything", empty, 5)


def test_both_retrieval_modes_search_the_same_index(indexed):
    """Both vectors are always written, so switching modes needs no re-ingest."""
    dense = indexed.model_copy(update={"retrieval_mode": "dense"})
    for settings in (indexed, dense):
        hits = retrieval.candidate_search("What problem does PEP 668 solve?", settings, 10)
        assert len(hits) == 10


def test_hybrid_matches_a_pep_by_its_number(indexed):
    """The exact-token case BM25 is there for: the number is the whole signal."""
    hits = retrieval.candidate_search("What problem does PEP 668 solve?", indexed, 10)
    assert "pep-0668" in {h["doc_id"] for h in hits}


def test_a_collection_from_before_hybrid_search_is_an_explicit_failure():
    from qdrant_client import models

    old = Settings(qdrant_url=retrieval.IN_MEMORY, qdrant_collection="dense_only_legacy")
    retrieval.get_client(old).create_collection(
        "dense_only_legacy",
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )
    with pytest.raises(retrieval.RetrievalUnavailable, match="--recreate"):
        retrieval.candidate_search("anything", old, 5)


def test_recreate_rebuilds_a_legacy_collection(monkeypatch):
    from qdrant_client import models

    from scripts import ingest_corpus as script

    old = Settings(qdrant_url=retrieval.IN_MEMORY, qdrant_collection="legacy_to_rebuild")
    client = retrieval.get_client(old)
    client.create_collection(
        "legacy_to_rebuild",
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )
    monkeypatch.setattr(script, "get_settings", lambda: old)
    assert script.main(["--recreate", "--quiet"]) == 0
    assert retrieval.candidate_search("PEP 8 line length", old, 3)


# Held out: generic questions that name a PEP by number, none from the golden
# set. Hybrid is the default because of this comparison. On the golden set it
# loses half of one two-document case after reranking (M006), so the scorecard
# alone would argue the other way; this is the wider evidence.
_NUMBERED = [
    template.format(n=n)
    for n in (1, 8, 20, 257, 484, 572, 621, 668)
    for template in (
        "What is PEP {n} about?",
        "What motivated PEP {n}?",
        "What does PEP {n} specify?",
        "Summarize the rationale given in PEP {n}.",
        "What problem does PEP {n} solve?",
    )
]


def _found(question: str, settings: Settings) -> bool:
    n = int(question.split("PEP ")[1].split()[0].rstrip("?."))
    hits = retrieval.candidate_search(question, settings, settings.retrieval_top_k)
    return f"pep-{n:04d}" in {h["doc_id"] for h in hits}


def test_hybrid_beats_dense_on_questions_that_name_a_pep_by_number(indexed):
    dense = indexed.model_copy(update={"retrieval_mode": "dense"})
    hybrid_hits = sum(_found(q, indexed) for q in _NUMBERED)
    dense_hits = sum(_found(q, dense) for q in _NUMBERED)
    assert hybrid_hits > dense_hits, (hybrid_hits, dense_hits)
    assert hybrid_hits >= 34, f"hybrid found {hybrid_hits}/40, measured 36/40 when chosen"
