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
    hits = retrieval.dense_search("maximum line length", indexed, indexed.retrieval_top_k)
    assert len(hits) == indexed.retrieval_top_k
    ranked = retrieval.rerank("maximum line length", hits, indexed)
    assert len(ranked) == indexed.rerank_top_n
    assert ranked == sorted(ranked, key=lambda h: h["score"], reverse=True)


def test_querying_a_collection_that_does_not_exist_is_an_explicit_failure():
    empty = Settings(qdrant_url=retrieval.IN_MEMORY, qdrant_collection="never_created")
    with pytest.raises(retrieval.RetrievalUnavailable, match="ingest documents first"):
        retrieval.dense_search("anything", empty, 5)
