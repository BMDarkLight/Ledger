"""Qdrant search + cross-encoder rerank.

Phase 1 territory (see the roadmap). The signatures below are the contract the
rest of the pipeline is already written against; the bodies land with Phase 1.
"""

from api.config import Settings
from api.schemas import Receipt


class RetrievalUnavailable(RuntimeError):
    """Raised when the vector store isn't reachable or isn't populated yet."""


def index_document(doc_id: str, text: str, metadata: dict, settings: Settings) -> int:
    """Chunk, embed and upsert a document. Returns the number of chunks written."""
    raise NotImplementedError("Phase 1: ingest → chunk → embed → upsert into Qdrant.")


def search(query: str, settings: Settings, top_k: int | None = None) -> list[Receipt]:
    """Top-k dense search, cross-encoder reranked, returned as retrieval receipts.

    Receipts come back tagged R1..Rn in rank order — the tags the synthesis step
    is required to cite.
    """
    raise NotImplementedError("Phase 1: Qdrant top-k search + cross-encoder rerank.")


def tag_receipts(snippets: list[tuple[str, str, float]]) -> list[Receipt]:
    """Turn (doc_id, snippet, score) triples into tagged retrieval receipts.

    Pure and already usable — the tagging convention lives here so ingestion,
    search and synthesis can't drift apart on it.
    """
    from api.schemas import ReceiptKind

    return [
        Receipt(
            tag=f"R{i}",
            kind=ReceiptKind.RETRIEVAL,
            source=doc_id,
            snippet=snippet,
            score=score,
        )
        for i, (doc_id, snippet, score) in enumerate(snippets, start=1)
    ]
