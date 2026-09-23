"""Qdrant search and cross-encoder rerank, producing R-tagged receipts.

Embeddings run locally through fastembed (ONNX, no torch, no API key), and the
vector store speaks the same client whether it is a real Qdrant or an in-process
one. Together that lets CI grade retrieval on every pull request without a
secret, so the scorecard in the README is regenerated rather than remembered.
"""

import uuid
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from api.config import Settings
from api.schemas import Receipt, ReceiptKind
from api.services.chunking import chunk_document

if TYPE_CHECKING:  # pragma: no cover - deferred to keep the import cheap
    from qdrant_client import QdrantClient

IN_MEMORY = ":memory:"
"""QDRANT_URL value that runs the store in-process, as tests and CI do."""

_NAMESPACE = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")


class RetrievalUnavailable(RuntimeError):
    """Raised when the vector store isn't reachable or isn't populated yet."""


def point_id(chunk_id: str) -> str:
    """A stable id per chunk, so re-ingesting a document replaces its points."""
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


# Models are expensive to construct and safe to share, so they are built once
# per process and keyed on the setting that selects them.


@lru_cache(maxsize=4)
def _embedder(model_name: str, cache_dir: str):
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=model_name, cache_dir=cache_dir or None)


@lru_cache(maxsize=4)
def _reranker(model_name: str, cache_dir: str):
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=model_name, cache_dir=cache_dir or None)


@lru_cache(maxsize=4)
def _client(url: str, api_key: str) -> "QdrantClient":
    from qdrant_client import QdrantClient

    if url == IN_MEMORY:
        return QdrantClient(location=IN_MEMORY)
    return QdrantClient(url=url, api_key=api_key or None)


def get_client(settings: Settings) -> "QdrantClient":
    return _client(settings.qdrant_url, settings.qdrant_api_key)


def ensure_collection(settings: Settings) -> None:
    from qdrant_client import models

    client = get_client(settings)
    if client.collection_exists(settings.qdrant_collection):
        return
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=models.VectorParams(
            size=settings.embedding_dim, distance=models.Distance.COSINE
        ),
    )


def embed(texts: list[str], settings: Settings) -> list[list[float]]:
    model = _embedder(settings.embedding_model, settings.model_cache_dir)
    return [v.tolist() for v in model.embed(texts)]


def embed_query(query: str, settings: Settings) -> list[float]:
    """Query embedding. Some models prepend an instruction prefix for queries."""
    model = _embedder(settings.embedding_model, settings.model_cache_dir)
    return next(iter(model.query_embed(query))).tolist()


def index_document(doc_id: str, text: str, metadata: dict, settings: Settings) -> int:
    """Chunk, embed and upsert a document. Returns the number of chunks written."""
    from qdrant_client import models

    chunks = chunk_document(doc_id, text)
    if not chunks:
        return 0

    ensure_collection(settings)
    vectors = embed([c.text for c in chunks], settings)
    get_client(settings).upsert(
        collection_name=settings.qdrant_collection,
        points=[
            models.PointStruct(
                id=point_id(chunk.chunk_id),
                vector=vector,
                payload={
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "section": chunk.section,
                    "text": chunk.text,
                    **metadata,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ],
    )
    return len(chunks)


def dense_search(query: str, settings: Settings, top_k: int) -> list[dict[str, Any]]:
    """Vector search only, before reranking.

    Exposed so the eval harness can measure recall at both stages and tell a
    retrieval miss from a rerank drop.
    """
    client = get_client(settings)

    # Every transport failure means the same thing to a caller: there is no
    # evidence to be had. They collapse into one error rather than leaking a
    # driver exception up as a 500.
    try:
        exists = client.collection_exists(settings.qdrant_collection)
    except Exception as exc:
        raise RetrievalUnavailable(
            f"vector store at {settings.qdrant_url} is unreachable: {exc}"
        ) from exc

    if not exists:
        raise RetrievalUnavailable(
            f"collection {settings.qdrant_collection!r} does not exist at "
            f"{settings.qdrant_url}, so ingest documents first "
            "(`python -m scripts.ingest_corpus`)"
        )

    response = client.query_points(
        collection_name=settings.qdrant_collection,
        query=embed_query(query, settings),
        limit=top_k,
        with_payload=True,
    )
    return [{**p.payload, "score": p.score} for p in response.points]


def rerank(query: str, hits: list[dict[str, Any]], settings: Settings) -> list[dict[str, Any]]:
    """Reorder dense hits with a cross-encoder, which reads query and passage together.

    Dense retrieval decides what is available; the cross-encoder decides what is
    responsive. Only the reranked top-n become receipts, so this step determines
    what the answer is allowed to cite.
    """
    if not hits:
        return []
    encoder = _reranker(settings.rerank_model, settings.model_cache_dir)
    scores = list(encoder.rerank(query, [h["text"] for h in hits]))
    ordered = sorted(
        ({**hit, "score": float(score)} for hit, score in zip(hits, scores, strict=True)),
        key=lambda h: h["score"],
        reverse=True,
    )
    return ordered[: settings.rerank_top_n]


def search(query: str, settings: Settings, top_k: int | None = None) -> list[Receipt]:
    """Top-k dense search, cross-encoder reranked, returned as retrieval receipts.

    Receipts come back tagged R1..Rn in rank order. Those are the tags the
    synthesis step is required to cite.
    """
    hits = dense_search(query, settings, top_k or settings.retrieval_top_k)
    ranked = rerank(query, hits, settings)
    return tag_receipts(
        [(h["doc_id"], h["text"], h["score"]) for h in ranked],
        chunk_ids=[h["chunk_id"] for h in ranked],
    )


def tag_receipts(
    snippets: list[tuple[str, str, float]], chunk_ids: list[str] | None = None
) -> list[Receipt]:
    """Turn (doc_id, snippet, score) triples into tagged retrieval receipts.

    The tagging convention lives here alone, so ingestion, search and synthesis
    cannot drift apart on it.
    """
    return [
        Receipt(
            tag=f"R{i}",
            kind=ReceiptKind.RETRIEVAL,
            source=doc_id,
            snippet=snippet,
            score=score,
            metadata={"chunk_id": chunk_ids[i - 1]} if chunk_ids else {},
        )
        for i, (doc_id, snippet, score) in enumerate(snippets, start=1)
    ]
