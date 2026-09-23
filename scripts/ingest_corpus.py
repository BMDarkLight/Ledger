"""Ingest the fetched corpus into the configured vector store.

Idempotent: chunk ids are stable, so re-running replaces points rather than
duplicating them. With QDRANT_URL=":memory:" the index lives only for the
lifetime of the process, which is what tests and CI use.

Usage:
    python -m scripts.ingest_corpus
    QDRANT_URL=":memory:" python -m scripts.ingest_corpus   # no server needed
"""

import argparse
import sys

from api.config import Settings, get_settings
from api.services.retrieval import index_document
from scripts.fetch_corpus import corpus_path, load_document, referenced_pep_ids


def ingest_corpus(settings: Settings, verbose: bool = False) -> int:
    """Index every referenced corpus document. Returns the chunk count."""
    missing = [d for d in referenced_pep_ids() if not corpus_path(d).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} corpus document(s) not fetched: {', '.join(missing)}. "
            "run `python -m scripts.fetch_corpus`"
        )

    total = 0
    for doc_id in referenced_pep_ids():
        count = index_document(doc_id, load_document(doc_id), {"corpus": "peps"}, settings)
        total += count
        if verbose:
            print(f"  {doc_id}: {count} chunks")
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    settings = get_settings()
    try:
        total = ingest_corpus(settings, verbose=not args.quiet)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    where = settings.qdrant_url
    print(f"\n{total} chunks indexed into {settings.qdrant_collection!r} at {where}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
