"""Fetch the Phase 0 corpus (Python PEPs) into corpus/.

PEPs are pulled as reStructuredText source rather than as rendered HTML: the
source is a third the size, needs no tag stripping, and carries the metadata
headers (Title, Author, Status, Created) as plain text at the top of the file,
which is exactly what several golden-set questions ask about.

Only the PEPs the golden set actually references are fetched, so the corpus and
the eval set cannot drift apart.

Usage:
    python -m scripts.fetch_corpus            # what the golden set needs
    python -m scripts.fetch_corpus --force    # re-download even if cached
"""

import argparse
import re
import sys
from pathlib import Path

import httpx

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"
SOURCE_URL = "https://raw.githubusercontent.com/python/peps/main/peps/{doc_id}.rst"
_PEP_REF = re.compile(r"pep-\d{4}")


def referenced_pep_ids() -> list[str]:
    """Every `pep-XXXX` id named in the golden set, deduplicated and sorted."""
    from eval.golden_set import GOLDEN_SET_PATH

    return sorted(set(_PEP_REF.findall(GOLDEN_SET_PATH.read_text(encoding="utf-8"))))


def corpus_path(doc_id: str) -> Path:
    return CORPUS_DIR / f"{doc_id}.rst"


def load_document(doc_id: str) -> str:
    """Read a fetched document. Raises FileNotFoundError if the corpus is absent."""
    return corpus_path(doc_id).read_text(encoding="utf-8")


def fetch(doc_id: str, client: httpx.Client) -> str:
    response = client.get(SOURCE_URL.format(doc_id=doc_id), follow_redirects=True)
    response.raise_for_status()
    return response.text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download cached documents")
    args = parser.parse_args(argv)

    CORPUS_DIR.mkdir(exist_ok=True)
    doc_ids = referenced_pep_ids()
    if not doc_ids:
        print("no pep-XXXX ids referenced by the golden set", file=sys.stderr)
        return 1

    with httpx.Client(timeout=30.0) as client:
        for doc_id in doc_ids:
            destination = corpus_path(doc_id)
            if destination.exists() and not args.force:
                print(f"· {destination.name} (cached)")
                continue
            destination.write_text(fetch(doc_id, client), encoding="utf-8")
            print(f"↓ {destination.name} ({destination.stat().st_size:,} bytes)")

    print(f"\n{len(doc_ids)} documents in {CORPUS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
