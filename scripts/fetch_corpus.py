"""Fetch the Phase 0 corpus (Python PEPs) into corpus/.

Only the PEPs the golden set actually references are fetched by default, so a
first run is fast and the eval set and the corpus cannot drift apart.

Usage:
    python -m scripts.fetch_corpus            # just what the golden set needs
    python -m scripts.fetch_corpus --all      # every PEP in the index
"""

import argparse
import re
from pathlib import Path

import httpx

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"
PEP_TEXT_URL = "https://peps.python.org/{doc_id}/"
_PEP_REF = re.compile(r"pep-\d{4}")


def referenced_pep_ids() -> list[str]:
    """Every `pep-XXXX` id named in the golden set, deduplicated and sorted."""
    from eval.golden_set import GOLDEN_SET_PATH

    return sorted(set(_PEP_REF.findall(GOLDEN_SET_PATH.read_text(encoding="utf-8"))))


def fetch(doc_id: str, client: httpx.Client) -> str:
    response = client.get(PEP_TEXT_URL.format(doc_id=doc_id), follow_redirects=True)
    response.raise_for_status()
    return response.text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="not yet implemented")
    args = parser.parse_args(argv)

    if args.all:
        raise NotImplementedError("Phase 1: walk the full PEP index.")

    CORPUS_DIR.mkdir(exist_ok=True)
    with httpx.Client(timeout=30.0) as client:
        for doc_id in referenced_pep_ids():
            destination = CORPUS_DIR / f"{doc_id}.html"
            destination.write_text(fetch(doc_id, client), encoding="utf-8")
            print(f"→ {destination.relative_to(CORPUS_DIR.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
