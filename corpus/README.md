# Corpus

Ledger's Phase 0 corpus is the **Python Enhancement Proposals** — public, stable,
plain text, and dense with checkable facts (numbers, statuses, dates, authors).
It also has clean edges, which is what the adversarial set needs: a question
about a PEP that does not exist, or about something no PEP states, has an
unambiguously correct answer, and that answer is a refusal.

Nothing is vendored here. Fetch it:

```bash
python -m scripts.fetch_corpus          # writes corpus/pep-XXXX.txt
```

Then ingest it into Qdrant via `POST /v1/documents` (Phase 1).

## Swapping the corpus

The corpus is referenced in exactly three places:

1. `eval/golden_set.jsonl` — the questions and their `expected_sources`
2. `api/services/router.py` — the `_CORPUS_SUBJECT` pattern
3. `scripts/fetch_corpus.py` — how documents are obtained

Nothing else in the pipeline knows what it has been fed.
