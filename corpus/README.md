# Corpus

Ledger's Phase 0 corpus is the **Python Enhancement Proposals**: public, stable,
plain text, and dense with checkable facts (numbers, statuses, dates, authors).
It also has clean edges, which is what the adversarial set needs. A question
about a PEP that does not exist, or about something no PEP states, has an
unambiguously correct answer, and that answer is a refusal.

Nothing is vendored here. Fetch it:

```bash
python -m scripts.fetch_corpus          # writes corpus/pep-XXXX.rst
```

Documents are pulled as reStructuredText source rather than rendered HTML: a
third the size, no tag stripping, and the `PEP:`/`Title:`/`Author:`/`Created:`
headers arrive as plain text, which is what several golden-set questions ask
about.

Once fetched, the eval set is checked against the real text:

```bash
pytest -m corpus
```

That asserts every `answer_contains` string in a non-tool case actually occurs
in the document it cites. It also asserts the converse: that a tool case's
expected answer is not already sitting in a document, which would mean the tool
call was never really required.

Then ingest it into Qdrant:

```bash
python -m scripts.ingest_corpus
```

## Swapping the corpus

The corpus is referenced in exactly three places:

1. `eval/golden_set.jsonl`, the questions and their `expected_sources`
2. `api/services/router.py`, the `_CORPUS_SUBJECT` pattern
3. `scripts/fetch_corpus.py`, how documents are obtained

Nothing else in the pipeline knows what it has been fed.
