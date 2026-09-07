# Ledger scorecard

_Mode: routing-only (deterministic baseline router, no LLM calls). Read this number with suspicion: the baseline's patterns were written against these same questions, so it is fitted to them and its score is an upper bound, not a generalization estimate. · generated 2026-09-07 22:59 UTC by `python -m eval.run_golden_set`._

| Metric | Score |
|---|---|
| Cases run | 31 |
| Routing accuracy | 96.8% |
| Refusal accuracy (adversarial) | 100.0% · 2/8 cases observed |
| Tool selection accuracy | 71.4% |
| Retrieval recall@k | _not measured_ |
| Citation coverage | _not measured_ |

## Routing accuracy by category

| Category | Score |
|---|---|
| `adversarial` | 87.5% |
| `multi_hop` | 100.0% |
| `retrieval` | 100.0% |
| `tool` | 100.0% |

## Failing cases

| Case | Category | Why |
|---|---|---|
| `T006` | tool | chose ['calculator'], expected ['code_exec'] |
| `M002` | multi_hop | chose ['calculator'], expected ['clock', 'calculator'] |
| `M005` | multi_hop | chose ['calculator'], expected ['web_search'] |
| `A008` | adversarial | routed `retrieve_then_tool`, expected `retrieve` |
