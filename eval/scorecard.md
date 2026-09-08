# Ledger scorecard

_Mode: retrieval (local ONNX embeddings, no LLM calls). Routing numbers carry the same fitted-baseline caveat as routing-only mode. Citation coverage still needs synthesis, so it stays unmeasured here. · generated 2026-09-08 19:21 UTC by `python -m eval.run_golden_set`._

| Metric | Score |
|---|---|
| Cases run | 31 |
| Routing accuracy | 96.8% |
| Refusal accuracy (adversarial) | 100.0% · 2/8 cases observed |
| Tool selection accuracy | 71.4% |
| Retrieval recall@k (dense) | 76.5% |
| Recall after rerank | 76.5% |
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
