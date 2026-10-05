# Ledger scorecard

_Mode: retrieval (local ONNX embeddings, no answers generated). Citation coverage needs synthesis, so it stays unmeasured here. Router: deterministic baseline. Read its routing numbers with suspicion: its patterns were written against these same questions, so its score is an upper bound, not a generalization estimate. Generated 2026-10-05 05:02 UTC by `python -m eval.run_golden_set`._

| Metric | Score |
|---|---|
| Cases run | 31 |
| Routing accuracy | 96.8% |
| Refusal accuracy (adversarial) | 100.0% (2 of 8 cases observed) |
| Tool selection accuracy | 92.9% |
| Retrieval recall@k (dense) | 94.1% |
| Recall after rerank | 94.1% |
| Citation coverage | _not measured_ |
| Expected answer match | _not measured_ |
| Faithfulness (judged) | _not measured_ |

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
| `M004` | multi_hop | expected ['pep-0484'] not retrieved: nothing |
| `A008` | adversarial | routed `retrieve_then_tool`, expected `retrieve` |
