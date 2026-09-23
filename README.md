# Ledger

**Nothing enters the answer without a receipt.**

[![Tests](https://github.com/BMDarkLight/Ledger/actions/workflows/tests.yml/badge.svg)](https://github.com/BMDarkLight/Ledger/actions/workflows/tests.yml) [![Lint](https://github.com/BMDarkLight/Ledger/actions/workflows/lint.yml/badge.svg)](https://github.com/BMDarkLight/Ledger/actions/workflows/lint.yml) [![Eval](https://github.com/BMDarkLight/Ledger/actions/workflows/eval.yml/badge.svg)](https://github.com/BMDarkLight/Ledger/actions/workflows/eval.yml) [![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Table of contents

- [Highlights](#highlights)
- [The problem](#the-problem)
- [How it works](#how-it-works)
- [The one rule](#the-one-rule)
- [Eval harness](#eval-harness)
- [API](#api)
- [Getting started](#getting-started)
- [Testing](#testing)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Status](#status)
- [Roadmap](#roadmap)
- [Design goals](#design-goals)
- [License](#license)

---

## Highlights

| | |
|---|---|
| **Receipts only** | Every claim in an answer is tagged `[R#]` (retrieved chunk) or `[T#]` (tool call). No tag, no claim: it gets flagged unverified instead. |
| **Transparent routing** | `POST /v1/route` returns the routing decision and its rationale without spending a generation call. |
| **Adversarial eval set** | The golden set includes unanswerable questions and disguised tool-questions, not just easy wins. |
| **Multi-hop chaining** | A question can retrieve a figure from the corpus and then compute against it, with both halves cited. |
| **Living scorecard** | CI re-runs the eval set on every PR and updates a checked-in scorecard, so the numbers are never stale. |
| **OpenAI-compatible** | `/v1/chat/completions`, so existing clients point at Ledger unchanged. |

---

## The problem

Most RAG demos are built and judged on one happy-path question. They fall apart
the moment a question cannot be answered from the corpus, or secretly needs a
live tool instead of a document, or needs both chained together. Ledger is built
the other way around: the evaluation set, including the questions designed to
break it, exists before most of the pipeline code does.

---

## How it works

```
==========================
USER QUERY
==========================

==========================
ROUTER
==========================
Decides: retrieve | call a tool | both, chained
Emits a routing rationale, inspectable via /v1/route

==========================
RETRIEVAL              TOOLS
==========================
Qdrant top-k search     Calculator, clock
+ cross-encoder rerank  (web search, code exec: not built)

==========================
SYNTHESIS
==========================
Every sentence tagged with a receipt:
  [R3] retrieved chunk id 3
  [T1] tool call id 1
No receipt on a claim -> answer marked "unverified"
No receipt at all -> Ledger declines rather than invents

==========================
EVAL HARNESS
==========================
Golden set + adversarial traps
Retrieval recall@k, routing accuracy, refusal accuracy, citation coverage
```

---

## The one rule

Ledger has exactly one non-negotiable behavior, and everything else is built to
serve it:

> **A claim without a receipt is not an answer.**

In practice:

- Every generated sentence carrying a fact is checked for a citation tag before
  it is returned.
- A citation tag that does not resolve to a receipt that actually exists is
  dropped rather than counted. A fabricated citation is worse than a missing one.
- A question the corpus cannot answer and no tool can resolve gets a plain
  refusal, never a confident guess.
- Routing decisions are never hidden inside a single opaque generation call.
  They are a separate, inspectable step.

---

## Eval harness

The eval set lives in `eval/golden_set.jsonl` as labeled cases across four
categories:

| Category | What it tests |
|---|---|
| `retrieval` | Answerable directly from the corpus |
| `tool` | Needs a live tool (price, date, calculation), not in the corpus at all |
| `multi_hop` | Needs retrieval and then a tool, chained |
| `adversarial` | Unanswerable by design, where the correct behavior is a refusal |

Each run reports:

- **Retrieval recall@k**, measured before and after reranking, so a rerank drop
  is distinguishable from a retrieval miss
- **Routing accuracy**, overall and per category
- **Tool selection accuracy**, which routing accuracy alone hides: the right
  route with the wrong tool still fails
- **Refusal accuracy** on the adversarial set, alongside how many of those cases
  the run was actually able to observe
- **Citation coverage**, the share of factual sentences carrying a valid receipt
- **Expected answer match**, the share of each case's required strings that
  turned up in the prose. Coverage says every claim was cited; this says the
  question got answered

Results are written to `eval/scorecard.md` and committed by CI, so the badge and
the numbers stay attached to a real run.

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/v1/route` | Routing decision and rationale, no generation |
| `POST` | `/v1/ask` | Full pipeline: route, retrieve or call tools, synthesize |
| `POST` | `/v1/ask/receipts` | The same, with receipts returned as structured data separate from the prose |
| `POST` | `/v1/documents` | Ingest a document into the vector store |
| `GET` | `/v1/tools` | List available tools and whether each is configured |
| `POST` | `/v1/eval/run` | Run the golden set on demand, return a scorecard |
| `POST` | `/v1/chat/completions` | OpenAI-compatible (streaming not built yet) |

A request that cannot be answered says so with a status code rather than a
degraded answer: `503` when the vector store is unreachable, a tool is
unconfigured or the model cannot be reached, `501` for a stage that is not
built yet.

---

## Getting started

### Prerequisites

- **Python 3.10+**
- **Qdrant** running locally (`docker run -p 6333:6333 qdrant/qdrant`)
- An OpenAI-compatible endpoint and key, for generation only

### Install and run

```bash
git clone https://github.com/BMDarkLight/Ledger.git
cd Ledger
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # set LLM_API_KEY, QDRANT_URL

python -m scripts.fetch_corpus    # pull the PEP corpus
python -m scripts.ingest_corpus   # chunk, embed and index it

uvicorn api.main:app --reload
```

No Qdrant server handy? Set `QDRANT_URL=":memory:"` to run the store in-process.
Embeddings are local either way, so retrieval needs no API key. Only synthesis
does.

- Docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest                            # unit tests, no corpus or API key needed

python -m scripts.fetch_corpus
pytest -m corpus                  # every expected answer checked against real PEP text
pytest -m retrieval               # end-to-end retrieval, downloads ONNX models once

python -m eval.run_golden_set     # eval harness
```

The harness picks the richest mode the environment supports and writes
`eval/scorecard.md`:

| Mode | Needs | Measures |
|---|---|---|
| `routing` | nothing | routing, tool selection |
| `retrieval` | the corpus on disk | the above, plus recall@k before and after reranking |
| `full` | `LLM_API_KEY` | the above, plus citation coverage, expected answer match, and refusals as they actually happened rather than as the route implied |

Embeddings run locally, so CI reaches `retrieval` mode with no secrets. The
retrieval numbers are produced on every pull request, not just on `main`.

---

## Tech stack

- **FastAPI** and **Uvicorn** for the HTTP surface and OpenAPI docs
- **Qdrant** for the vector store, with an in-process mode for tests and CI
- **fastembed** for local ONNX embeddings and cross-encoder reranking: no API
  key and no torch, so CI can grade retrieval on every pull request
- **Pydantic v2** for schemas and configuration
- **openai** as the client for any OpenAI-compatible generation endpoint
- **pytest** for testing
- **Docker** for packaging

Retrieval is written directly against `qdrant-client` rather than through an
orchestration framework. Receipts depend on an exact chunk-to-citation-tag
mapping, and owning that code outright is simpler than configuring a framework
to preserve it.

---

## Project structure

```
Ledger/
├── api/
│   ├── routers/
│   │   ├── ask.py           # /v1/ask, /v1/ask/receipts
│   │   ├── route.py         # /v1/route
│   │   ├── documents.py     # ingestion
│   │   ├── tools.py         # /v1/tools
│   │   ├── eval.py          # /v1/eval/run
│   │   └── chat.py          # OpenAI-compatible surface
│   ├── services/
│   │   ├── router.py        # routing decision logic
│   │   ├── retrieval.py     # Qdrant search + rerank
│   │   ├── tools.py         # tool registry + execution
│   │   ├── receipts.py      # citation tagging + refusal logic
│   │   └── synthesis.py     # receipted generation, and tool-argument planning
│   ├── config.py            # environment-backed settings
│   ├── schemas.py           # Receipt, Claim, RouteDecision, ...
│   └── main.py
├── eval/
│   ├── golden_set.jsonl     # labeled eval cases, incl. adversarial
│   ├── golden_set.py        # case schema + loader
│   ├── metrics.py           # recall@k, routing/refusal accuracy, coverage
│   ├── run_golden_set.py
│   └── scorecard.md         # CI-updated results
├── corpus/                  # fetched, not vendored, see corpus/README.md
├── scripts/fetch_corpus.py
├── tests/
├── Dockerfile
├── docker-compose.yml       # Ledger + Qdrant
├── requirements.txt
└── README.md
```

---

## Status

Built and covered by tests: chunking, retrieval with reranking, the receipts and
refusal gate, the deterministic baseline router, the calculator and clock tools,
receipted synthesis, tool-argument planning for chained questions, and the eval
harness in all three modes.

Not built yet: the web search and code execution tools (both report themselves
as unavailable rather than pretending), the model-backed router that is meant to
beat the deterministic baseline, SSE streaming on the OpenAI-compatible surface,
and faithfulness scoring by an LLM judge. Citation coverage measures whether a
claim carries a receipt, not yet whether the receipt supports it.

---

## Roadmap

- [x] **Phase 0** Pick a corpus (Python PEPs), write ~30 golden questions
      (including adversarial traps) before any pipeline code
- [x] **Phase 1** Retrieval MVP: ingest, embed, search, rerank, cited answers,
      baseline recall@k
- [ ] **Phase 2** Model-backed router, plus web search and sandboxed code
      execution
- [x] **Phase 3** Receipts and refusal logic, with the adversarial set graded on
      refusals as they happen
- [x] **Phase 4** Multi-hop chaining, where the tool argument is written from
      the retrieved evidence
- [ ] **Phase 5** Streaming, faithfulness judging, optional live demo

---

## Design goals

- **Evaluation before generation.** The golden set exists before the pipeline does
- **Refuse rather than invent.** A refusal beats a confident fabrication
- **Inspectable.** Routing and receipts are always available on request
- **Receipts as a first-class citizen**, not a layer added afterwards

---

## License

Licensed under the **MIT License**. See [LICENSE](LICENSE).

## Author

Built by **Behdad**.
