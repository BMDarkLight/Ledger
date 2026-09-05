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
- [Roadmap](#roadmap)
- [Design goals](#design-goals)
- [License](#license)

---

## Highlights

| | |
|---|---|
| 🧾 **Receipts only** | Every claim in an answer is tagged `[R#]` (retrieved chunk) or `[T#]` (tool call). No tag, no claim — it gets flagged unverified instead. |
| 🧭 **Transparent routing** | `POST /v1/route` returns the routing decision and rationale without spending a generation call. |
| 🪤 **Adversarial eval set** | The golden set includes unanswerable questions and disguised tool-questions on purpose — not just easy wins. |
| 🔁 **Multi-hop chaining** | The router can sequence retrieval *then* a tool call for questions that need both. |
| 📊 **Living scorecard** | CI re-runs the eval set on every PR and updates a checked-in scorecard — the numbers are never stale. |
| 🔌 **OpenAI-compatible** | `/v1/chat/completions` with streaming, so existing clients point at Ledger unchanged. |

---

## The problem

Most RAG demos are built and judged on one happy-path question. They fall apart the moment a question can't be answered from the corpus, or secretly needs a live tool instead of a document, or needs both chained together. Ledger is built the other way around: the evaluation set — including the questions designed to break it — exists before most of the pipeline code does.

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
Emits a routing rationale — inspectable via /v1/route

==========================
RETRIEVAL              TOOLS
==========================
Qdrant top-k search     Web search
+ cross-encoder rerank  Calculator / code exec

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
Retrieval recall@k · routing accuracy · faithfulness · citation coverage
```

---

## The one rule

Ledger has exactly one non-negotiable behavior, and everything else is built to serve it:

> **A claim without a receipt is not an answer.**

In practice:

- Every generated sentence carrying a fact is checked for a citation tag before it's returned.
- A question the corpus can't answer and no tool can resolve gets a plain refusal — never a confident guess.
- Routing decisions are never hidden inside a single opaque generation call; they're a separate, inspectable step.

---

## Eval harness

The eval set lives in `eval/golden_set.jsonl` as labeled cases across four categories:

| Category | What it tests |
|---|---|
| `retrieval` | Answerable directly from the corpus |
| `tool` | Needs a live tool (price, date, calculation) — not in the corpus at all |
| `multi_hop` | Needs retrieval *then* a tool, chained |
| `adversarial` | Deliberately unanswerable — the correct behavior is a refusal |

Each CI run reports:

- **Retrieval recall@k** — did the right chunk make it into the top k?
- **Routing accuracy** — did the router pick the category it should have?
- **Faithfulness** — does every claim's receipt actually support the claim (LLM-as-judge or `ragas`)?
- **Citation coverage** — % of factual sentences carrying a valid receipt tag

Results are written to `eval/scorecard.md` and committed by CI, so the README badge and the numbers in it stay honest.

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/v1/route` | Routing decision + rationale, no generation |
| `POST` | `/v1/ask` | Full pipeline: route → retrieve/tools → synthesize |
| `POST` | `/v1/ask/receipts` | Same, with receipts returned as structured data separate from prose |
| `POST` | `/v1/documents` | Ingest a document into the vector store |
| `GET` | `/v1/tools` | List available tools |
| `POST` | `/v1/eval/run` | Run the golden set on demand, return a scorecard |
| `POST` | `/v1/chat/completions` | OpenAI-compatible, streaming supported |

---

## Getting started

### Prerequisites

- **Python 3.10+**
- **Qdrant** running locally (`docker run -p 6333:6333 qdrant/qdrant`)
- An OpenAI-compatible endpoint and key

### Install and run

```bash
git clone https://github.com/BMDarkLight/Ledger.git
cd Ledger
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # set LLM_API_KEY, QDRANT_URL
uvicorn api.main:app --reload
```

- Docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest                            # unit tests, no API key needed
python -m eval.run_golden_set     # eval harness — grades the router with no API key,
                                  # runs the full pipeline once LLM_API_KEY is set
```

Without `LLM_API_KEY` the harness runs in **routing-only** mode: it grades the
deterministic baseline router against all 31 golden cases and writes
`eval/scorecard.md`. That is the floor the Phase-2 LLM router has to beat.

---

## Tech stack

- **FastAPI** + **Uvicorn** — HTTP surface, OpenAPI docs
- **LangChain** — retrieval and tool orchestration
- **Qdrant** — vector store
- **Pydantic v2** — schemas and configuration
- **pytest** + **ragas** (or an LLM-judge script) — testing and faithfulness scoring
- **Docker** — packaging

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
│   │   └── synthesis.py     # receipted generation prompt
│   ├── config.py            # environment-backed settings
│   ├── schemas.py           # Receipt, Claim, RouteDecision, ...
│   └── main.py
├── eval/
│   ├── golden_set.jsonl     # labeled eval cases, incl. adversarial
│   ├── golden_set.py        # case schema + loader
│   ├── metrics.py           # recall@k, routing/refusal accuracy, coverage
│   ├── run_golden_set.py
│   └── scorecard.md         # CI-updated results
├── corpus/                  # fetched, not vendored — see corpus/README.md
├── scripts/fetch_corpus.py
├── tests/
├── Dockerfile
├── docker-compose.yml       # Ledger + Qdrant
├── requirements.txt
└── README.md
```

---

## Roadmap

- [x] **Phase 0** — Pick a corpus (Python PEPs), write ~30 golden questions (including adversarial traps) before any pipeline code
- [ ] **Phase 1** — Retrieval-only MVP: ingest, embed, search, rerank, cited answers; baseline recall@k
- [ ] **Phase 2** — Add the agent + router with two tools (web search, calculator/code-exec)
- [ ] **Phase 3** — Receipts and refusal logic; pass the adversarial set
- [ ] **Phase 4** — Multi-hop chaining (retrieval → tool in sequence)
- [ ] **Phase 5** — Docker, CI eval job, scorecard badge, optional live demo

---

## Design goals

- **Evaluation before generation** — the golden set exists before the pipeline does
- **Refuse rather than invent** — an honest refusal beats a confident fabrication
- **Inspectable, not a black box** — routing and receipts are always visible on request
- **Receipts as a first-class citizen** — not a nice-to-have layered on afterward

---

## License

Licensed under the **MIT License**. See [LICENSE](LICENSE).

## Author

Built with ❤️ by **Behdad**