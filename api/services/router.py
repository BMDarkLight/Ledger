"""Routing: retrieve, call a tool, or chain both — and always say why.

This is the deterministic Phase-1 baseline. It runs with no API key, which makes
it the floor that the Phase-2 LLM router has to beat on the golden set. Replace
`decide()` with the model-backed version; keep the rationale contract.
"""

import re

from api.schemas import Route, RouteDecision

# Questions whose answer changes with the wall clock or the outside world.
_LIVE_MARKERS = (
    "today",
    "right now",
    "currently",
    "current",
    "latest",
    "this week",
    "this month",
    "this year",
    "as of now",
    "up to date",
    "recent",
    "price",
    "stock",
    "weather",
    "exchange rate",
    "how old is",
)

# Questions asking for a computation rather than a fact.
_ARITHMETIC = re.compile(
    r"(\d+\s*[-+*/^]\s*\d+)|\bhow many\b"
    r"|\b(calculate|compute|percent(age)? of|square root|divided by|multiplied by)\b",
    re.IGNORECASE,
)

# Questions that no corpus and no tool can settle, because they aren't factual.
_UNANSWERABLE = re.compile(
    r"\b(will .* (ever|eventually)|do you think|in your opinion|which is better"
    r"|should i|predict|going to happen)\b",
    re.IGNORECASE,
)


def _has_live_marker(q: str) -> bool:
    return any(marker in q for marker in _LIVE_MARKERS)


def decide(question: str) -> RouteDecision:
    """Pick a route for `question` and explain the choice."""
    q = question.lower()

    if _UNANSWERABLE.search(q):
        return RouteDecision(
            route=Route.REFUSE,
            rationale=(
                "The question asks for an opinion or a prediction, which no document "
                "and no tool can supply as a fact."
            ),
            confidence=0.6,
        )

    wants_computation = bool(_ARITHMETIC.search(q))
    wants_live = _has_live_marker(q)

    # A live/computed question that also names something the corpus knows about
    # needs both, in sequence: look the fact up, then compute or check against it.
    if (wants_computation or wants_live) and _mentions_corpus_subject(q):
        tools = ["calculator"] if wants_computation else ["web_search"]
        if "today" in q or "current date" in q:
            tools = ["clock", *tools]
        return RouteDecision(
            route=Route.RETRIEVE_THEN_TOOL,
            rationale=(
                "The question names a subject the corpus covers but resolves to a value "
                "that must be computed or looked up live — retrieve the fact first, then "
                "chain the tool."
            ),
            tools=tools,
            confidence=0.55,
        )

    if wants_computation:
        return RouteDecision(
            route=Route.TOOL,
            rationale="Pure computation — no document contains this; the calculator does.",
            tools=["calculator"],
            confidence=0.7,
        )

    if wants_live:
        tools = ["clock"] if ("today" in q or "current date" in q) else ["web_search"]
        return RouteDecision(
            route=Route.TOOL,
            rationale="The answer depends on the present moment or the live outside world.",
            tools=tools,
            confidence=0.65,
        )

    return RouteDecision(
        route=Route.RETRIEVE,
        rationale="A static factual question — the corpus is the right place to look.",
        tools=[],
        confidence=0.6,
    )


# Corpus-specific vocabulary. Swap this when the corpus changes; it is the only
# place in the router that knows what Ledger has been fed.
_CORPUS_SUBJECT = re.compile(r"\b(pep[\s-]?\d+|pep\b|python|guido|typing|asyncio)\b", re.IGNORECASE)


def _mentions_corpus_subject(q: str) -> bool:
    return bool(_CORPUS_SUBJECT.search(q))
