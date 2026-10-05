"""Routing: retrieve, call a tool, or chain both, and always say why.

`decide()` is the deterministic baseline. It runs with no API key, which makes
it the floor that the model-backed router (`model_router.py`) has to beat on the
golden set. `route()` picks between them according to the ROUTER setting; both
keep the rationale contract.
"""

import re

from api.config import Settings
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
    r"(\d+\s*(\*\*|[-+*/^%])\s*\d+)"
    r"|\b(calculate|compute|percent(age)? of|divided by|multiplied by)\b",
    re.IGNORECASE,
)

# Maths beyond what the calculator evaluates (+ - * / % **, see tools._OPS).
# Matched on capability, not on phrasing: these need real code.
_BEYOND_CALCULATOR = re.compile(
    r"\b(square root|cube root|nth root|sqrt|logarithm|log of|factorial"
    r"|sine|cosine|tangent|prime)\b",
    re.IGNORECASE,
)

# Time elapsed up to the present, a computation that needs the wall clock.
_ELAPSED = re.compile(
    r"\b(years?|months?|weeks?|days?|hours?)\s+ago\b|\bhow long ago\b", re.IGNORECASE
)

# "How many" is only arithmetic when the question supplies something to count
# with: a number (PEP numbers excluded, since they are names, not operands), or a
# span between two things. Otherwise it is asking to count things in the world.
_HOW_MANY = re.compile(r"\bhow many\b", re.IGNORECASE)
_OPERAND = re.compile(
    r"\b(\d+(\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|twelve"
    r"|dozen|hundred|thousand)\b"
    r"|\b(separate|between|difference|apart)\b",
    re.IGNORECASE,
)
_PEP_NUMBER = re.compile(r"\bpep[\s-]?\d+", re.IGNORECASE)

# Questions that no corpus and no tool can settle, because they aren't factual.
_UNANSWERABLE = re.compile(
    r"\b(will .* (ever|eventually)|do you think|in your opinion|which is better"
    r"|should i|predict|going to happen)\b",
    re.IGNORECASE,
)


def _has_live_marker(q: str) -> bool:
    return any(marker in q for marker in _LIVE_MARKERS)


def _counts_with_operands(q: str) -> bool:
    return bool(_OPERAND.search(_PEP_NUMBER.sub("", q)))


def _needs_clock(q: str) -> bool:
    return "today" in q or "current date" in q or bool(_ELAPSED.search(q))


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

    beyond_calculator = bool(_BEYOND_CALCULATOR.search(q))
    counting = bool(_HOW_MANY.search(q))
    wants_computation = (
        bool(_ARITHMETIC.search(q))
        or beyond_calculator
        or bool(_ELAPSED.search(q))
        or (counting and _counts_with_operands(q))
    )
    # A count with nothing to compute from is a fact about the world, and the
    # world is live: it can only be looked up, never derived.
    wants_live = _has_live_marker(q) or (counting and not wants_computation)
    compute_tool = "code_exec" if beyond_calculator else "calculator"

    # A live or computed question that also names something the corpus knows
    # about needs both, in sequence: look the fact up, then compute against it.
    if (wants_computation or wants_live) and _mentions_corpus_subject(q):
        tools = [compute_tool] if wants_computation else ["web_search"]
        if _needs_clock(q):
            tools = ["clock", *tools]
        return RouteDecision(
            route=Route.RETRIEVE_THEN_TOOL,
            rationale=(
                "The question names a subject the corpus covers but resolves to a value "
                "that must be computed or looked up live, so retrieve the fact first and "
                "chain the tool onto it."
            ),
            tools=tools,
            confidence=0.55,
        )

    if wants_computation:
        tools = ["clock", compute_tool] if _needs_clock(q) else [compute_tool]
        return RouteDecision(
            route=Route.TOOL,
            rationale=(
                "Pure computation. No document contains this; "
                + (
                    "it needs real code, beyond the calculator."
                    if beyond_calculator
                    else "the calculator does."
                )
            ),
            tools=tools,
            confidence=0.7,
        )

    if wants_live:
        tools = ["clock"] if _needs_clock(q) else ["web_search"]
        return RouteDecision(
            route=Route.TOOL,
            rationale="The answer depends on the present moment or the live outside world.",
            tools=tools,
            confidence=0.65,
        )

    return RouteDecision(
        route=Route.RETRIEVE,
        rationale="A static factual question, so the corpus is the right place to look.",
        tools=[],
        confidence=0.6,
    )


# Corpus-specific knowledge. Swap these when the corpus changes; they are the
# only place in routing that knows what Ledger has been fed.
CORPUS_DESCRIPTION = (
    "Python Enhancement Proposals (PEPs), Python's design documents. They cover "
    "code style, docstrings, type hints, packaging metadata, the PEP process "
    "itself and individual language features, and each one records its number, "
    "title, authors, status, type and creation date."
)
_CORPUS_SUBJECT = re.compile(r"\b(pep[\s-]?\d+|pep\b|python|guido|typing|asyncio)\b", re.IGNORECASE)


def _mentions_corpus_subject(q: str) -> bool:
    return bool(_CORPUS_SUBJECT.search(q))


def route(question: str, settings: Settings) -> RouteDecision:
    """The routing decision, made by whichever router ROUTER selects."""
    if settings.router == "model":
        from api.services import model_router

        return model_router.decide(question, settings, CORPUS_DESCRIPTION, fallback=decide)
    return decide(question)
