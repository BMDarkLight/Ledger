"""The one rule, in code: a claim without a receipt is not an answer.

Everything here is pure and dependency-free on purpose — this is the part of the
system that must stay testable without an API key, a vector store, or a network.
"""

import re

from api.schemas import AnswerStatus, Claim, Receipt

TAG_PATTERN = re.compile(r"\[([RT]\d+)\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")

# Sentences that carry no factual load and so need no receipt. Kept deliberately
# small: when in doubt a sentence counts as factual and must be cited.
_NON_FACTUAL_PREFIXES = (
    "here is",
    "here are",
    "in short",
    "in summary",
    "to summarize",
    "i cannot",
    "i can't",
    "i don't have",
    "the corpus does not",
    "the corpus doesn't",
    "no source in the corpus",
)


def extract_tags(text: str) -> list[str]:
    """Every citation tag in `text`, in order, deduplicated."""
    seen: dict[str, None] = {}
    for tag in TAG_PATTERN.findall(text):
        seen.setdefault(tag, None)
    return list(seen)


def strip_tags(text: str) -> str:
    """The prose with its citation tags removed, for display or judging."""
    return re.sub(r"\s*\[[RT]\d+\]", "", text).strip()


def split_sentences(text: str) -> list[str]:
    """Cheap sentence split. Good enough for citation accounting, not for NLP."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT.split(cleaned) if s.strip()]


def looks_factual(sentence: str) -> bool:
    """Whether a sentence asserts something that needs backing.

    Heuristic and deliberately conservative: hedges and refusals are exempt,
    everything else is on the hook for a receipt.
    """
    bare = strip_tags(sentence).strip().lower()
    if len(bare.split()) < 3:
        return False
    return not bare.startswith(_NON_FACTUAL_PREFIXES)


def extract_claims(answer: str, receipts: list[Receipt] | None = None) -> list[Claim]:
    """Decompose an answer into factual claims, resolving each one's receipts.

    A tag that doesn't correspond to a real receipt does not count as support —
    a fabricated citation is worse than a missing one.
    """
    known = {r.tag for r in (receipts or [])}
    claims: list[Claim] = []
    for sentence in split_sentences(answer):
        if not looks_factual(sentence):
            continue
        tags = extract_tags(sentence)
        resolved = [t for t in tags if t in known] if receipts is not None else tags
        claims.append(
            Claim(text=strip_tags(sentence), receipt_tags=resolved, supported=bool(resolved))
        )
    return claims


def citation_coverage(claims: list[Claim]) -> float:
    """Share of factual claims carrying at least one valid receipt.

    An answer with no factual claims at all is vacuously fully covered.
    """
    if not claims:
        return 1.0
    return sum(1 for c in claims if c.supported) / len(claims)


def decide_status(claims: list[Claim], receipts: list[Receipt], floor: float) -> AnswerStatus:
    """Verified, unverified, or refused — the gate every answer passes through."""
    if not receipts:
        return AnswerStatus.REFUSED
    return AnswerStatus.VERIFIED if citation_coverage(claims) >= floor else AnswerStatus.UNVERIFIED


REFUSAL_TEXT = (
    "The corpus does not contain an answer to this, and no available tool can "
    "resolve it. Rather than guess, Ledger is declining to answer."
)
