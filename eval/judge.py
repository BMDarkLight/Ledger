"""Faithfulness: whether the receipt a claim cites actually supports it.

Citation coverage checks that a claim carries a tag pointing at a real receipt.
It cannot see a claim that cites R1 for something R1 never says. This judge
reads each cited claim next to the text of the receipts it cites, and only
those, and decides whether they state or directly imply it.

It is a model grading a model, so it is a measurement with error bars rather
than a guarantee. Set JUDGE_MODEL to grade with a different model from the one
that wrote the answers, which removes the most obvious bias.
"""

import json
import re
from dataclasses import dataclass

from api.config import Settings
from api.schemas import Claim, Receipt
from api.services import synthesis

JUDGE_PROMPT = """\
You check whether evidence supports claims. Each numbered claim lists the tags
of the receipts it cites. Judge each claim against the text of those receipts
only, not against anything you know.

A claim is supported when its cited receipts state it or directly imply it.
Paraphrase is fine. A claim is not supported when it adds a detail the receipts
do not contain, is more specific than they are, or contradicts them.

Reply with one JSON object and nothing else:
{"verdicts": [{"claim": <number>, "supported": <true or false>, "reason": "<short>"}]}
Give exactly one verdict per claim.
"""


@dataclass(frozen=True)
class Verdict:
    claim: str
    supported: bool
    reason: str


def build_messages(claims: list[Claim], receipts: list[Receipt]) -> list[dict[str, str]]:
    cited = {tag for claim in claims for tag in claim.receipt_tags}
    evidence = "\n\n".join(f"[{r.tag}]\n{r.snippet}" for r in receipts if r.tag in cited)
    numbered = "\n".join(
        f"{i}. {claim.text} (cites {', '.join(claim.receipt_tags)})"
        for i, claim in enumerate(claims, start=1)
    )
    return [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": f"Receipts:\n\n{evidence}\n\nClaims:\n{numbered}"},
    ]


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_verdicts(reply: str, claims: list[Claim]) -> list[Verdict] | None:
    """One verdict per claim, in claim order, or None if the reply cannot give that."""
    match = _JSON_OBJECT.search(reply)
    if not match:
        return None
    try:
        entries = json.loads(match.group(0)).get("verdicts")
    except (json.JSONDecodeError, AttributeError):
        return None
    if not isinstance(entries, list):
        return None

    by_number: dict[int, dict] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("supported"), bool):
            return None
        try:
            by_number[int(entry.get("claim"))] = entry
        except (TypeError, ValueError):
            return None

    if sorted(by_number) != list(range(1, len(claims) + 1)):
        return None
    return [
        Verdict(claim.text, by_number[i]["supported"], str(by_number[i].get("reason", "")))
        for i, claim in enumerate(claims, start=1)
    ]


def judge(claims: list[Claim], receipts: list[Receipt], settings: Settings) -> list[Verdict] | None:
    """Judge every cited claim in one call. None means the judge gave no usable answer.

    Claims without a receipt are left out: citation coverage already counts
    them, and there is nothing to check them against.
    """
    cited = [c for c in claims if c.receipt_tags]
    if not cited:
        return []
    if settings.judge_model:
        settings = settings.model_copy(update={"llm_model": settings.judge_model})
    reply = synthesis._complete(build_messages(cited, receipts), settings)
    return parse_verdicts(reply, cited)


def faithfulness(verdicts: list[Verdict]) -> float | None:
    """Share of judged claims that were supported; None when nothing was judged."""
    if not verdicts:
        return None
    return sum(1 for v in verdicts if v.supported) / len(verdicts)
