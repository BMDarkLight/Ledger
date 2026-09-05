"""Turn receipts into prose that cites them — and nothing else.

The system prompt is the enforcement point at generation time; `receipts.py` is
the enforcement point after generation. Both exist because the first one is a
request and the second one is a guarantee.
"""

from api.config import Settings
from api.schemas import Receipt

SYSTEM_PROMPT = """\
You answer strictly from the receipts supplied below. You have no other knowledge.

Rules, in order of precedence:
1. Every sentence that asserts a fact must end with the tag of the receipt that
   supports it, e.g. "PEP 8 was authored by Guido van Rossum [R2]."
2. A sentence may cite more than one receipt: "... [R1][T2]".
3. If the receipts do not support a claim, do not make the claim.
4. If the receipts do not answer the question at all, reply with exactly:
   INSUFFICIENT_EVIDENCE
5. Never invent a tag. Only tags listed in the receipts below exist.
"""

INSUFFICIENT = "INSUFFICIENT_EVIDENCE"


def format_receipts(receipts: list[Receipt]) -> str:
    """Render receipts into the block the model is allowed to draw from."""
    return "\n\n".join(f"[{r.tag}] ({r.kind.value} · {r.source})\n{r.snippet}" for r in receipts)


def build_messages(question: str, receipts: list[Receipt]) -> list[dict[str, str]]:
    """The exact message list sent to the model — inspectable, not buried."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Receipts:\n\n{format_receipts(receipts)}\n\nQuestion: {question}",
        },
    ]


def synthesize(question: str, receipts: list[Receipt], settings: Settings) -> str:
    """Generate a receipted answer. Returns INSUFFICIENT when nothing supports one."""
    raise NotImplementedError("Phase 1: call the OpenAI-compatible endpoint with build_messages().")
