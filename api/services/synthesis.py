"""Generation: turn receipts into prose that cites them.

The system prompt asks for a citation on every factual sentence. A model is
free to ignore that, so `receipts.py` re-checks the output afterwards.
"""

from functools import lru_cache

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

TOOL_ARGUMENT_PROMPT = """\
You prepare the input for one tool call.

Tool: {tool}
Purpose: {description}

Reply with the tool input on a single line and nothing else. Reply with NONE if
the question and the evidence below do not determine an input.

For the calculator, the input must be an arithmetic expression built only from
digits, parentheses and the operators + - * / % **. Substitute any figure you
need from the evidence, so that the expression contains no words.
"""

INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
NO_ARGUMENT = "NONE"


class SynthesisError(RuntimeError):
    """The model could not be reached, or returned nothing usable."""


def format_receipts(receipts: list[Receipt]) -> str:
    """Render receipts into the block the model is allowed to draw from."""
    return "\n\n".join(f"[{r.tag}] ({r.kind.value}, {r.source})\n{r.snippet}" for r in receipts)


def build_messages(question: str, receipts: list[Receipt]) -> list[dict[str, str]]:
    """The exact message list sent to the model, kept inspectable."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Receipts:\n\n{format_receipts(receipts)}\n\nQuestion: {question}",
        },
    ]


@lru_cache(maxsize=4)
def _client(base_url: str, api_key: str, timeout: float):
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)


def _complete(messages: list[dict[str, str]], settings: Settings) -> str:
    """One call to the configured OpenAI-compatible endpoint.

    Isolated so tests can replace it without an API key or a network.
    """
    if not settings.llm_configured:
        raise SynthesisError("LLM_API_KEY is not set, so no answer can be generated")

    client = _client(settings.llm_base_url, settings.llm_api_key, settings.llm_timeout)
    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            temperature=0,
        )
    except Exception as exc:
        raise SynthesisError(f"{settings.llm_base_url} could not be reached: {exc}") from exc
    return response.choices[0].message.content or ""


def synthesize(question: str, receipts: list[Receipt], settings: Settings) -> str:
    """Generate a receipted answer, or INSUFFICIENT when nothing supports one."""
    if not receipts:
        return INSUFFICIENT

    answer = _complete(build_messages(question, receipts), settings).strip()
    if not answer:
        raise SynthesisError("the model returned an empty completion")
    return INSUFFICIENT if INSUFFICIENT in answer else answer


def plan_tool_call(
    question: str, tool: str, description: str, evidence: list[Receipt], settings: Settings
) -> str | None:
    """Work out what to pass a tool, given the question and the evidence so far.

    This is what makes a chained question answerable: the retrieved figure is
    only in the receipts, so the expression to compute has to be written after
    retrieval rather than pulled out of the question with a regex. Returns None
    when the model declines to produce an input.
    """
    prompt = TOOL_ARGUMENT_PROMPT.format(tool=tool, description=description)
    evidence_block = format_receipts(evidence) if evidence else "(none)"
    reply = _complete(
        [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"Evidence:\n\n{evidence_block}\n\nQuestion: {question}",
            },
        ],
        settings,
    ).strip()

    first_line = reply.splitlines()[0].strip() if reply else ""
    if not first_line or first_line.upper().startswith(NO_ARGUMENT):
        return None
    return first_line
