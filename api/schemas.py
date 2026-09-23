"""Wire schemas. The receipt types here are the contract the rest of the code serves."""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Route(str, Enum):
    """What the router decided to do with a question."""

    RETRIEVE = "retrieve"
    TOOL = "tool"
    RETRIEVE_THEN_TOOL = "retrieve_then_tool"
    REFUSE = "refuse"


class AnswerStatus(str, Enum):
    VERIFIED = "verified"
    """Every factual claim carries a receipt that supports it."""

    UNVERIFIED = "unverified"
    """At least one factual claim came back without a receipt."""

    REFUSED = "refused"
    """Nothing in the corpus and no tool could ground an answer."""


class ReceiptKind(str, Enum):
    RETRIEVAL = "retrieval"
    TOOL = "tool"


class Receipt(BaseModel):
    """A single piece of evidence, referenced from prose by its `tag`."""

    tag: str = Field(description="Citation tag as it appears in the prose, e.g. 'R3' or 'T1'.")
    kind: ReceiptKind
    source: str = Field(description="Document id for retrieval, tool name for tool calls.")
    snippet: str = Field(description="The evidence itself: the text, or the tool's output.")
    score: float | None = Field(default=None, description="Rerank score, retrieval receipts only.")
    metadata: dict[str, Any] = Field(default_factory=dict)


class Claim(BaseModel):
    """One factual sentence from the answer, with the receipts backing it."""

    text: str
    receipt_tags: list[str] = Field(default_factory=list)
    supported: bool = Field(description="True when at least one cited receipt exists.")


class RouteDecision(BaseModel):
    route: Route
    rationale: str = Field(description="Why this route. Always returned, never hidden.")
    tools: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = None


class RouteRequest(BaseModel):
    question: str = Field(min_length=1)


class AskResponse(BaseModel):
    answer: str
    status: AnswerStatus
    route: RouteDecision
    receipts: list[Receipt] = Field(default_factory=list)
    citation_coverage: float = Field(
        ge=0.0, le=1.0, description="Share of factual sentences carrying a valid receipt."
    )


class ReceiptedAskResponse(AskResponse):
    """`/v1/ask/receipts`: the same answer, decomposed claim by claim."""

    claims: list[Claim] = Field(default_factory=list)


class DocumentRequest(BaseModel):
    doc_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentResponse(BaseModel):
    doc_id: str
    chunks_indexed: int


class ToolSpec(BaseModel):
    name: str
    description: str
    enabled: bool


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
