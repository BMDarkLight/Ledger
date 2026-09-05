"""Loading and validating the golden set.

The golden set is the spec. It exists before the pipeline, it is checked into
the repo, and it is what CI grades against — so it gets a schema and a test of
its own rather than being a loose data file.
"""

import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.jsonl"


class Category(str, Enum):
    RETRIEVAL = "retrieval"
    """Answerable directly from the corpus."""

    TOOL = "tool"
    """Needs a live tool — not in the corpus at all."""

    MULTI_HOP = "multi_hop"
    """Needs retrieval, then a tool, chained."""

    ADVERSARIAL = "adversarial"
    """Deliberately unanswerable, false-premise, or an instruction-override trap."""


class Behavior(str, Enum):
    ANSWER = "answer"
    REFUSE = "refuse"


class Case(BaseModel):
    id: str
    category: Category
    question: str = Field(min_length=1)

    expected_route: str = Field(
        description="retrieve | tool | retrieve_then_tool | refuse — graded as routing accuracy."
    )
    expected_behavior: Behavior = Field(
        description="Whether a correct system answers or declines. Graded separately from routing."
    )
    expected_sources: list[str] = Field(
        default_factory=list, description="Corpus doc ids that should surface — graded as recall@k."
    )
    expected_tools: list[str] = Field(default_factory=list)
    answer_contains: list[str] = Field(
        default_factory=list,
        description="Substrings a correct answer must contain. Cheap grading alongside the judge.",
    )
    notes: str = ""

    @model_validator(mode="after")
    def _coherent(self) -> "Case":
        if self.expected_behavior is Behavior.REFUSE and self.answer_contains:
            raise ValueError(f"{self.id}: a refusal case cannot assert answer_contains")
        if self.category is Category.MULTI_HOP and not (
            self.expected_sources and self.expected_tools
        ):
            raise ValueError(f"{self.id}: multi_hop needs both a source and a tool")
        return self


def load_golden_set(path: Path | None = None) -> list[Case]:
    """Parse and validate every case. Raises on a malformed or incoherent line."""
    target = path or GOLDEN_SET_PATH
    cases: list[Case] = []
    for lineno, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        try:
            cases.append(Case(**json.loads(stripped)))
        except Exception as exc:
            raise ValueError(f"{target.name}:{lineno}: {exc}") from exc
    return cases
