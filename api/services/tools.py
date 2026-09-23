"""Tool registry and execution.

Tools produce T-tagged receipts the same way retrieval produces R-tagged ones.
A tool result that is not recorded as a receipt cannot be cited, and a claim
that cites nothing is not an answer.
"""

import ast
import operator
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from api.config import Settings
from api.schemas import Receipt, ReceiptKind, ToolSpec


class ToolError(RuntimeError):
    """A tool was called and could not produce a result."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    run: Callable[[str, Settings], str]
    requires: str | None = None  # settings attribute that must be non-empty


# calculator

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_EXPRESSION = re.compile(r"\d+(?:\.\d+)?(?:\s*(?:\*\*|[-+*/%])\s*\d+(?:\.\d+)?)+")


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ToolError(f"unsupported expression element: {ast.dump(node)}")


def calculator(expression: str, settings: Settings) -> str:
    """Evaluate an arithmetic expression. No names, no calls, no attribute access."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"not a valid arithmetic expression: {expression!r}") from exc
    return str(_eval_node(tree.body))


def extract_expression(text: str) -> str | None:
    """The longest arithmetic expression written out in `text`, if there is one.

    Only useful when the question already contains the numbers. A question whose
    figures come from the corpus needs `synthesis.plan_tool_call` instead.
    """
    matches = _EXPRESSION.findall(text)
    return max(matches, key=len) if matches else None


def argument_for(name: str, question: str) -> str:
    """What to pass `name` for this question, without asking a model.

    Raises ToolError when the question alone does not determine an argument.
    """
    if name != "calculator":
        return question
    expression = extract_expression(question)
    if expression is None:
        raise ToolError(f"no arithmetic expression is written out in: {question!r}")
    return expression


# clock


def clock(_: str, settings: Settings) -> str:
    """The current UTC date and time, so that "today" questions have a source."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# not yet built


def web_search(query: str, settings: Settings) -> str:
    raise NotImplementedError("Phase 2: web search backend.")


def code_exec(code: str, settings: Settings) -> str:
    raise NotImplementedError("Phase 2: sandboxed code execution.")


REGISTRY: dict[str, Tool] = {
    t.name: t
    for t in (
        Tool("calculator", "Evaluate an arithmetic expression.", calculator),
        Tool("clock", "Current UTC date and time.", clock),
        Tool(
            "web_search",
            "Search the live web for facts not in the corpus.",
            web_search,
            requires="web_search_api_key",
        ),
        Tool("code_exec", "Run a short Python snippet in a sandbox.", code_exec),
    )
}


def is_enabled(tool: Tool, settings: Settings) -> bool:
    return not tool.requires or bool(getattr(settings, tool.requires, ""))


def list_tools(settings: Settings) -> list[ToolSpec]:
    return [
        ToolSpec(name=t.name, description=t.description, enabled=is_enabled(t, settings))
        for t in REGISTRY.values()
    ]


def describe(name: str) -> str:
    tool = REGISTRY.get(name)
    if tool is None:
        raise ToolError(f"unknown tool: {name!r}")
    return tool.description


def run_tools(calls: list[tuple[str, str]], settings: Settings, start: int = 1) -> list[Receipt]:
    """Execute (tool_name, argument) pairs into T-tagged receipts."""
    receipts: list[Receipt] = []
    for i, (name, argument) in enumerate(calls, start=start):
        tool = REGISTRY.get(name)
        if tool is None:
            raise ToolError(f"unknown tool: {name!r}")
        if not is_enabled(tool, settings):
            raise ToolError(f"tool {name!r} is not configured")
        receipts.append(
            Receipt(
                tag=f"T{i}",
                kind=ReceiptKind.TOOL,
                source=name,
                snippet=tool.run(argument, settings),
                metadata={"argument": argument},
            )
        )
    return receipts
