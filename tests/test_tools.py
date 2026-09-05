"""Tool execution, and the receipts it produces."""

import pytest

from api.schemas import ReceiptKind
from api.services import tools


def test_calculator_evaluates_arithmetic(settings):
    assert tools.calculator("1327 * 4519", settings) == "5996713"


@pytest.mark.parametrize("expression", ["__import__('os').system('ls')", "open('x')", "a + 1"])
def test_calculator_refuses_anything_that_is_not_arithmetic(settings, expression):
    with pytest.raises(tools.ToolError):
        tools.calculator(expression, settings)


def test_unconfigured_tools_report_themselves_as_disabled(settings):
    specs = {t.name: t for t in tools.list_tools(settings)}
    assert specs["calculator"].enabled is True
    assert specs["web_search"].enabled is False, "no API key set in the test settings"


def test_tool_results_come_back_as_tagged_receipts(settings):
    receipts = tools.run_tools([("calculator", "17 * 3")], settings)
    assert len(receipts) == 1
    assert receipts[0].tag == "T1"
    assert receipts[0].kind is ReceiptKind.TOOL
    assert receipts[0].snippet == "51"


def test_calling_a_disabled_tool_raises_rather_than_degrading(settings):
    with pytest.raises(tools.ToolError):
        tools.run_tools([("web_search", "bitcoin price")], settings)
