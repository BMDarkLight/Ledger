"""The code_exec sandbox: what it runs, and what it refuses to let a snippet do."""

import pytest

from api.services import tools


def run(code: str, settings) -> str:
    return tools.code_exec(code, settings)


def test_an_expression_prints_its_value(settings):
    assert run("math.sqrt(1764)", settings) == "42.0"


def test_statements_report_what_they_print(settings):
    assert run("x = math.factorial(5); print(x + 1)", settings) == "121"


def test_a_failing_snippet_is_a_tool_error_naming_the_exception(settings):
    with pytest.raises(tools.ToolError, match="ZeroDivisionError"):
        run("1 / 0", settings)


def test_a_runaway_loop_is_stopped(settings, monkeypatch):
    monkeypatch.setattr(tools, "CODE_EXEC_TIMEOUT_S", 1.0)
    with pytest.raises(tools.ToolError, match="time"):
        run("while True: pass", settings)


def test_a_snippet_cannot_write_files(settings, tmp_path):
    target = tmp_path / "written.txt"
    with pytest.raises(tools.ToolError, match="File too large"):
        run(f"f = open({str(target)!r}, 'w'); f.write('x' * 10); f.close()", settings)
    assert not target.exists() or target.stat().st_size == 0


def test_a_swallowed_write_error_still_writes_nothing(settings, tmp_path):
    """Without an explicit close the flush error is lost, but no bytes land."""
    target = tmp_path / "written.txt"
    run(f"open({str(target)!r}, 'w').write('x' * 10)", settings)
    assert not target.exists() or target.stat().st_size == 0


def test_a_snippet_cannot_start_processes(settings):
    with pytest.raises(tools.ToolError):
        run("import subprocess; subprocess.run(['true'])", settings)


def test_a_snippet_does_not_see_the_server_environment(settings, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-should-not-leak")
    assert "sk-should-not-leak" not in run("import os; print(dict(os.environ))", settings)


def test_output_is_truncated(settings):
    assert len(run("print('x' * 100_000)", settings)) <= tools.CODE_EXEC_MAX_OUTPUT + 20


def test_code_exec_never_receives_the_raw_question():
    """A question is not Python. The argument has to be written by the planner."""
    with pytest.raises(tools.ToolError):
        tools.argument_for("code_exec", "What is the square root of 1764?")
