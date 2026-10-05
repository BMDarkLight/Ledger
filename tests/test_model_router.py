"""The model-backed router, with the model replaced by scripted replies."""

import json

import pytest

from api.config import Settings
from api.schemas import Route
from api.services import model_router, synthesis
from api.services import router as routing
from api.services.model_router import UnusableReply, parse_decision
from eval.golden_set import load_golden_set


@pytest.fixture
def model_settings() -> Settings:
    return Settings(llm_api_key="test-key", router="model")


@pytest.fixture
def model(monkeypatch):
    """Script the router's replies, and record what it was sent."""
    calls: list[list[dict]] = []
    replies: list[str] = []

    def fake(messages, settings):
        calls.append(messages)
        return replies.pop(0)

    monkeypatch.setattr(synthesis, "_complete", fake)
    return replies, calls


def reply(route: str, tools=(), rationale="Because.", **extra) -> str:
    return json.dumps({"route": route, "tools": list(tools), "rationale": rationale, **extra})


# --- parsing ----------------------------------------------------------------


def test_a_well_formed_reply_becomes_a_decision():
    decision = parse_decision(reply("tool", ["code_exec"], confidence=0.9))
    assert (decision.route, decision.tools, decision.confidence) == (Route.TOOL, ["code_exec"], 0.9)


def test_wrapping_around_the_json_is_tolerated():
    decision = parse_decision("Sure:\n```json\n" + reply("retrieve") + "\n```")
    assert decision.route is Route.RETRIEVE


@pytest.mark.parametrize(
    "text,reason",
    [
        ("I think you should retrieve.", "no JSON"),
        ('{"route": "retrieve", ', "no JSON"),
        (reply("guess"), "unknown route"),
        (reply("tool", ["teleport"]), "unknown tools"),
        (reply("tool", []), "names no tools"),
        (reply("retrieve", rationale=""), "no rationale"),
    ],
)
def test_an_unusable_reply_is_rejected_with_the_reason(text, reason):
    with pytest.raises(UnusableReply, match=reason):
        parse_decision(text)


def test_tools_listed_against_a_route_that_runs_none_are_dropped():
    assert parse_decision(reply("retrieve", ["calculator"])).tools == []


def test_confidence_is_clamped_and_duplicates_removed():
    decision = parse_decision(reply("tool", ["clock", "clock"], confidence=7))
    assert decision.confidence == 1.0
    assert decision.tools == ["clock"]


# --- deciding ---------------------------------------------------------------


def test_the_model_decides_when_the_router_setting_asks_for_it(model, model_settings):
    replies, _ = model
    replies.append(
        reply("retrieve_then_tool", ["clock", "calculator"], "Needs a date, then maths.")
    )
    decision = routing.route("How long ago was PEP 8 written?", model_settings)
    assert decision.route is Route.RETRIEVE_THEN_TOOL
    assert decision.rationale == "Needs a date, then maths."


def test_an_unusable_reply_falls_back_to_the_baseline_and_says_so(model, model_settings):
    replies, _ = model
    replies.append("retrieve, probably")
    decision = routing.route("Who are the authors of PEP 8?", model_settings)

    assert decision.route is routing.decide("Who are the authors of PEP 8?").route
    assert decision.rationale.startswith("The model router's reply was unusable")
    assert decision.confidence <= 0.3


def test_the_baseline_setting_never_calls_the_model(model, settings):
    _, calls = model
    routing.route("Who are the authors of PEP 8?", settings)
    assert calls == []


def test_no_key_means_the_model_router_is_unavailable_not_silently_skipped():
    with pytest.raises(synthesis.SynthesisError):
        routing.route("Who wrote PEP 8?", Settings(llm_api_key="", router="model"))


def test_the_prompt_contains_nothing_from_the_golden_set(model, model_settings):
    """What keeps the model router's score honest, unlike the baseline's."""
    replies, calls = model
    replies.append(reply("retrieve"))
    routing.route("Anything at all?", model_settings)

    system_prompt = calls[0][0]["content"].lower()
    leaked = [c.id for c in load_golden_set() if c.question.lower() in system_prompt]
    assert leaked == []


def test_the_prompt_describes_every_tool(model, model_settings):
    replies, calls = model
    replies.append(reply("retrieve"))
    routing.route("Anything at all?", model_settings)
    assert all(name in calls[0][0]["content"] for name in model_router.tools.REGISTRY)


# --- surfaces ---------------------------------------------------------------


def test_route_endpoint_reports_an_unavailable_model_as_503(client):
    from api.config import get_settings
    from api.main import app

    app.dependency_overrides[get_settings] = lambda: Settings(llm_api_key="", router="model")
    try:
        response = client.post("/v1/route", json={"question": "Who wrote PEP 8?"})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 503


def test_the_eval_names_the_router_it_graded(model, model_settings):
    from datetime import datetime, timezone

    from eval.run_golden_set import caption, run

    replies, calls = model
    cases = [c for c in load_golden_set() if c.id in ("R001", "T006")]
    replies += [reply("retrieve"), reply("tool", ["code_exec"])]

    card = run("routing", model_settings, cases)

    assert len(calls) == 2, "one routing call per case"
    assert card.routing_accuracy == 1.0
    assert "model (gpt-4o-mini)" in caption("routing", datetime.now(timezone.utc), model_settings)
