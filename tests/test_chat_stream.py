"""Streaming on the OpenAI-compatible surface: the gate runs before a byte is sent."""

import json

import pytest

from api.routers import chat
from api.schemas import (
    AnswerStatus,
    Receipt,
    ReceiptedAskResponse,
    ReceiptKind,
    Route,
    RouteDecision,
)

ANSWER = "PEP 8 limits lines to 79 characters [R1].\n\nDocstrings get 72 [R1]. Done (see R1)."


@pytest.fixture
def answered(monkeypatch):
    result = ReceiptedAskResponse(
        answer=ANSWER,
        status=AnswerStatus.VERIFIED,
        route=RouteDecision(route=Route.RETRIEVE, rationale="test"),
        receipts=[Receipt(tag="R1", kind=ReceiptKind.RETRIEVAL, source="pep-0008", snippet="...")],
        citation_coverage=1.0,
    )
    monkeypatch.setattr(chat, "_guarded", lambda request, settings: result)
    return result


def stream(client) -> list:
    response = client.post(
        "/v1/chat/completions",
        json={"stream": True, "messages": [{"role": "user", "content": "Line length?"}]},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [line.removeprefix("data: ") for line in response.text.split("\n\n") if line]
    return [e if e == "[DONE]" else json.loads(e) for e in events]


def test_the_stream_reassembles_to_exactly_the_checked_answer(client, answered):
    events = stream(client)
    text = "".join(e["choices"][0]["delta"].get("content", "") for e in events[:-1])
    assert text == ANSWER, "line breaks and spacing survive the chunking"


def test_each_claim_arrives_with_its_tag_in_the_same_chunk(client, answered):
    contents = [e["choices"][0]["delta"].get("content") for e in stream(client)[1:-2]]
    assert contents[0].strip() == "PEP 8 limits lines to 79 characters [R1]."
    assert contents[1].strip() == "Docstrings get 72 [R1]."


def test_the_receipts_ride_on_the_final_chunk_before_done(client, answered):
    events = stream(client)
    assert events[-1] == "[DONE]"
    final = events[-2]
    assert final["choices"][0]["finish_reason"] == "stop"
    assert final["ledger"]["status"] == "verified"
    assert [r["tag"] for r in final["ledger"]["receipts"]] == ["R1"]
    assert all(e["object"] == "chat.completion.chunk" for e in events[:-1])


def test_a_failing_pipeline_is_an_http_error_not_a_broken_stream(unreachable_store):
    response = unreachable_store.post(
        "/v1/chat/completions",
        json={"stream": True, "messages": [{"role": "user", "content": "Who wrote PEP 8?"}]},
    )
    assert response.status_code == 503
