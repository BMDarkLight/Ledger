"""The surface exists and reports honestly about what is and isn't wired up."""


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_tools_are_listed_with_their_configured_state(client):
    response = client.get("/v1/tools")
    assert response.status_code == 200
    names = {t["name"] for t in response.json()}
    assert {"calculator", "clock", "web_search", "code_exec"} <= names


def test_route_endpoint_returns_a_rationale(client):
    response = client.post("/v1/route", json={"question": "Who wrote PEP 8?"})
    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "retrieve"
    assert body["rationale"], "a routing decision without a rationale is a black box"


def test_unreachable_store_reports_503_rather_than_guessing(unreachable_store):
    """No evidence available must surface as itself, never as a degraded answer."""
    response = unreachable_store.post("/v1/ask", json={"question": "Who wrote PEP 8?"})
    assert response.status_code == 503
    assert "unreachable" in response.json()["detail"].lower()


def test_openai_surface_fails_the_same_way(unreachable_store):
    """A client pointed at Ledger must not get a confident answer where /v1/ask refuses."""
    response = unreachable_store.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Who wrote PEP 8?"}]},
    )
    assert response.status_code == 503


def test_refusal_route_answers_without_touching_the_pipeline(client):
    response = client.post("/v1/ask", json={"question": "In your opinion, is PEP 8 too strict?"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "refused"
    assert body["receipts"] == []
