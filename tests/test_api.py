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


def test_unimplemented_pipeline_reports_501_rather_than_guessing(client):
    """Until Phase 1 lands, /v1/ask must fail loudly — never fall back to a guess."""
    response = client.post("/v1/ask", json={"question": "Who wrote PEP 8?"})
    assert response.status_code == 501


def test_refusal_route_answers_without_touching_the_pipeline(client):
    response = client.post("/v1/ask", json={"question": "In your opinion, is PEP 8 too strict?"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "refused"
    assert body["receipts"] == []
