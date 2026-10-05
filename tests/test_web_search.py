"""web_search against Tavily, with the HTTP layer replaced by a scripted transport."""

import json

import httpx
import pytest

from api.config import Settings
from api.services import tools


@pytest.fixture
def keyed() -> Settings:
    return Settings(web_search_api_key="tvly-test")


@pytest.fixture
def tavily(monkeypatch):
    """Answer every search with a canned response, recording the requests."""

    def install(status: int = 200, body: dict | None = None) -> list[httpx.Request]:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(status, json=body if body is not None else {})

        monkeypatch.setattr(
            tools, "_search_client", lambda: httpx.Client(transport=httpx.MockTransport(handler))
        )
        return seen

    return install


RESULTS = {
    "query": "latest CPython release",
    "answer": "Tavily's own generated summary, which must not become evidence.",
    "results": [
        {
            "title": "Python Release Python 3.14.0",
            "url": "https://www.python.org/downloads/release/python-3140/",
            "content": "Python 3.14.0 is the newest major release.",
            "score": 0.91,
        },
        {
            "title": "Status of Python versions",
            "url": "https://devguide.python.org/versions/",
            "content": "3.14 is in bugfix status.",
            "score": 0.77,
        },
    ],
}


def test_results_become_one_citable_snippet_with_their_sources(keyed, tavily):
    tavily(body=RESULTS)
    snippet = tools.web_search("latest CPython release", keyed)

    assert "Python 3.14.0 is the newest major release." in snippet
    assert "https://www.python.org/downloads/release/python-3140/" in snippet
    assert "https://devguide.python.org/versions/" in snippet


def test_the_providers_generated_answer_is_never_evidence(keyed, tavily):
    """A summary written by someone else's model is not a source."""
    seen = tavily(body=RESULTS)
    snippet = tools.web_search("latest CPython release", keyed)

    assert "generated summary" not in snippet
    assert json.loads(seen[0].content)["include_answer"] is False


def test_the_request_is_authenticated_and_carries_the_query(keyed, tavily):
    seen = tavily(body=RESULTS)
    tools.web_search("latest CPython release", keyed)

    (request,) = seen
    assert request.url == tools.TAVILY_URL
    assert request.headers["Authorization"] == "Bearer tvly-test"
    assert json.loads(request.content)["query"] == "latest CPython release"


def test_no_results_is_reported_as_such_rather_than_as_an_error(keyed, tavily):
    tavily(body={"query": "q", "results": []})
    assert "no results" in tools.web_search("q", keyed).lower()


@pytest.mark.parametrize("status,reason", [(401, "API key"), (429, "rate"), (500, "500")])
def test_provider_failures_surface_as_tool_errors(keyed, tavily, status, reason):
    tavily(status=status, body={"detail": {"error": "nope"}})
    with pytest.raises(tools.ToolError, match=reason):
        tools.web_search("q", keyed)


def test_an_unreachable_provider_is_a_tool_error(keyed, monkeypatch):
    def refuse(request):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(
        tools, "_search_client", lambda: httpx.Client(transport=httpx.MockTransport(refuse))
    )
    with pytest.raises(tools.ToolError, match="unreachable"):
        tools.web_search("q", keyed)
