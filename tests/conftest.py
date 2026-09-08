import pytest
from fastapi.testclient import TestClient

from api.config import Settings
from api.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def settings() -> Settings:
    return Settings(llm_api_key="", web_search_api_key="")


@pytest.fixture
def unreachable_store(settings) -> TestClient:
    """A client whose vector store cannot be reached.

    Pinned to a closed port rather than relying on the default URL, so the test
    means the same thing on a machine that happens to be running Qdrant.
    """
    from api.config import get_settings

    app.dependency_overrides[get_settings] = lambda: Settings(
        qdrant_url="http://127.0.0.1:59999", llm_api_key=""
    )
    yield TestClient(app)
    app.dependency_overrides.clear()
