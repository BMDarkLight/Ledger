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
