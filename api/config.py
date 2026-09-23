"""Runtime configuration, loaded from the environment (see .env.example)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_timeout: float = 60.0

    # Embeddings. fastembed runs these locally via ONNX, with no API key, which
    # is what lets CI grade retrieval on every pull request.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "ledger"

    # Retrieval
    retrieval_top_k: int = 10
    rerank_top_n: int = 4
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    model_cache_dir: str = ""
    """Where ONNX model files are stored. Empty means fastembed's default, a
    temp directory, which is fine locally but useless to a CI cache."""

    # Tools
    web_search_api_key: str = ""

    # Behavior
    citation_coverage_floor: float = 1.0

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key)

    @property
    def qdrant_in_memory(self) -> bool:
        return self.qdrant_url == ":memory:"


@lru_cache
def get_settings() -> Settings:
    return Settings()
