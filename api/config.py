"""Runtime configuration, loaded from the environment (see .env.example)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"

    # Embeddings
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "ledger"

    # Retrieval
    retrieval_top_k: int = 10
    rerank_top_n: int = 4
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Tools
    web_search_api_key: str = ""

    # Behavior
    citation_coverage_floor: float = 1.0

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
