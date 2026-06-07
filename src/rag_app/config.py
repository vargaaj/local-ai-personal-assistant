from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    qdrant_url: str = "http://qdrant.tail663206.ts.net:6333"
    qdrant_api_key: SecretStr = Field(..., min_length=1)
    qdrant_collection: str = "local_documents"
    rag_docs_root: Path = Field(...)
    rag_manifest_path: Path = Path(".rag_manifest.json")
    rag_state_db_path: Path = Path(".rag_state.sqlite3")
    rag_user_name: str = "AJ Varga"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "nemotron-3-nano:4b"
    ollama_max_tokens: int = 192
    ollama_api_key: SecretStr | None = None

    mlflow_tracking_uri: str = "http://mlflow.tail663206.ts.net:5000/"
    mlflow_experiment: str = "local-ai-assistant"
    mlflow_prompt_name: str = "local-ai-assistant-system-prompt"
    mlflow_prompt_alias: str = "production"

    fastembed_model: str = "mixedbread-ai/mxbai-embed-large-v1"
    fastembed_vector_size: int = 1024
    fastembed_cuda: bool = True
    fastembed_providers: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["CUDAExecutionProvider"]
    )
    fastembed_device_ids: Annotated[list[int] | None, NoDecode] = Field(
        default_factory=lambda: [0]
    )
    fastembed_batch_size: int = 256
    chunk_size: int = 1200
    chunk_overlap: int = 200
    retrieval_score_threshold: float = 0.35
    agent_max_concurrent_runs: int = 1
    memory_fact_limit: int = 100
    web_search_max_results: int = 5
    research_query_count: int = 3

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("rag_docs_root")
    @classmethod
    def docs_root_must_be_absolute(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("RAG_DOCS_ROOT must be an absolute path")
        return value

    @field_validator("chunk_overlap")
    @classmethod
    def chunk_overlap_must_be_smaller(cls, value: int) -> int:
        if value < 0:
            raise ValueError("chunk_overlap must be non-negative")
        return value

    @field_validator("chunk_size")
    @classmethod
    def chunk_size_must_be_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("chunk_size must be positive")
        return value

    @model_validator(mode="after")
    def chunk_overlap_must_be_smaller_than_chunk_size(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    @field_validator("retrieval_score_threshold")
    @classmethod
    def retrieval_score_threshold_must_be_cosine_range(cls, value: float) -> float:
        if not -1 <= value <= 1:
            raise ValueError("retrieval_score_threshold must be between -1 and 1")
        return value

    @field_validator("ollama_max_tokens")
    @classmethod
    def ollama_max_tokens_must_be_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("ollama_max_tokens must be positive")
        return value

    @field_validator(
        "agent_max_concurrent_runs",
        "memory_fact_limit",
        "web_search_max_results",
        "research_query_count",
    )
    @classmethod
    def positive_integer_settings(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("value must be positive")
        return value

    @field_validator("web_search_max_results")
    @classmethod
    def web_search_max_results_must_be_supported(cls, value: int) -> int:
        if value > 10:
            raise ValueError("web_search_max_results must not exceed 10")
        return value

    @field_validator("research_query_count")
    @classmethod
    def research_query_count_must_be_supported(cls, value: int) -> int:
        if value > 8:
            raise ValueError("research_query_count must not exceed 8")
        return value

    @field_validator("fastembed_batch_size")
    @classmethod
    def fastembed_batch_size_must_be_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("fastembed_batch_size must be positive")
        return value

    @field_validator("fastembed_providers", mode="before")
    @classmethod
    def parse_fastembed_providers(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("fastembed_device_ids", mode="before")
    @classmethod
    def parse_fastembed_device_ids(cls, value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped or stripped.lower() in {"none", "null"}:
                return None
            return [int(item.strip()) for item in stripped.split(",") if item.strip()]
        return value

    @property
    def qdrant_api_key_value(self) -> str:
        return self.qdrant_api_key.get_secret_value()

    @property
    def ollama_api_key_value(self) -> str | None:
        if self.ollama_api_key is None:
            return None
        value = self.ollama_api_key.get_secret_value().strip()
        return value or None


@lru_cache
def get_settings() -> Settings:
    return Settings()
