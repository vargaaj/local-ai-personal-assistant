from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    qdrant_url: str = "http://qdrant.tail663206.ts.net:6333"
    qdrant_api_key: SecretStr = Field(..., min_length=1)
    qdrant_collection: str = "local_documents"
    rag_docs_root: Path = Field(...)
    rag_manifest_path: Path = Path(".rag_manifest.json")
    rag_user_name: str = "AJ Varga"

    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "gemma4:e4b"
    ollama_max_tokens: int = 192

    mlflow_tracking_uri: str = "http://mlflow.tail663206.ts.net:5000/"
    mlflow_experiment: str = "local-rag-gemma"
    mlflow_prompt_name: str = "local-rag-system-prompt"
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

    @field_validator("ollama_max_tokens")
    @classmethod
    def ollama_max_tokens_must_be_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("ollama_max_tokens must be positive")
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


@lru_cache
def get_settings() -> Settings:
    return Settings()
