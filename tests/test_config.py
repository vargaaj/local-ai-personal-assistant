from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_app.config import Settings


def test_settings_require_qdrant_api_key(tmp_path: Path):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, rag_docs_root=tmp_path)


def test_settings_require_absolute_docs_root():
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            qdrant_api_key="secret",
            rag_docs_root=Path("relative"),
        )


def test_settings_defaults_use_hosted_mlflow(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        qdrant_api_key="secret",
        rag_docs_root=tmp_path,
    )

    assert settings.mlflow_tracking_uri == "http://mlflow.tail663206.ts.net:5000/"
    assert settings.mlflow_experiment == "local-ai-assistant"
    assert settings.mlflow_prompt_name == "local-ai-assistant-system-prompt"
    assert settings.qdrant_collection == "local_documents"
    assert settings.fastembed_model == "mixedbread-ai/mxbai-embed-large-v1"
    assert settings.fastembed_vector_size == 1024
    assert settings.fastembed_cuda is True
    assert settings.fastembed_providers == ["CUDAExecutionProvider"]
    assert settings.fastembed_device_ids == [0]
    assert settings.rag_manifest_path == Path(".rag_manifest.json")
    assert settings.rag_state_db_path == Path(".rag_state.sqlite3")
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.ollama_model == "nemotron-3-nano:4b"
    assert settings.ollama_max_tokens == 192
    assert settings.agent_max_concurrent_runs == 1
    assert settings.retrieval_score_threshold == 0.35


def test_settings_parse_fastembed_lists(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        qdrant_api_key="secret",
        rag_docs_root=tmp_path,
        fastembed_providers="CUDAExecutionProvider,CPUExecutionProvider",
        fastembed_device_ids="0,1",
    )

    assert settings.fastembed_providers == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert settings.fastembed_device_ids == [0, 1]


def test_settings_reject_invalid_retrieval_score_threshold(tmp_path: Path):
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            qdrant_api_key="secret",
            rag_docs_root=tmp_path,
            retrieval_score_threshold=1.1,
        )


def test_settings_reject_chunk_overlap_not_smaller_than_chunk_size(tmp_path: Path):
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            qdrant_api_key="secret",
            rag_docs_root=tmp_path,
            chunk_size=200,
            chunk_overlap=200,
        )


def test_settings_reject_unsupported_research_query_count(tmp_path: Path):
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            qdrant_api_key="secret",
            rag_docs_root=tmp_path,
            research_query_count=9,
        )
