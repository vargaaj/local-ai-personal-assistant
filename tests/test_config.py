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
    assert settings.qdrant_collection == "local_documents"
    assert settings.fastembed_model == "mixedbread-ai/mxbai-embed-large-v1"
    assert settings.fastembed_vector_size == 1024
    assert settings.fastembed_cuda is True
    assert settings.fastembed_providers == ["CUDAExecutionProvider"]
    assert settings.fastembed_device_ids == [0]
    assert settings.rag_manifest_path == Path(".rag_manifest.json")
    assert settings.ollama_max_tokens == 192


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
