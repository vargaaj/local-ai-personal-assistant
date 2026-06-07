from pathlib import Path
from types import SimpleNamespace

from rag_app import tracing
from rag_app.config import Settings


class FakeAutolog:
    def __init__(self, *, error: Exception | None = None):
        self.called = False
        self.error = error

    def autolog(self):
        self.called = True
        if self.error:
            raise self.error


class FakeMlflow:
    def __init__(self, *, langchain_error: Exception | None = None):
        self.langchain = FakeAutolog(error=langchain_error)
        self.tracking_uri = None
        self.experiment = None

    def set_tracking_uri(self, value):
        self.tracking_uri = value

    def set_experiment(self, value):
        self.experiment = value


def test_configure_mlflow_enables_langchain_autolog(tmp_path: Path, monkeypatch):
    fake_mlflow = FakeMlflow()
    monkeypatch.setattr(tracing, "mlflow", fake_mlflow)
    monkeypatch.setattr(tracing, "_tracking_server_reachable", lambda uri: (True, None))

    status = tracing.configure_mlflow(_settings(tmp_path))

    assert status["configured"] is True
    assert fake_mlflow.langchain.called


def test_configure_mlflow_remains_non_fatal_when_langchain_setup_fails(
    tmp_path: Path,
    monkeypatch,
):
    fake_mlflow = FakeMlflow(langchain_error=RuntimeError("unsupported"))
    monkeypatch.setattr(tracing, "mlflow", fake_mlflow)
    monkeypatch.setattr(tracing, "_tracking_server_reachable", lambda uri: (True, None))

    status = tracing.configure_mlflow(_settings(tmp_path))

    assert status["configured"] is False
    assert "unsupported" in status["error"]


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        qdrant_api_key="secret",
        rag_docs_root=tmp_path,
        fastembed_cuda=False,
        fastembed_providers=[],
    )
