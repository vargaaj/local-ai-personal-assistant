from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Literal

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from rag_app.errors import ServiceUnavailableError
from rag_app import ollama_client as client_module
from rag_app.ollama_client import OllamaChatClient, normalize_ollama_base_url


class RouteDecision(BaseModel):
    mode: Literal["general", "documents", "web", "research"]
    reason: str = ""


class FakeChatOllama:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.messages = []
        self.structured_schema = None

    def invoke(self, messages):
        self.messages.append(messages)
        if self.structured_schema is not None:
            return {"mode": "general", "reason": "test"}
        return SimpleNamespace(content="answer")

    def with_structured_output(self, schema):
        structured = FakeChatOllama(**self.kwargs)
        structured.structured_schema = schema
        return structured


class FakeOllamaClient:
    models_response = {"models": [{"model": "nemotron-3-nano:4b"}]}

    def __init__(self, *, host: str):
        self.host = host

    def list(self):
        return self.models_response


@pytest.fixture
def fake_runtime(monkeypatch):
    monkeypatch.setattr(client_module, "ChatOllama", FakeChatOllama)
    monkeypatch.setitem(
        sys.modules,
        "ollama",
        SimpleNamespace(Client=FakeOllamaClient),
    )


def test_normalize_ollama_base_url_strips_legacy_v1():
    assert normalize_ollama_base_url("http://localhost:11434/v1/") == "http://localhost:11434"


def test_ollama_health_parses_model_list(fake_runtime):
    client = _client()

    health = client.health()

    assert health["ok"] is True
    assert health["model_available"] is True
    assert health["available_models"] == ["nemotron-3-nano:4b"]


def test_ollama_client_rejects_unavailable_model(fake_runtime):
    FakeOllamaClient.models_response = {"models": [{"model": "other-model"}]}
    client = _client()

    with pytest.raises(ServiceUnavailableError, match="Ollama model"):
        client.assert_model_available()

    FakeOllamaClient.models_response = {"models": [{"model": "nemotron-3-nano:4b"}]}


def test_ollama_structured_output_validates_schema(fake_runtime):
    client = _client()

    result = client.structured_output([HumanMessage(content="route this")], RouteDecision)

    assert result == RouteDecision(mode="general", reason="test")


def _client() -> OllamaChatClient:
    return OllamaChatClient(
        base_url="http://localhost:11434/v1",
        model="nemotron-3-nano:4b",
        max_tokens=192,
        system_prompt="system",
    )
