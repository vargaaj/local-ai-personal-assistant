from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from .errors import ServiceUnavailableError
from .tracing import timed, trace

StructuredOutput = TypeVar("StructuredOutput", bound=BaseModel)


def normalize_ollama_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3]
    return normalized.rstrip("/")


class OllamaChatClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        max_tokens: int,
        system_prompt: str,
        system_prompt_metadata: dict[str, str] | None = None,
    ) -> None:
        self.base_url = normalize_ollama_base_url(base_url)
        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.system_prompt_metadata = system_prompt_metadata or {}
        self.client = ChatOllama(
            base_url=self.base_url,
            model=model,
            num_predict=max_tokens,
            temperature=0.1,
        )

    @trace
    def health(self) -> dict[str, Any]:
        try:
            from ollama import Client

            with timed("ollama.health", model=self.model):
                response = Client(host=self.base_url).list()
            raw_models = getattr(response, "models", None)
            if raw_models is None and isinstance(response, dict):
                raw_models = response.get("models", [])
            available = sorted(
                str(
                    getattr(item, "model", None)
                    or getattr(item, "name", None)
                    or (item.get("model") if isinstance(item, dict) else "")
                    or (item.get("name") if isinstance(item, dict) else "")
                )
                for item in (raw_models or [])
            )
            return {
                "ok": self.model in available,
                "base_url": self.base_url,
                "model": self.model,
                "model_available": self.model in available,
                "available_models": available,
            }
        except Exception as exc:
            return {
                "ok": False,
                "base_url": self.base_url,
                "model": self.model,
                "model_available": False,
                "available_models": [],
                "error": str(exc),
            }

    @trace
    def assert_model_available(self) -> None:
        health = self.health()
        if not health.get("ok"):
            raise ServiceUnavailableError(
                f"Ollama model {self.model!r} is not available. "
                f"Start Ollama and run: ollama pull {self.model}"
            )

    @trace
    def chat(self, user_prompt: str) -> str:
        return self.chat_messages(
            [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )

    @trace
    def chat_messages(self, messages: Sequence[BaseMessage]) -> str:
        self.assert_model_available()
        with timed(
            "ollama.chat",
            model=self.model,
            messages=len(messages),
            max_tokens=self.max_tokens,
        ):
            response = self.client.invoke(list(messages))
        return str(response.content or "")

    @trace
    def structured_output(
        self,
        messages: Sequence[BaseMessage],
        schema: type[StructuredOutput],
    ) -> StructuredOutput:
        self.assert_model_available()
        with timed("ollama.structured_output", model=self.model, schema=schema.__name__):
            structured_client = self.client.with_structured_output(schema)
            result = structured_client.invoke(list(messages))
        if isinstance(result, schema):
            return result
        return schema.model_validate(result)
