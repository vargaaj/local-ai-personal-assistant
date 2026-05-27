from __future__ import annotations

from typing import Any

from openai import OpenAI

from .errors import ServiceUnavailableError
from .tracing import timed, trace


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
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.system_prompt_metadata = system_prompt_metadata or {}
        self.client = OpenAI(base_url=base_url, api_key="ollama")

    @trace
    def health(self) -> dict[str, Any]:
        try:
            with timed("ollama.health", model=self.model):
                models = self.client.models.list()
            available = sorted(model.id for model in models.data)
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
        self.assert_model_available()
        with timed(
            "ollama.chat_completion",
            model=self.model,
            prompt_chars=len(user_prompt),
            max_tokens=self.max_tokens,
        ):
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        return response.choices[0].message.content or ""
