from __future__ import annotations

from typing import Any, Protocol

from .errors import ServiceUnavailableError
from .schemas import SourceInfo
from .tracing import timed, trace


class WebSearchProvider(Protocol):
    @property
    def configured(self) -> bool: ...

    def search(self, query: str) -> list[dict[str, str]]: ...


class OllamaWebSearchProvider:
    def __init__(self, *, api_key: str | None, max_results: int = 5) -> None:
        self.api_key = api_key
        self.max_results = max_results

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    @trace
    def search(self, query: str) -> list[dict[str, str]]:
        if not self.configured:
            raise ValueError("OLLAMA_API_KEY is not configured.")

        from ollama import Client

        client = Client(
            host="https://ollama.com",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with timed("ollama.web_search", query_chars=len(query), max_results=self.max_results):
                response = client.web_search(query=query, max_results=self.max_results)
        except Exception as exc:
            raise ServiceUnavailableError("Ollama web search is unavailable.") from exc
        raw_results = getattr(response, "results", None)
        if raw_results is None and isinstance(response, dict):
            raw_results = response.get("results", [])
        return [_result_dict(result) for result in raw_results or []]


def source_info_from_web_result(result: dict[str, str]) -> SourceInfo:
    return SourceInfo(
        kind="web",
        source=result.get("url", ""),
        title=result.get("title") or None,
        url=result.get("url") or None,
        snippet=_snippet(result.get("content", "")),
    )


def _result_dict(result: Any) -> dict[str, str]:
    if isinstance(result, dict):
        return {
            "title": str(result.get("title", "")),
            "url": str(result.get("url", "")),
            "content": str(result.get("content", "")),
        }
    return {
        "title": str(getattr(result, "title", "")),
        "url": str(getattr(result, "url", "")),
        "content": str(getattr(result, "content", "")),
    }


def _snippet(text: str, limit: int = 400) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."
