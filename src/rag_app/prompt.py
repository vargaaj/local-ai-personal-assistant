from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .qdrant_store import SearchResult

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT_TEMPLATE = (
    "You answer questions using the retrieved local document context as your "
    "primary evidence. When the context directly answers the question, answer "
    "directly and cite relevant source filenames. When the context does not "
    "explicitly answer but supports a reasonable inference, provide the "
    "inference and label it as an inference. If the context does not contain "
    "enough information for either a direct answer or a reasonable inference, "
    "say that the indexed documents do not contain enough information. Do not "
    "invent specific dates, names, amounts, or events that are not supported by "
    "the context. When the user asks first-person questions using words like I, "
    "me, my, or myself, interpret the user as {{user_name}}."
)


@dataclass(frozen=True)
class SystemPrompt:
    content: str
    name: str | None = None
    alias: str | None = None
    version: str | None = None
    uri: str | None = None
    source: str = "fallback"
    error: str | None = None

    def trace_metadata(self) -> dict[str, str]:
        metadata = {
            "rag.prompt.source": self.source,
        }
        if self.name:
            metadata["rag.prompt.name"] = self.name
        if self.alias:
            metadata["rag.prompt.alias"] = self.alias
        if self.version:
            metadata["rag.prompt.version"] = str(self.version)
        if self.uri:
            metadata["rag.prompt.uri"] = self.uri
        if self.error:
            metadata["rag.prompt.error"] = self.error
        return metadata


def build_system_prompt(
    *,
    user_name: str | None = None,
    template: str = DEFAULT_SYSTEM_PROMPT_TEMPLATE,
) -> str:
    name = (user_name or "").strip()
    if not name:
        name = "the current user"
    return template.replace("{{user_name}}", name)


def resolve_system_prompt(
    *,
    user_name: str,
    prompt_name: str,
    prompt_alias: str,
    registry_enabled: bool,
) -> SystemPrompt:
    if not registry_enabled:
        return SystemPrompt(
            content=build_system_prompt(user_name=user_name),
            name=prompt_name,
            alias=prompt_alias,
            source="fallback",
        )

    try:
        import mlflow

        prompt = _load_or_create_prompt(
            mlflow,
            prompt_name=prompt_name,
            prompt_alias=prompt_alias,
        )
        content = prompt.format(user_name=user_name)
        if not isinstance(content, str):
            raise ValueError("System prompt must resolve to a text prompt.")
        return SystemPrompt(
            content=content,
            name=prompt.name,
            alias=prompt_alias,
            version=str(prompt.version),
            uri=getattr(prompt, "uri", None),
            source="mlflow",
        )
    except Exception as exc:
        logger.warning("Falling back to local system prompt: %s", exc)
        return SystemPrompt(
            content=build_system_prompt(user_name=user_name),
            name=prompt_name,
            alias=prompt_alias,
            source="fallback",
            error=str(exc),
        )


def _load_or_create_prompt(
    mlflow: Any,
    *,
    prompt_name: str,
    prompt_alias: str,
) -> Any:
    prompt_uri = f"prompts:/{prompt_name}@{prompt_alias}"
    prompt = mlflow.load_prompt(
        prompt_uri,
        allow_missing=True,
        cache_ttl_seconds=0,
    )
    if prompt is not None:
        return prompt

    prompt = mlflow.register_prompt(
        name=prompt_name,
        template=DEFAULT_SYSTEM_PROMPT_TEMPLATE,
        commit_message="Initial balanced local RAG system prompt.",
        tags={
            "app": "local-rag",
            "prompt_role": "system",
        },
    )
    mlflow.set_prompt_alias(prompt_name, prompt_alias, int(prompt.version))
    return prompt


def build_user_prompt(question: str, results: list[SearchResult]) -> str:
    context_blocks = []
    for index, result in enumerate(results, start=1):
        filename = result.payload.get("filename", "unknown")
        chunk_index = result.payload.get("chunk_index", "unknown")
        text = result.payload.get("text", "")
        context_blocks.append(
            f"[{index}] Source: {filename} | Chunk: {chunk_index}\n{text}"
        )

    context = "\n\n".join(context_blocks).strip()
    return (
        "Context:\n"
        f"{context if context else 'No retrieved context.'}\n\n"
        f"Question:\n{question}"
    )
