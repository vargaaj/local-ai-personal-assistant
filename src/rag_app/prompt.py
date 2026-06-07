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

GENERAL_SYSTEM_PROMPT_TEMPLATE = (
    "You are a helpful personal assistant for {{user_name}}. Answer normal "
    "questions directly from your available knowledge and the conversation. "
    "Use saved facts only when relevant. If the user asks for current external "
    "information and no web results are provided, say that live web lookup is "
    "needed. Do not claim that local documents were consulted unless document "
    "context is present."
)

WEB_SYSTEM_PROMPT_TEMPLATE = (
    "You answer using the provided live web search snippets. Cite relevant URLs "
    "inline. State uncertainty when the snippets are incomplete or conflicting. "
    "Do not invent details that are absent from the search results."
)

ROUTER_SYSTEM_PROMPT_TEMPLATE = (
    "Classify the user's request into exactly one mode: general, documents, "
    "web, or research. Use documents for questions likely answered by the "
    "user's indexed local files. Use research for complex local-document "
    "questions requiring several searches or a broad synthesis. Use web for "
    "explicitly current, latest, external, or internet-related questions. Use "
    "general for ordinary conversation and stable general knowledge."
)

RESEARCH_PLANNER_PROMPT_TEMPLATE = (
    "Create concise local-document search queries for the user's research task. "
    "Return queries that cover distinct aspects of the task without using web "
    "search."
)

RESEARCH_SYNTHESIS_PROMPT_TEMPLATE = (
    "Synthesize a sourced answer to the research task from the retrieved local "
    "document context. Cite relevant filenames. Separate direct evidence from "
    "inferences and state when the documents are insufficient."
)

PROMPT_TEMPLATES = {
    "documents": DEFAULT_SYSTEM_PROMPT_TEMPLATE,
    "general": GENERAL_SYSTEM_PROMPT_TEMPLATE,
    "web": WEB_SYSTEM_PROMPT_TEMPLATE,
    "router": ROUTER_SYSTEM_PROMPT_TEMPLATE,
    "research-planner": RESEARCH_PLANNER_PROMPT_TEMPLATE,
    "research-synthesis": RESEARCH_SYNTHESIS_PROMPT_TEMPLATE,
}


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


@dataclass(frozen=True)
class PromptBundle:
    prompts: dict[str, str]
    metadata: dict[str, str]

    def get(self, role: str) -> str:
        return self.prompts[role]


def resolve_prompt_bundle(
    *,
    user_name: str,
    prompt_name: str,
    prompt_alias: str,
    registry_enabled: bool,
) -> PromptBundle:
    prompts = {}
    metadata: dict[str, str] = {}
    for role, template in PROMPT_TEMPLATES.items():
        resolved = resolve_role_prompt(
            user_name=user_name,
            prompt_name=f"{prompt_name}-{role}",
            prompt_alias=prompt_alias,
            registry_enabled=registry_enabled,
            template=template,
            role=role,
        )
        prompts[role] = resolved.content
        metadata.update(
            {
                f"rag.prompt.{role}.{key.removeprefix('rag.prompt.')}": value
                for key, value in resolved.trace_metadata().items()
            }
        )
    return PromptBundle(prompts=prompts, metadata=metadata)


def resolve_role_prompt(
    *,
    user_name: str,
    prompt_name: str,
    prompt_alias: str,
    registry_enabled: bool,
    template: str,
    role: str,
) -> SystemPrompt:
    if not registry_enabled:
        return SystemPrompt(
            content=build_system_prompt(user_name=user_name, template=template),
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
            template=template,
            role=role,
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
        logger.warning("Falling back to local %s prompt: %s", role, exc)
        return SystemPrompt(
            content=build_system_prompt(user_name=user_name, template=template),
            name=prompt_name,
            alias=prompt_alias,
            source="fallback",
            error=str(exc),
        )


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
            template=DEFAULT_SYSTEM_PROMPT_TEMPLATE,
            role="documents",
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
    template: str,
    role: str,
) -> Any:
    prompt_uri = f"prompts:/{prompt_name}@{prompt_alias}"
    prompt = mlflow.genai.load_prompt(
        prompt_uri,
        allow_missing=True,
        cache_ttl_seconds=0,
    )
    if prompt is not None:
        return prompt

    prompt = mlflow.genai.register_prompt(
        name=prompt_name,
        template=template,
        commit_message=f"Initial local assistant {role} prompt.",
        tags={
            "app": "local-ai-assistant",
            "prompt_role": role,
        },
    )
    mlflow.genai.set_prompt_alias(prompt_name, prompt_alias, int(prompt.version))
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


def build_web_user_prompt(question: str, results: list[dict[str, str]]) -> str:
    blocks = []
    for index, result in enumerate(results, start=1):
        blocks.append(
            f"[{index}] {result.get('title', 'Untitled')}\n"
            f"URL: {result.get('url', '')}\n"
            f"{result.get('content', '')}"
        )
    context = "\n\n".join(blocks).strip() or "No web search results."
    return f"Web search results:\n{context}\n\nQuestion:\n{question}"
