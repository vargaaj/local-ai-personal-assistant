from types import SimpleNamespace

from rag_app.documents import DocumentChunk
from rag_app.ids import point_id_for_chunk
from rag_app.prompt import _load_or_create_prompt, build_system_prompt, build_user_prompt
from rag_app.qdrant_store import SearchResult


def test_point_id_is_deterministic():
    chunk = DocumentChunk(
        source="/docs/a.txt",
        filename="a.txt",
        chunk_index=2,
        text="hello",
        content_sha256="abc123",
        source_sha256="source123",
    )

    assert point_id_for_chunk(chunk) == point_id_for_chunk(chunk)


def test_build_user_prompt_includes_context_and_question():
    results = [
        SearchResult(
            id="1",
            score=0.9,
            payload={"filename": "a.txt", "chunk_index": 0, "text": "important text"},
        )
    ]

    prompt = build_user_prompt("What matters?", results)

    assert "Source: a.txt" in prompt
    assert "important text" in prompt
    assert "What matters?" in prompt


def test_build_system_prompt_includes_user_identity():
    prompt = build_system_prompt(user_name="AJ Varga")

    assert "interpret the user as AJ Varga" in prompt
    assert "first-person questions" in prompt
    assert "label it as an inference" in prompt


def test_load_or_create_prompt_uses_mlflow_genai_namespace():
    calls = []

    class GenAI:
        def load_prompt(self, uri, **kwargs):
            calls.append(("load", uri, kwargs))
            return None

        def register_prompt(self, **kwargs):
            calls.append(("register", kwargs))
            return SimpleNamespace(version=7)

        def set_prompt_alias(self, name, alias, version):
            calls.append(("alias", name, alias, version))

    prompt = _load_or_create_prompt(
        SimpleNamespace(genai=GenAI()),
        prompt_name="assistant-router",
        prompt_alias="production",
        template="route this",
        role="router",
    )

    assert prompt.version == 7
    assert calls == [
        (
            "load",
            "prompts:/assistant-router@production",
            {"allow_missing": True, "cache_ttl_seconds": 0},
        ),
        (
            "register",
            {
                "name": "assistant-router",
                "template": "route this",
                "commit_message": "Initial local assistant router prompt.",
                "tags": {"app": "local-ai-assistant", "prompt_role": "router"},
            },
        ),
        ("alias", "assistant-router", "production", 7),
    ]
