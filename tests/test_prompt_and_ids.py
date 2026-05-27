from rag_app.documents import DocumentChunk
from rag_app.ids import point_id_for_chunk
from rag_app.prompt import build_system_prompt, build_user_prompt
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
