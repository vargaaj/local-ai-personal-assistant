import asyncio
from threading import Event

import httpx
import pytest

from rag_app.api import create_app
from rag_app.schemas import (
    AgentRunInfo,
    ChatResponse,
    IngestResponse,
    MemoryFact,
    ThreadDetail,
    ThreadSummary,
)


class FakeService:
    def health(self):
        return {
            "status": "ok",
            "qdrant": {"ok": True},
            "ollama": {"ok": True},
            "mlflow": {"configured": True},
            "config": {"qdrant_collection": "local_documents"},
        }

    def ingest(self, *, reset: bool):
        return IngestResponse(
            files_processed=1,
            chunks_indexed=2,
            skipped_files=[],
            parser_errors=[],
        )

    def chat(
        self,
        *,
        question: str,
        top_k: int,
        session_id: str | None = None,
        thread_id: str | None = None,
        mode: str = "auto",
    ):
        return ChatResponse(
            answer=f"answer: {question} ({top_k})",
            sources=[],
            thread_id=thread_id or "thread-1",
            resolved_mode="general",
        )

    def create_thread(self, *, title: str | None = None):
        return ThreadSummary(
            id="thread-1",
            title=title or "New chat",
            created_at="now",
            updated_at="now",
        )

    def list_threads(self):
        return [self.create_thread()]

    def get_thread(self, thread_id: str):
        return ThreadDetail(**self.create_thread().model_dump(), messages=[])

    def delete_thread(self, thread_id: str):
        return thread_id == "thread-1"

    def list_memory_facts(self):
        return [MemoryFact(id="fact-1", content="likes tea", created_at="now")]

    def add_memory_fact(self, content: str):
        return MemoryFact(id="fact-2", content=content, created_at="now")

    def delete_memory_fact(self, fact_id: str):
        return fact_id == "fact-1"

    def list_agent_runs(self, *, thread_id: str | None = None):
        return []


class BlockingChatService(FakeService):
    def __init__(self) -> None:
        self.chat_started = Event()
        self.release_chat = Event()

    def chat(self, **kwargs):
        self.chat_started.set()
        self.release_chat.wait(timeout=3)
        return super().chat(**kwargs)


class BlockingThreadService(FakeService):
    def __init__(self) -> None:
        self.thread_started = Event()
        self.release_thread = Event()

    def get_thread(self, thread_id: str):
        self.thread_started.set()
        self.release_thread.wait(timeout=3)
        return super().get_thread(thread_id)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_health_endpoint():
    app = create_app(service=FakeService(), enable_ui=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"


@pytest.mark.anyio
async def test_ingest_endpoint():
    app = create_app(service=FakeService(), enable_ui=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post("/ingest", json={"reset": False})

        assert response.status_code == 200
        assert response.json()["chunks_indexed"] == 2


@pytest.mark.anyio
async def test_chat_endpoint():
    app = create_app(service=FakeService(), enable_ui=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post("/chat", json={"question": "hello", "top_k": 3})

        assert response.status_code == 200
        assert response.json()["answer"] == "answer: hello (3)"
        assert response.json()["resolved_mode"] == "general"


@pytest.mark.anyio
async def test_health_remains_responsive_while_chat_is_blocked():
    service = BlockingChatService()
    app = create_app(service=service, enable_ui=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        chat = asyncio.create_task(client.post("/chat", json={"question": "hello"}))
        assert await asyncio.to_thread(service.chat_started.wait, 1)
        try:
            health = await asyncio.wait_for(client.get("/health"), timeout=1)
        finally:
            service.release_chat.set()
        await chat

    assert health.status_code == 200


@pytest.mark.anyio
async def test_health_remains_responsive_while_thread_load_is_blocked():
    service = BlockingThreadService()
    app = create_app(service=service, enable_ui=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        thread = asyncio.create_task(client.get("/threads/thread-1"))
        assert await asyncio.to_thread(service.thread_started.wait, 1)
        try:
            health = await asyncio.wait_for(client.get("/health"), timeout=1)
        finally:
            service.release_thread.set()
        await thread

    assert health.status_code == 200


@pytest.mark.anyio
async def test_thread_and_memory_endpoints():
    app = create_app(service=FakeService(), enable_ui=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        thread = await client.post("/threads", json={"title": "Notes"})
        facts = await client.get("/memory")
        added = await client.post("/memory", json={"content": "prefers local models"})
        deleted = await client.delete("/memory/fact-1")

    assert thread.status_code == 200
    assert thread.json()["title"] == "Notes"
    assert facts.json()[0]["content"] == "likes tea"
    assert added.json()["id"] == "fact-2"
    assert deleted.status_code == 204
