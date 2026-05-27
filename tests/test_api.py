import httpx
import pytest

from rag_app.api import create_app
from rag_app.schemas import ChatResponse, IngestResponse


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

    def chat(self, *, question: str, top_k: int, session_id: str | None = None):
        return ChatResponse(answer=f"answer: {question} ({top_k})", sources=[])


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
