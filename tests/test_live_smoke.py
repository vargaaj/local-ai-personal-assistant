import os

import httpx
import pytest


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1",
    reason="live Qdrant/Ollama/MLflow smoke test is opt-in",
)
@pytest.mark.anyio
async def test_live_stack_health():
    from rag_app.main import app

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/health")

    assert response.status_code in {200, 503}
    assert "qdrant" in response.json()
    assert "ollama" in response.json()
    assert "mlflow" in response.json()
