from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Response
from pydantic import ValidationError

from .config import Settings, get_settings
from .errors import ServiceUnavailableError
from .schemas import ChatRequest, ChatResponse, HealthResponse, IngestRequest, IngestResponse
from .service import RagService, build_service
from .tracing import configure_mlflow
from .ui import register_nicegui_ui


def create_app(
    *,
    settings: Settings | None = None,
    service: RagService | Any | None = None,
    enable_ui: bool = True,
) -> FastAPI:
    app = FastAPI(title="Local RAG Service", version="0.1.0")

    configuration_error: str | None = None
    mlflow_status: dict[str, Any] = {}
    rag_service = service

    if rag_service is None:
        try:
            resolved_settings = settings or get_settings()
            mlflow_status = configure_mlflow(resolved_settings)
            rag_service = build_service(resolved_settings, mlflow_status)
        except (ValidationError, ValueError) as exc:
            configuration_error = str(exc)

    def require_service() -> RagService | Any:
        if configuration_error or rag_service is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "configuration_error",
                    "message": configuration_error or "service is not initialized",
                },
            )
        return rag_service

    @app.get("/health", response_model=HealthResponse)
    async def health(response: Response):
        if configuration_error or rag_service is None:
            response.status_code = 503
            return {
                "status": "configuration_error",
                "qdrant": {"ok": False},
                "ollama": {"ok": False},
                "mlflow": mlflow_status,
                "config": {
                    "error": configuration_error or "service is not initialized",
                },
            }

        result = rag_service.health()
        if result.get("status") != "ok":
            response.status_code = 503
        return result

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest(request: IngestRequest):
        try:
            service = require_service()
            return service.ingest(reset=request.reset)
        except ServiceUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        try:
            service = require_service()
            return service.chat(
                question=request.question,
                top_k=request.top_k,
                session_id=request.session_id,
            )
        except ServiceUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    if enable_ui:
        register_nicegui_ui(app, get_service=require_service)

    return app
