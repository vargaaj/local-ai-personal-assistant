from __future__ import annotations

from contextlib import asynccontextmanager
from functools import partial
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from .config import Settings, get_settings
from .errors import ServiceUnavailableError
from .schemas import (
    AgentLaunchRequest,
    AgentRunInfo,
    ApprovalDecisionRequest,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    MemoryFact,
    MemoryFactCreate,
    ThreadCreateRequest,
    ThreadDetail,
    ThreadSummary,
)
from .service import RagService, build_service
from .tracing import configure_mlflow
from .ui import register_nicegui_ui


def create_app(
    *,
    settings: Settings | None = None,
    service: RagService | Any | None = None,
    enable_ui: bool = True,
) -> FastAPI:
    rag_service = service

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            close = getattr(rag_service, "close", None)
            if callable(close):
                await run_in_threadpool(close)

    app = FastAPI(title="Local AI Personal Assistant", version="0.1.0", lifespan=lifespan)

    configuration_error: str | None = None
    mlflow_status: dict[str, Any] = {}

    @app.exception_handler(ServiceUnavailableError)
    async def service_unavailable_handler(_request, exc: ServiceUnavailableError):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

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

        result = await run_in_threadpool(rag_service.health)
        if result.get("status") != "ok":
            response.status_code = 503
        return result

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest(request: IngestRequest):
        try:
            service = require_service()
            return await run_in_threadpool(partial(service.ingest, reset=request.reset))
        except ServiceUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        try:
            service = require_service()
            return await run_in_threadpool(
                partial(
                    service.chat,
                    question=request.question,
                    top_k=request.top_k,
                    session_id=request.session_id,
                    thread_id=request.thread_id,
                    mode=request.mode,
                )
            )
        except ServiceUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/threads", response_model=ThreadSummary)
    async def create_thread(request: ThreadCreateRequest):
        service = require_service()
        return await run_in_threadpool(partial(service.create_thread, title=request.title))

    @app.get("/threads", response_model=list[ThreadSummary])
    async def list_threads():
        service = require_service()
        return await run_in_threadpool(service.list_threads)

    @app.get("/threads/{thread_id}", response_model=ThreadDetail)
    async def get_thread(thread_id: str):
        service = require_service()
        thread = await run_in_threadpool(partial(service.get_thread, thread_id))
        if thread is None:
            raise HTTPException(status_code=404, detail="Thread not found.")
        return thread

    @app.delete("/threads/{thread_id}", status_code=204)
    async def delete_thread(thread_id: str):
        service = require_service()
        deleted = await run_in_threadpool(partial(service.delete_thread, thread_id))
        if not deleted:
            raise HTTPException(status_code=404, detail="Thread not found.")
        return Response(status_code=204)

    @app.get("/memory", response_model=list[MemoryFact])
    async def list_memory_facts():
        service = require_service()
        return await run_in_threadpool(service.list_memory_facts)

    @app.post("/memory", response_model=MemoryFact)
    async def add_memory_fact(request: MemoryFactCreate):
        try:
            service = require_service()
            return await run_in_threadpool(partial(service.add_memory_fact, request.content))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/memory/{fact_id}", status_code=204)
    async def delete_memory_fact(fact_id: str):
        service = require_service()
        deleted = await run_in_threadpool(partial(service.delete_memory_fact, fact_id))
        if not deleted:
            raise HTTPException(status_code=404, detail="Saved memory not found.")
        return Response(status_code=204)

    @app.get("/agents/runs", response_model=list[AgentRunInfo])
    async def list_agent_runs(thread_id: str | None = Query(default=None)):
        service = require_service()
        return await run_in_threadpool(partial(service.list_agent_runs, thread_id=thread_id))

    @app.post("/agents/runs", response_model=AgentRunInfo)
    async def launch_agent(request: AgentLaunchRequest):
        try:
            service = require_service()
            return await run_in_threadpool(
                partial(
                    service.launch_agent,
                    agent_name=request.agent_name,
                    thread_id=request.thread_id,
                    task=request.task,
                    top_k=request.top_k,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/agents/runs/{run_id}", response_model=AgentRunInfo)
    async def get_agent_run(run_id: str):
        service = require_service()
        run = await run_in_threadpool(partial(service.get_agent_run, run_id))
        if run is None:
            raise HTTPException(status_code=404, detail="Agent run not found.")
        return run

    @app.post("/agents/runs/{run_id}/cancel", response_model=AgentRunInfo)
    async def cancel_agent_run(run_id: str):
        try:
            service = require_service()
            return await run_in_threadpool(partial(service.cancel_agent_run, run_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.delete("/agents/runs/{run_id}", status_code=204)
    async def delete_agent_run(run_id: str):
        try:
            service = require_service()
            deleted = await run_in_threadpool(partial(service.delete_agent_run, run_id))
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(status_code=404, detail="Agent run not found.")
        return Response(status_code=204)

    @app.post("/agents/runs/{run_id}/approval", response_model=AgentRunInfo)
    async def decide_agent_approval(run_id: str, request: ApprovalDecisionRequest):
        try:
            service = require_service()
            return await run_in_threadpool(
                partial(
                    service.decide_agent_approval,
                    run_id,
                    decision=request.decision,
                    edited_payload=request.edited_payload,
                )
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    if enable_ui:
        register_nicegui_ui(app, get_service=require_service)

    return app
