from __future__ import annotations

import asyncio
from uuid import uuid4
from collections.abc import Callable
from functools import partial
from typing import Any

from fastapi import FastAPI

from .errors import ServiceUnavailableError
from .uploads import save_uploaded_document


def register_nicegui_ui(
    fastapi_app: FastAPI,
    *,
    get_service: Callable[[], Any],
) -> None:
    try:
        from nicegui import events, run, ui
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "NiceGUI is not installed. Install project dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    @ui.page("/")
    def chat_page() -> None:
        session_id = f"nicegui-{uuid4().hex}"
        ui.page_title("Local RAG Chat")
        ui.colors(primary="#0f766e", secondary="#64748b", accent="#0f766e")
        _add_styles(ui)

        with ui.header().classes(
            "items-center justify-between bg-white text-slate-900 border-b "
            "border-slate-200 px-5"
        ):
            with ui.row().classes("items-center gap-3"):
                ui.label("RAG").classes(
                    "bg-primary text-white font-bold rounded-lg px-3 py-2"
                )
                with ui.column().classes("gap-0"):
                    ui.label("Local RAG Chat").classes("text-lg font-semibold")
                    ui.label("Qdrant, Ollama, FastEmbed, MLflow").classes(
                        "text-xs text-slate-500"
                    )
            status_chip = ui.chip("Checking", icon="radio_button_unchecked").props(
                "outline"
            )

        with ui.row().classes("rag-page w-full no-wrap"):
            with ui.column().classes("chat-panel"):
                messages = ui.column().classes("message-list")
                with messages:
                    empty_state = ui.label(
                        "Ask a question about your indexed local documents."
                    ).classes("empty-state")

                with ui.row().classes("composer-row"):
                    question = (
                        ui.textarea(placeholder="Ask a question")
                        .props("outlined autogrow")
                        .classes("question-input")
                    )
                    top_k = (
                        ui.number("Top K", value=4, min=1, max=20)
                        .props("outlined dense")
                        .classes("top-k-input")
                    )
                    send_button = ui.button("Send", icon="send").classes(
                        "send-button"
                    )
                    question.on(
                        "keydown.enter",
                        js_handler=(
                            "(event) => {"
                            " if (!event.shiftKey) {"
                            "   event.preventDefault();"
                            "   document.querySelector('.send-button')?.click();"
                            " }"
                            "}"
                        ),
                    )

            with ui.column().classes("side-panel"):
                with ui.card().classes("tool-card"):
                    ui.label("Documents").classes("card-title")
                    uploader = ui.upload(
                        label="Upload PDF, Markdown, text, or zip files",
                        multiple=True,
                        auto_upload=True,
                        max_file_size=50 * 1024 * 1024,
                        on_rejected=lambda: ui.notify(
                            "Only .pdf, .md, .txt, and .zip files up to 50 MB are accepted.",
                            type="warning",
                        ),
                    ).props("accept=.pdf,.md,.txt,.zip flat bordered").classes("w-full")
                    upload_status = ui.label("No files uploaded this session.").classes(
                        "text-sm text-slate-500"
                    )
                    ui.separator()
                    ingest_button = ui.button(
                        "Ingest new and changed files",
                        icon="sync",
                    ).props("outline").classes("w-full")
                    reset_button = ui.button(
                        "Reset and rebuild index",
                        icon="restart_alt",
                    ).props("outline color=negative").classes("w-full")

                with ui.card().classes("tool-card"):
                    ui.label("System").classes("card-title")
                    health_grid = ui.column().classes("gap-2 w-full")
                    health_button = ui.button(
                        "Refresh status",
                        icon="refresh",
                    ).props("outline").classes("w-full")

                with ui.card().classes("tool-card"):
                    ui.label("Sources").classes("card-title")
                    source_list = ui.column().classes("gap-2 w-full")
                    with source_list:
                        ui.label("Sources from the latest answer will appear here.").classes(
                            "text-sm text-slate-500"
                        )

        busy = {"chat": False, "ingest": False}
        ingest_lock = asyncio.Lock()

        async def refresh_health() -> None:
            status_chip.set_text("Checking")
            status_chip.props("icon=radio_button_unchecked color=grey")
            try:
                health = await run.io_bound(lambda: get_service().health())
            except Exception as exc:
                status_chip.set_text("Unavailable")
                status_chip.props("icon=error color=negative")
                _replace(health_grid, lambda: _health_error(ui, exc))
                return

            ok = health.get("status") == "ok"
            status_chip.set_text("Ready" if ok else "Degraded")
            status_chip.props(
                "icon=check_circle color=positive"
                if ok
                else "icon=warning color=warning"
            )
            _replace(health_grid, lambda: _health_rows(ui, health))

        async def handle_chat() -> None:
            if busy["chat"]:
                return

            text = str(question.value or "").strip()
            if not text:
                ui.notify("Enter a question first.", type="warning")
                return

            try:
                limit = int(top_k.value or 4)
            except (TypeError, ValueError):
                limit = 4
            limit = max(1, min(20, limit))
            top_k.value = limit
            question.value = ""

            empty_state.visible = False
            _chat_message(ui, messages, text, name="You", sent=True)
            pending = _chat_message(ui, messages, "Thinking...", name="Assistant")
            _replace(source_list, lambda: ui.label("Waiting for answer...").classes(
                "text-sm text-slate-500"
            ))

            busy["chat"] = True
            send_button.disable()
            try:
                response = await run.io_bound(
                    lambda: get_service().chat(
                        question=text,
                        top_k=limit,
                        session_id=session_id,
                    )
                )
            except Exception as exc:
                pending.delete()
                _chat_message(ui, messages, _format_error(exc), name="Error")
            else:
                pending.delete()
                _chat_message(ui, messages, response.answer, name="Assistant")
                _replace(source_list, lambda: _source_rows(ui, response.sources))
            finally:
                busy["chat"] = False
                send_button.enable()

        async def run_ingest(*, reset: bool) -> None:
            if ingest_lock.locked():
                upload_status.set_text("Ingestion is running; queued another pass.")

            async with ingest_lock:
                busy["ingest"] = True
                ingest_button.disable()
                reset_button.disable()
                upload_status.set_text(
                    "Rebuilding index..." if reset else "Ingesting documents..."
                )
                try:
                    result = await run.io_bound(
                        lambda: get_service().ingest(reset=reset)
                    )
                except Exception as exc:
                    upload_status.set_text("Ingestion failed.")
                    ui.notify(_format_error(exc), type="negative")
                else:
                    upload_status.set_text(
                        f"Indexed {result.chunks_indexed} chunks from "
                        f"{result.files_processed} files. "
                        f"Skipped {len(result.skipped_files)} unchanged files."
                    )
                    if result.parser_errors:
                        ui.notify(
                            f"{len(result.parser_errors)} file(s) could not be parsed.",
                            type="warning",
                        )
                    else:
                        ui.notify("Ingestion complete.", type="positive")
                finally:
                    busy["ingest"] = False
                    ingest_button.enable()
                    reset_button.enable()
                    await refresh_health()

        async def handle_upload(e: events.UploadEventArguments) -> None:
            try:
                service = get_service()
                upload_result = await save_uploaded_document(
                    e.file,
                    docs_root=service.settings.rag_docs_root,
                )
            except Exception as exc:
                ui.notify(_format_error(exc), type="negative")
                return

            upload_status.set_text(
                f"{upload_result.status_text(service.settings.rag_docs_root)} "
                "Indexing uploaded document(s)..."
            )
            await run_ingest(reset=False)

        send_button.on_click(handle_chat)
        ingest_button.on_click(partial(run_ingest, reset=False))
        reset_button.on_click(partial(run_ingest, reset=True))
        health_button.on_click(refresh_health)
        uploader.on_upload(handle_upload)
        ui.timer(0.1, refresh_health, once=True)

    ui.run_with(fastapi_app, title="Local RAG Chat")


def _add_styles(ui: Any) -> None:
    ui.add_css(
        """
        body {
          background: #f6f7f9;
        }
        .rag-page {
          height: calc(100vh - 64px);
          padding: 18px clamp(14px, 3vw, 32px);
          gap: 18px;
        }
        .chat-panel {
          flex: 1 1 auto;
          min-width: 0;
          height: 100%;
          border: 1px solid #d6dee7;
          border-radius: 8px;
          background: white;
          overflow: hidden;
        }
        .message-list {
          flex: 1 1 auto;
          min-height: 0;
          width: 100%;
          overflow-y: auto;
          padding: 18px;
          gap: 12px;
        }
        .empty-state {
          margin: auto;
          color: #64748b;
          text-align: center;
        }
        .composer-row {
          width: 100%;
          align-items: end;
          gap: 10px;
          padding: 12px;
          border-top: 1px solid #d6dee7;
          background: #ffffff;
        }
        .question-input {
          flex: 1 1 auto;
          min-width: 220px;
        }
        .top-k-input {
          width: 96px;
        }
        .send-button {
          height: 48px;
        }
        .side-panel {
          width: 340px;
          min-width: 300px;
          height: 100%;
          overflow-y: auto;
          gap: 14px;
        }
        .tool-card {
          width: 100%;
          border-radius: 8px;
          box-shadow: none;
          border: 1px solid #d6dee7;
          gap: 10px;
        }
        .card-title {
          font-weight: 700;
          color: #1e293b;
        }
        .source-row {
          padding: 10px;
          border: 1px solid #e2e8f0;
          border-radius: 8px;
          background: #f8fafc;
        }
        @media (max-width: 860px) {
          .rag-page {
            height: auto;
            min-height: calc(100vh - 64px);
            flex-wrap: wrap;
          }
          .chat-panel {
            height: 70vh;
            flex-basis: 100%;
          }
          .side-panel {
            width: 100%;
            min-width: 0;
            height: auto;
          }
        }
        """
    )


def _chat_message(
    ui: Any,
    parent: Any,
    text: str,
    *,
    name: str,
    sent: bool = False,
) -> Any:
    with parent:
        return ui.chat_message(text, name=name, sent=sent).classes("w-full")


def _replace(container: Any, render: Callable[[], None]) -> None:
    container.clear()
    with container:
        render()


def _health_rows(ui: Any, health: dict[str, Any]) -> None:
    _status_row(ui, "Qdrant", health.get("qdrant", {}).get("ok"))
    _status_row(ui, "Ollama", health.get("ollama", {}).get("ok"))
    _status_row(ui, "MLflow", health.get("mlflow", {}).get("configured"))
    config = health.get("config", {})
    if config:
        ui.label(str(config.get("qdrant_collection", ""))).classes(
            "text-xs text-slate-500"
        )
        ui.label(str(config.get("rag_docs_root", ""))).classes(
            "text-xs text-slate-500 break-all"
        )


def _health_error(ui: Any, exc: Exception) -> None:
    ui.label(_format_error(exc)).classes("text-sm text-red-700")


def _status_row(ui: Any, label: str, ok: Any) -> None:
    icon = "check_circle" if ok else "error"
    color = "text-green-700" if ok else "text-red-700"
    with ui.row().classes("items-center justify-between w-full"):
        ui.label(label).classes("text-sm text-slate-600")
        with ui.row().classes(f"items-center gap-1 {color}"):
            ui.icon(icon).classes("text-base")
            ui.label("OK" if ok else "Unavailable").classes("text-sm font-medium")


def _source_rows(ui: Any, sources: list[Any]) -> None:
    if not sources:
        ui.label("No sources returned.").classes("text-sm text-slate-500")
        return
    for source in sources:
        with ui.column().classes("source-row gap-1"):
            ui.label(source.filename).classes("text-sm font-semibold text-slate-800")
            ui.label(f"Chunk {source.chunk_index}").classes("text-xs text-slate-500")
            if source.score is not None:
                ui.label(f"Score {source.score:.3f}").classes("text-xs text-slate-500")
            if source.snippet:
                ui.label(source.snippet).classes("text-xs text-slate-600")


def _format_error(exc: Exception) -> str:
    if isinstance(exc, ServiceUnavailableError):
        return str(exc)
    detail = getattr(exc, "detail", None)
    if detail:
        return str(detail)
    return str(exc)
