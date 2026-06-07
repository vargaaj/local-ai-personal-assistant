from __future__ import annotations

import asyncio
import json
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
        ui.page_title("Local AI Personal Assistant")
        ui.colors(primary="#0f766e", secondary="#64748b", accent="#0f766e")
        _add_styles(ui)

        with ui.header().classes(
            "items-center justify-between bg-white text-slate-900 border-b "
            "border-slate-200 px-5"
        ):
            with ui.row().classes("items-center gap-3"):
                ui.label("AI").classes(
                    "bg-primary text-white font-bold rounded-lg px-3 py-2"
                )
                with ui.column().classes("gap-0"):
                    ui.label("Local AI Personal Assistant").classes("text-lg font-semibold")
                    ui.label("LangGraph, Qdrant, Ollama, FastEmbed, MLflow").classes(
                        "text-xs text-slate-500"
                    )
            status_chip = ui.chip("Checking", icon="radio_button_unchecked").props(
                "outline"
            )

        with ui.row().classes("rag-page w-full no-wrap"):
            with ui.column().classes("thread-panel"):
                with ui.card().classes("tool-card"):
                    with ui.row().classes("items-center justify-between w-full"):
                        ui.label("Threads").classes("card-title")
                        new_thread_button = ui.button(icon="add").props("flat dense")
                    thread_list = ui.column().classes("gap-1 w-full")
                    delete_thread_button = ui.button(
                        "Delete selected",
                        icon="delete_outline",
                    ).props("outline color=negative").classes("w-full")

            with ui.column().classes("chat-panel"):
                messages = ui.column().classes("message-list")
                with messages:
                    ui.label("Ask a question to start this thread.").classes("empty-state")

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
                    mode = (
                        ui.select(
                            ["auto", "general", "documents", "web", "research"],
                            value="auto",
                            label="Mode",
                        )
                        .props("outlined dense")
                        .classes("mode-input")
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

                with ui.card().classes("tool-card"):
                    ui.label("Saved Memory").classes("card-title")
                    memory_input = ui.input(placeholder="Add an explicit saved fact").props(
                        "outlined dense"
                    ).classes("w-full")
                    add_memory_button = ui.button("Remember", icon="bookmark_add").props(
                        "outline"
                    ).classes("w-full")
                    memory_list = ui.column().classes("gap-2 w-full")

                with ui.card().classes("tool-card"):
                    with ui.row().classes("items-center justify-between w-full"):
                        ui.label("Agent Runs").classes("card-title")
                        refresh_runs_button = ui.button(icon="refresh").props("flat dense")
                    run_list = ui.column().classes("gap-2 w-full")

        busy = {"chat": False, "ingest": False}
        ingest_lock = asyncio.Lock()
        current_thread = {"id": None}

        async def select_thread(thread_id: str) -> None:
            current_thread["id"] = thread_id
            try:
                detail = await run.io_bound(lambda: get_service().get_thread(thread_id))
            except Exception as exc:
                ui.notify(_format_error(exc), type="negative")
                return
            _replace(messages, lambda: _thread_messages(ui, detail.messages if detail else []))
            await refresh_threads()
            await refresh_runs()

        async def refresh_threads() -> None:
            try:
                threads = await run.io_bound(lambda: get_service().list_threads())
            except Exception as exc:
                _replace(thread_list, lambda: _health_error(ui, exc))
                return
            _replace(
                thread_list,
                lambda: _thread_rows(
                    ui,
                    threads,
                    current_thread["id"],
                    select_thread,
                ),
            )

        async def start_new_thread() -> None:
            current_thread["id"] = None
            question.value = ""
            _replace(messages, lambda: _thread_messages(ui, []))
            _replace(
                source_list,
                lambda: ui.label(
                    "Sources from the latest answer will appear here."
                ).classes("text-sm text-slate-500"),
            )
            await refresh_threads()
            await refresh_runs()

        async def delete_thread() -> None:
            thread_id = current_thread["id"]
            if not thread_id:
                return
            try:
                await run.io_bound(lambda: get_service().delete_thread(thread_id))
            except Exception as exc:
                ui.notify(_format_error(exc), type="negative")
                return
            current_thread["id"] = None
            try:
                threads = await run.io_bound(lambda: get_service().list_threads())
            except Exception:
                await start_new_thread()
                return
            if threads:
                await select_thread(threads[0].id)
            else:
                await start_new_thread()

        async def refresh_memory() -> None:
            try:
                facts = await run.io_bound(lambda: get_service().list_memory_facts())
            except Exception as exc:
                _replace(memory_list, lambda: _health_error(ui, exc))
                return
            _replace(memory_list, lambda: _memory_rows(ui, facts, delete_memory))

        async def add_memory() -> None:
            content = str(memory_input.value or "").strip()
            if not content:
                ui.notify("Enter a fact to remember.", type="warning")
                return
            try:
                await run.io_bound(lambda: get_service().add_memory_fact(content))
            except Exception as exc:
                ui.notify(_format_error(exc), type="negative")
                return
            memory_input.value = ""
            await refresh_memory()

        async def delete_memory(fact_id: str) -> None:
            try:
                await run.io_bound(lambda: get_service().delete_memory_fact(fact_id))
            except Exception as exc:
                ui.notify(_format_error(exc), type="negative")
                return
            await refresh_memory()

        async def refresh_runs() -> None:
            if not current_thread["id"]:
                _replace(
                    run_list,
                    lambda: ui.label("No saved thread selected.").classes(
                        "text-sm text-slate-500"
                    ),
                )
                return
            try:
                runs = await run.io_bound(
                    lambda: get_service().list_agent_runs(thread_id=current_thread["id"])
                )
            except Exception as exc:
                _replace(run_list, lambda: _health_error(ui, exc))
                return
            _replace(run_list, lambda: _run_rows(ui, runs, cancel_run, delete_run, decide_run))
            if any(item.state == "completed" for item in runs):
                thread_id = current_thread["id"]
                if thread_id:
                    detail = await run.io_bound(lambda: get_service().get_thread(thread_id))
                    _replace(messages, lambda: _thread_messages(ui, detail.messages if detail else []))

        async def cancel_run(run_id: str) -> None:
            try:
                await run.io_bound(lambda: get_service().cancel_agent_run(run_id))
            except Exception as exc:
                ui.notify(_format_error(exc), type="negative")
            await refresh_runs()

        async def delete_run(run_id: str) -> None:
            try:
                await run.io_bound(lambda: get_service().delete_agent_run(run_id))
            except Exception as exc:
                ui.notify(_format_error(exc), type="negative")
            await refresh_runs()

        async def decide_run(
            run_id: str,
            decision: str,
            edited_payload: dict[str, Any] | None = None,
        ) -> None:
            try:
                await run.io_bound(
                    lambda: get_service().decide_agent_approval(
                        run_id,
                        decision=decision,
                        edited_payload=edited_payload,
                    )
                )
            except Exception as exc:
                ui.notify(_format_error(exc), type="negative")
            await refresh_runs()

        async def initialize_page() -> None:
            await refresh_health()
            await refresh_memory()
            try:
                threads = await run.io_bound(lambda: get_service().list_threads())
            except Exception:
                return
            if threads:
                await select_thread(threads[0].id)
            else:
                await start_new_thread()

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
            memory_command = text.lower().startswith(("/remember", "/forget"))

            try:
                limit = int(top_k.value or 4)
            except (TypeError, ValueError):
                limit = 4
            limit = max(1, min(20, limit))
            top_k.value = limit
            question.value = ""

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
                        thread_id=current_thread["id"],
                        mode=str(mode.value or "auto"),
                    )
                )
            except Exception as exc:
                pending.delete()
                _chat_message(ui, messages, _format_error(exc), name="Error")
            else:
                pending.delete()
                _chat_message(ui, messages, response.answer, name="Assistant")
                _replace(source_list, lambda: _source_rows(ui, response.sources))
                current_thread["id"] = response.thread_id
                for warning in response.warnings:
                    ui.notify(warning, type="warning")
            finally:
                busy["chat"] = False
                send_button.enable()
                if memory_command:
                    await refresh_memory()
                await refresh_threads()
                await refresh_runs()

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
        new_thread_button.on_click(start_new_thread)
        delete_thread_button.on_click(delete_thread)
        add_memory_button.on_click(add_memory)
        refresh_runs_button.on_click(refresh_runs)
        ui.timer(0.1, initialize_page, once=True)
        ui.timer(3.0, refresh_runs)

    ui.run_with(fastapi_app, title="Local AI Personal Assistant")


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
        .thread-panel {
          width: 250px;
          min-width: 220px;
          height: 100%;
          overflow-y: auto;
        }
        .thread-separator {
          margin: 4px 0;
          background-color: #e2e8f0;
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
        .mode-input {
          width: 130px;
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
          .thread-panel {
            width: 100%;
            min-width: 0;
            height: auto;
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
            label = source.title or source.filename or source.source
            ui.label(label).classes("text-sm font-semibold text-slate-800")
            if source.kind == "web" and source.url:
                ui.link(source.url, source.url, new_tab=True).classes(
                    "text-xs text-primary break-all"
                )
            elif source.filename:
                ui.label(f"Chunk {source.chunk_index}").classes("text-xs text-slate-500")
            if source.score is not None:
                ui.label(f"Score {source.score:.3f}").classes("text-xs text-slate-500")
            if source.snippet:
                ui.label(source.snippet).classes("text-xs text-slate-600")


def _thread_rows(
    ui: Any,
    threads: list[Any],
    selected_id: str | None,
    select_thread: Callable[[str], Any],
) -> None:
    if not threads:
        ui.label("No saved threads.").classes("text-sm text-slate-500")
        return
    for index, thread in enumerate(threads):
        if index:
            ui.separator().classes("thread-separator")
        props = "flat align=left"
        if thread.id == selected_id:
            props += " color=primary"
        ui.button(
            thread.title,
            on_click=partial(select_thread, thread.id),
        ).props(props).classes("w-full")


def _thread_messages(ui: Any, messages: list[Any]) -> None:
    if not messages:
        ui.label("Ask a question to start this thread.").classes("empty-state")
        return
    for message in messages:
        if message.role not in {"user", "assistant"}:
            continue
        ui.chat_message(
            message.content,
            name="You" if message.role == "user" else "Assistant",
            sent=message.role == "user",
        ).classes("w-full")


def _memory_rows(ui: Any, facts: list[Any], delete_memory: Callable[[str], Any]) -> None:
    if not facts:
        ui.label("No explicit saved facts.").classes("text-sm text-slate-500")
        return
    for fact in facts:
        with ui.row().classes("source-row items-start justify-between w-full no-wrap"):
            with ui.column().classes("gap-1 min-w-0"):
                ui.label(fact.content).classes("text-xs text-slate-700")
                ui.label(f"ID: {fact.id}").classes("text-[11px] text-slate-500 break-all")
            ui.button(icon="delete_outline", on_click=partial(delete_memory, fact.id)).props(
                "flat dense color=negative"
            )


def _run_rows(
    ui: Any,
    runs: list[Any],
    cancel_run: Callable[[str], Any],
    delete_run: Callable[[str], Any],
    decide_run: Callable[[str, str], Any],
) -> None:
    if not runs:
        ui.label("No agent runs for this thread.").classes("text-sm text-slate-500")
        return
    for run in runs:
        with ui.column().classes("source-row gap-1"):
            ui.label(f"{run.agent_name}: {run.state}").classes(
                "text-sm font-semibold text-slate-800"
            )
            ui.label(run.task).classes("text-xs text-slate-600")
            if run.error:
                ui.label(run.error).classes("text-xs text-red-700")
            if run.sources:
                names = [
                    source.filename or source.title or source.url or source.source
                    for source in run.sources
                ]
                ui.label(f"Sources: {', '.join(name for name in names if name)}").classes(
                    "text-xs text-slate-500"
                )
            if run.approval and run.approval.state == "pending":
                ui.label(str(run.approval.payload)).classes("text-xs text-amber-700")
                edited_payload = ui.textarea(
                    value=json.dumps(run.approval.payload, indent=2),
                    label="Edit approval payload",
                ).props("outlined dense").classes("w-full")
                with ui.row().classes("gap-1"):
                    ui.button("Approve", on_click=partial(decide_run, run.id, "approve")).props(
                        "outline dense color=positive"
                    )
                    ui.button(
                        "Edit and approve",
                        on_click=partial(_submit_edit, ui, decide_run, run.id, edited_payload),
                    ).props("outline dense")
                    ui.button("Reject", on_click=partial(decide_run, run.id, "reject")).props(
                        "outline dense color=negative"
                    )
            with ui.row().classes("gap-1"):
                if run.state in {"queued", "running", "awaiting_approval"}:
                    ui.button("Cancel", on_click=partial(cancel_run, run.id)).props(
                        "flat dense color=negative"
                    )
                else:
                    ui.button("Delete", on_click=partial(delete_run, run.id)).props(
                        "flat dense color=negative"
                    )


async def _submit_edit(
    ui: Any,
    decide_run: Callable[[str, str, dict[str, Any] | None], Any],
    run_id: str,
    input_element: Any,
) -> None:
    try:
        payload = json.loads(str(input_element.value or "{}"))
    except json.JSONDecodeError as exc:
        ui.notify(f"Approval payload must be valid JSON: {exc}", type="warning")
        return
    if not isinstance(payload, dict):
        ui.notify("Approval payload must be a JSON object.", type="warning")
        return
    await decide_run(run_id, "edit", payload)


def _format_error(exc: Exception) -> str:
    if isinstance(exc, ServiceUnavailableError):
        return str(exc)
    detail = getattr(exc, "detail", None)
    if detail:
        return str(detail)
    return str(exc)
