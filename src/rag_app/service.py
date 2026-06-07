from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from threading import RLock
from typing import Any

from .agents import AgentManager
from .assistant import ConversationGraph, relevant_results
from .config import Settings
from .documents import (
    DocumentChunk,
    file_sha256,
    iter_supported_files,
    parse_document,
    split_document,
)
from .embeddings import EmbeddingProvider, FastEmbedProvider
from .errors import NoIndexedDocumentsError, ServiceUnavailableError
from .manifest import IngestManifest
from .ollama_client import OllamaChatClient
from .prompt import build_user_prompt, resolve_prompt_bundle
from .qdrant_store import QdrantStore, SearchResult
from .schemas import (
    AgentRunInfo,
    ChatMode,
    ChatResponse,
    FileError,
    IngestResponse,
    MemoryFact,
    SourceInfo,
    ThreadDetail,
    ThreadSummary,
)
from .state import AppStateStore
from .tracing import timed, trace, update_current_trace
from .web_search import OllamaWebSearchProvider


class RagService:
    def __init__(
        self,
        *,
        settings: Settings,
        store: QdrantStore,
        embedder: EmbeddingProvider,
        chat_client: OllamaChatClient,
        mlflow_status: dict[str, Any],
        manifest: IngestManifest | None = None,
        state_store: AppStateStore | None = None,
        assistant: ConversationGraph | None = None,
        agent_manager: AgentManager | None = None,
        web_search: OllamaWebSearchProvider | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self.chat_client = chat_client
        self.mlflow_status = mlflow_status
        self.manifest = manifest or IngestManifest(settings.rag_manifest_path)
        self.state_store = state_store
        self.assistant = assistant
        self.agent_manager = agent_manager
        self.web_search = web_search
        self._ingest_lock = RLock()
        self._thread_lock = RLock()

    def close(self) -> None:
        if self.agent_manager is not None:
            self.agent_manager.close()
        if self.assistant is not None:
            self.assistant.close()
        if self.state_store is not None:
            self.state_store.close()

    @trace
    def health(self) -> dict[str, Any]:
        qdrant = _safe_health(self.store.health)
        ollama = self.chat_client.health()
        status = "ok" if qdrant.get("ok") and ollama.get("ok") else "degraded"
        return {
            "status": status,
            "qdrant": qdrant,
            "ollama": ollama,
            "mlflow": self.mlflow_status,
            "config": {
                "qdrant_url": self.settings.qdrant_url,
                "qdrant_collection": self.settings.qdrant_collection,
                "rag_docs_root": str(self.settings.rag_docs_root),
                "embedding_model": self.embedder.model_name,
                "embedding_vector_size": self.embedder.vector_size,
                "rag_state_db_path": str(self.settings.rag_state_db_path),
                "web_search_configured": bool(self.web_search and self.web_search.configured),
                "agents": self.agent_manager.agent_names() if self.agent_manager else [],
            },
        }

    @trace
    def ingest(self, *, reset: bool = False) -> IngestResponse:
        with self._ingest_lock:
            return self._ingest(reset=reset)

    def _ingest(self, *, reset: bool) -> IngestResponse:
        try:
            if reset:
                self.store.reset_collection()
                self.manifest.clear()
                collection_is_empty = True
            else:
                self.store.ensure_collection()
                collection_is_empty = self.store.count_points() == 0
        except Exception as exc:
            raise ServiceUnavailableError(f"Qdrant is unavailable: {exc}") from exc

        files_processed = 0
        chunks_indexed = 0
        skipped_files: list[str] = []
        parser_errors: list[FileError] = []

        files = iter_supported_files(self.settings.rag_docs_root)
        current_sources = {str(path) for path in files}
        for source in sorted(self.manifest.sources() - current_sources):
            try:
                self.store.delete_source(source)
                self.manifest.remove(source)
            except Exception as exc:
                raise ServiceUnavailableError(
                    f"Failed to remove chunks for deleted source {source}: {exc}"
                ) from exc

        for path in files:
            try:
                source_sha256 = file_sha256(path)
            except Exception as exc:
                parser_errors.append(FileError(source=str(path), error=str(exc)))
                continue

            if not reset and not collection_is_empty and self.manifest.is_current(
                source=str(path),
                source_sha256=source_sha256,
                qdrant_collection=self.settings.qdrant_collection,
                embedding_model=self.embedder.model_name,
                embedding_vector_size=self.embedder.vector_size,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            ):
                skipped_files.append(str(path))
                continue

            with timed("ingest.file", filename=path.name):
                try:
                    chunks = self._chunks_for_path(path, source_sha256=source_sha256)
                except Exception as exc:
                    parser_errors.append(FileError(source=str(path), error=str(exc)))
                    continue

                if not chunks:
                    try:
                        self.store.delete_source(str(path))
                        self.manifest.remove(str(path))
                    except Exception as exc:
                        raise ServiceUnavailableError(
                            f"Failed to remove stale chunks for {path}: {exc}"
                        ) from exc
                    skipped_files.append(str(path))
                    continue

                try:
                    self.store.delete_source(str(path))
                    vectors = self.embedder.embed_texts(
                        [chunk.text for chunk in chunks]
                    )
                    self.store.upsert_chunks(
                        chunks,
                        vectors,
                        embedding_model=self.embedder.model_name,
                        chunk_size=self.settings.chunk_size,
                        chunk_overlap=self.settings.chunk_overlap,
                    )
                    self.manifest.update(
                        source=str(path),
                        source_sha256=source_sha256,
                        qdrant_collection=self.settings.qdrant_collection,
                        embedding_model=self.embedder.model_name,
                        embedding_vector_size=self.embedder.vector_size,
                        chunk_size=self.settings.chunk_size,
                        chunk_overlap=self.settings.chunk_overlap,
                        chunk_count=len(chunks),
                    )
                except Exception as exc:
                    raise ServiceUnavailableError(
                        f"Failed to index {path} into Qdrant: {exc}"
                    ) from exc

                files_processed += 1
                chunks_indexed += len(chunks)

        return IngestResponse(
            files_processed=files_processed,
            chunks_indexed=chunks_indexed,
            skipped_files=skipped_files,
            parser_errors=parser_errors,
        )

    @trace
    def _chunks_for_path(self, path: Path, *, source_sha256: str) -> list[DocumentChunk]:
        document = parse_document(path, source_sha256=source_sha256)
        if not document.text:
            return []
        return split_document(
            document,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

    @trace
    def chat(
        self,
        *,
        question: str,
        top_k: int,
        session_id: str | None = None,
        thread_id: str | None = None,
        mode: ChatMode = "auto",
    ) -> ChatResponse:
        if self.assistant is not None:
            return self._assistant_chat(
                question=question,
                top_k=top_k,
                session_id=session_id,
                thread_id=thread_id,
                mode=mode,
            )
        return self._legacy_rag_chat(question=question, top_k=top_k, session_id=session_id)

    @trace
    def _assistant_chat(
        self,
        *,
        question: str,
        top_k: int,
        session_id: str | None,
        thread_id: str | None,
        mode: ChatMode,
    ) -> ChatResponse:
        self._update_chat_trace(
            question=question,
            top_k=top_k,
            session_id=session_id,
            source_filenames=None,
            thread_id=thread_id,
        )
        forced_answer = self._memory_command(question)
        response = self.assistant.chat(
            question=question,
            top_k=top_k,
            thread_id=thread_id,
            mode=mode,
            forced_answer=forced_answer,
        )
        source_names = [
            source.filename or source.url or source.source
            for source in response.sources
            if source.filename or source.url or source.source
        ]
        self._update_chat_trace(
            question=question,
            top_k=top_k,
            session_id=session_id,
            source_filenames=source_names,
            answer_mode=response.resolved_mode,
            thread_id=response.thread_id,
            agent_run_id=response.agent_run_id,
        )
        update_current_trace(response_preview=response.answer[:500])
        return response

    def _memory_command(self, question: str) -> str | None:
        if self.state_store is None:
            return None
        stripped = question.strip()
        if stripped.lower().startswith("/remember"):
            content = stripped[len("/remember") :].strip()
            if not content:
                return "Usage: /remember <fact>"
            fact = self.state_store.add_memory_fact(content)
            return f"Saved memory {fact.id}: {fact.content}"
        if stripped.lower().startswith("/forget"):
            fact_id = stripped[len("/forget") :].strip()
            if not fact_id:
                return "Usage: /forget <memory-id>"
            deleted = self.state_store.delete_memory_fact(fact_id)
            return (
                f"Deleted memory {fact_id}."
                if deleted
                else f"No saved memory exists with id {fact_id}."
            )
        return None

    @trace
    def _legacy_rag_chat(
        self,
        *,
        question: str,
        top_k: int,
        session_id: str | None = None,
    ) -> ChatResponse:
        self._update_chat_trace(
            question=question,
            top_k=top_k,
            session_id=session_id,
            source_filenames=None,
        )
        try:
            with timed("chat.count_points"):
                if self.store.count_points() == 0:
                    raise NoIndexedDocumentsError()
            with timed("chat.embed_query", question_chars=len(question)):
                query_vector = self.embedder.embed_query(question)
            with timed("chat.retrieve", top_k=top_k):
                results = relevant_results(
                    self.store.search(query_vector, limit=top_k),
                    minimum_score=self.settings.retrieval_score_threshold,
                )
        except NoIndexedDocumentsError:
            return ChatResponse(
                answer="No indexed documents are available. Run /ingest first.",
                sources=[],
            )
        except Exception as exc:
            raise ServiceUnavailableError(f"Retrieval failed: {exc}") from exc

        if not results:
            self._update_chat_trace(
                question=question,
                top_k=top_k,
                session_id=session_id,
                source_filenames=[],
            )
            return ChatResponse(
                answer="No relevant indexed document chunks were found for that question.",
                sources=[],
            )

        self._update_chat_trace(
            question=question,
            top_k=top_k,
            session_id=session_id,
            source_filenames=_source_filenames(results),
        )
        with timed("chat.build_prompt", source_count=len(results)):
            user_prompt = build_user_prompt(question, results)
        with timed("chat.ollama", prompt_chars=len(user_prompt)):
            answer = self.chat_client.chat(user_prompt)
        update_current_trace(response_preview=answer[:500])
        return ChatResponse(answer=answer, sources=_sources_from_results(results))

    def _update_chat_trace(
        self,
        *,
        question: str,
        top_k: int,
        session_id: str | None,
        source_filenames: list[str] | None,
        answer_mode: str = "documents",
        thread_id: str | None = None,
        agent_run_id: str | None = None,
    ) -> None:
        metadata: dict[str, Any] = {}
        if session_id:
            metadata["rag.request_session_id"] = session_id
        if thread_id:
            metadata["rag.thread_id"] = thread_id
        if agent_run_id:
            metadata["rag.agent_run_id"] = agent_run_id
        if source_filenames is not None:
            metadata.update({
                "rag.top_k": top_k,
                "rag.chunk_size": self.settings.chunk_size,
                "rag.chunk_overlap": self.settings.chunk_overlap,
                "rag.qdrant_collection": self.settings.qdrant_collection,
                "rag.embedding_model": self.embedder.model_name,
                "rag.source_filenames": json.dumps(source_filenames),
                **self.chat_client.system_prompt_metadata,
            })
        update_current_trace(
            tags={
                "rag.operation": "chat",
                "rag.model": self.chat_client.model,
                "rag.answer_mode": answer_mode,
            },
            metadata=metadata,
            session_id=thread_id or session_id,
            user=self.settings.rag_user_name,
            request_preview=question[:500],
        )

    def create_thread(self, *, title: str | None = None) -> ThreadSummary:
        return self._require_state_store().ensure_thread(title=title)

    def list_threads(self) -> list[ThreadSummary]:
        return self._require_state_store().list_threads()

    def get_thread(self, thread_id: str) -> ThreadDetail | None:
        return self._require_assistant().thread_detail(thread_id)

    def delete_thread(self, thread_id: str) -> bool:
        with self._thread_lock:
            if self.agent_manager is not None:
                self.agent_manager.cancel_thread_runs(thread_id, wait=True)
            return self._require_assistant().delete_thread(thread_id)

    def list_memory_facts(self) -> list[MemoryFact]:
        return self._require_state_store().list_memory_facts()

    def add_memory_fact(self, content: str) -> MemoryFact:
        return self._require_state_store().add_memory_fact(content)

    def delete_memory_fact(self, fact_id: str) -> bool:
        return self._require_state_store().delete_memory_fact(fact_id)

    def list_agent_runs(self, *, thread_id: str | None = None) -> list[AgentRunInfo]:
        return self._require_state_store().list_agent_runs(thread_id=thread_id)

    def get_agent_run(self, run_id: str) -> AgentRunInfo | None:
        return self._require_state_store().get_agent_run(run_id)

    def launch_agent(
        self,
        *,
        agent_name: str,
        thread_id: str | None,
        task: str,
        top_k: int,
    ) -> AgentRunInfo:
        with self._thread_lock:
            thread = self._require_state_store().ensure_thread(thread_id)
            return self._require_agent_manager().launch(
                agent_name=agent_name,
                thread_id=thread.id,
                task=task,
                top_k=top_k,
            )

    def cancel_agent_run(self, run_id: str) -> AgentRunInfo:
        return self._require_agent_manager().cancel(run_id)

    def delete_agent_run(self, run_id: str) -> bool:
        return self._require_agent_manager().delete(run_id)

    def decide_agent_approval(
        self,
        run_id: str,
        *,
        decision: str,
        edited_payload: dict[str, Any] | None,
    ) -> AgentRunInfo:
        return self._require_agent_manager().decide_approval(
            run_id,
            decision=decision,  # type: ignore[arg-type]
            edited_payload=edited_payload,
        )

    def _require_state_store(self) -> AppStateStore:
        if self.state_store is None:
            raise ServiceUnavailableError("Persistent assistant state is not initialized.")
        return self.state_store

    def _require_assistant(self) -> ConversationGraph:
        if self.assistant is None:
            raise ServiceUnavailableError("LangGraph assistant is not initialized.")
        return self.assistant

    def _require_agent_manager(self) -> AgentManager:
        if self.agent_manager is None:
            raise ServiceUnavailableError("Agent runtime is not initialized.")
        return self.agent_manager


def build_service(settings: Settings, mlflow_status: dict[str, Any]) -> RagService:
    embedder = FastEmbedProvider(
        model_name=settings.fastembed_model,
        vector_size=settings.fastembed_vector_size,
        cuda=settings.fastembed_cuda,
        providers=settings.fastembed_providers,
        device_ids=settings.fastembed_device_ids,
        batch_size=settings.fastembed_batch_size,
    )
    store = QdrantStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key_value,
        collection_name=settings.qdrant_collection,
        vector_size=settings.fastembed_vector_size,
        score_threshold=settings.retrieval_score_threshold,
    )
    prompts = resolve_prompt_bundle(
        user_name=settings.rag_user_name,
        prompt_name=settings.mlflow_prompt_name,
        prompt_alias=settings.mlflow_prompt_alias,
        registry_enabled=bool(mlflow_status.get("configured")),
    )
    chat_client = OllamaChatClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        max_tokens=settings.ollama_max_tokens,
        system_prompt=prompts.get("documents"),
        system_prompt_metadata=prompts.metadata,
    )
    state_store = AppStateStore(settings.rag_state_db_path)
    web_search = OllamaWebSearchProvider(
        api_key=settings.ollama_api_key_value,
        max_results=settings.web_search_max_results,
    )
    assistant = ConversationGraph(
        settings=settings,
        state_store=state_store,
        store=store,
        embedder=embedder,
        chat_client=chat_client,
        web_search=web_search,
        prompts=prompts,
    )
    agent_manager = AgentManager(
        state_store=state_store,
        conversation=assistant,
        chat_client=chat_client,
        prompts=prompts,
        max_concurrent_runs=settings.agent_max_concurrent_runs,
        research_query_count=settings.research_query_count,
    )
    assistant.set_research_launcher(agent_manager.launch_document_research)
    return RagService(
        settings=settings,
        store=store,
        embedder=embedder,
        chat_client=chat_client,
        mlflow_status=mlflow_status,
        state_store=state_store,
        assistant=assistant,
        agent_manager=agent_manager,
        web_search=web_search,
    )


def _safe_health(func) -> dict[str, Any]:
    try:
        return func()
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _sources_from_results(results: Iterable[SearchResult]) -> list[SourceInfo]:
    sources = []
    for result in results:
        text = str(result.payload.get("text", ""))
        sources.append(
            SourceInfo(
                source=str(result.payload.get("source", "")),
                filename=str(result.payload.get("filename", "")),
                chunk_index=int(result.payload.get("chunk_index", 0)),
                score=result.score,
                snippet=_snippet(text),
            )
        )
    return sources


def _source_filenames(results: Iterable[SearchResult]) -> list[str]:
    filenames = {
        str(result.payload.get("filename", ""))
        for result in results
        if result.payload.get("filename")
    }
    return sorted(filenames)


def _snippet(text: str, limit: int = 240) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."
