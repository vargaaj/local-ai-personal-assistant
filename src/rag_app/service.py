from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

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
from .prompt import build_user_prompt, resolve_system_prompt
from .qdrant_store import QdrantStore, SearchResult
from .schemas import ChatResponse, FileError, IngestResponse, SourceInfo
from .tracing import timed, trace, update_current_trace


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
    ) -> None:
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self.chat_client = chat_client
        self.mlflow_status = mlflow_status
        self.manifest = manifest or IngestManifest(settings.rag_manifest_path)

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
            },
        }

    @trace
    def ingest(self, *, reset: bool = False) -> IngestResponse:
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

        for path in iter_supported_files(self.settings.rag_docs_root):
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
                results = self.store.search(query_vector, limit=top_k)
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
    ) -> None:
        metadata: dict[str, Any] = {}
        if source_filenames is not None:
            metadata = {
                "rag.top_k": top_k,
                "rag.chunk_size": self.settings.chunk_size,
                "rag.chunk_overlap": self.settings.chunk_overlap,
                "rag.qdrant_collection": self.settings.qdrant_collection,
                "rag.embedding_model": self.embedder.model_name,
                "rag.source_filenames": json.dumps(source_filenames),
                **self.chat_client.system_prompt_metadata,
            }
        update_current_trace(
            tags={
                "rag.operation": "chat",
                "rag.model": self.chat_client.model,
                "rag.answer_mode": "balanced",
            },
            metadata=metadata,
            session_id=session_id,
            user=self.settings.rag_user_name,
            request_preview=question[:500],
        )


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
    )
    system_prompt = resolve_system_prompt(
        user_name=settings.rag_user_name,
        prompt_name=settings.mlflow_prompt_name,
        prompt_alias=settings.mlflow_prompt_alias,
        registry_enabled=bool(mlflow_status.get("configured")),
    )
    chat_client = OllamaChatClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        max_tokens=settings.ollama_max_tokens,
        system_prompt=system_prompt.content,
        system_prompt_metadata=system_prompt.trace_metadata(),
    )
    return RagService(
        settings=settings,
        store=store,
        embedder=embedder,
        chat_client=chat_client,
        mlflow_status=mlflow_status,
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
