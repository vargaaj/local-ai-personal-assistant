from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock
from time import sleep
from typing import Any

from rag_app.config import Settings
from rag_app.documents import file_sha256
from rag_app.manifest import IngestManifest
from rag_app import service as service_module
from rag_app.service import RagService


class FakeStore:
    def __init__(self, *, point_count: int = 1) -> None:
        self.point_count = point_count
        self.deleted_sources: list[str] = []
        self.upserts: list[dict[str, Any]] = []
        self.reset_called = False
        self.ensure_called = False
        self.count_called = False

    def ensure_collection(self) -> None:
        self.ensure_called = True

    def reset_collection(self) -> None:
        self.reset_called = True
        self.point_count = 0

    def count_points(self) -> int:
        self.count_called = True
        return self.point_count

    def delete_source(self, source: str) -> None:
        self.deleted_sources.append(source)

    def upsert_chunks(self, chunks, vectors, **kwargs: Any) -> None:
        self.upserts.append({"chunks": chunks, "vectors": vectors, "kwargs": kwargs})
        self.point_count += len(chunks)


class FakeEmbedder:
    model_name = "fake-embedding-model"
    vector_size = 3

    def __init__(self) -> None:
        self.embedded_texts: list[list[str]] = []

    def embed_texts(self, texts) -> list[list[float]]:
        texts = list(texts)
        self.embedded_texts.append(texts)
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


class FakeChatClient:
    model = "fake-chat-model"
    system_prompt_metadata = {}

    def health(self) -> dict[str, bool]:
        return {"ok": True}

    def chat(self, prompt: str) -> str:
        return prompt


def test_ingest_skips_unchanged_indexed_files(tmp_path: Path):
    path = tmp_path / "notes.txt"
    path.write_text("alpha beta gamma", encoding="utf-8")
    store = FakeStore(point_count=1)
    embedder = FakeEmbedder()
    manifest = _manifest(tmp_path)
    manifest.update(
        source=str(path),
        source_sha256=file_sha256(path),
        qdrant_collection="local_documents",
        embedding_model="fake-embedding-model",
        embedding_vector_size=3,
        chunk_size=1200,
        chunk_overlap=200,
        chunk_count=1,
    )
    service = _service(tmp_path, store, embedder, manifest)

    response = service.ingest(reset=False)

    assert response.files_processed == 0
    assert response.chunks_indexed == 0
    assert response.skipped_files == [str(path)]
    assert embedder.embedded_texts == []
    assert store.deleted_sources == []
    assert store.upserts == []
    assert store.count_called


def test_ingest_reindexes_changed_files(tmp_path: Path):
    path = tmp_path / "notes.txt"
    path.write_text("alpha beta gamma", encoding="utf-8")
    store = FakeStore(point_count=1)
    embedder = FakeEmbedder()
    manifest = _manifest(tmp_path)
    service = _service(tmp_path, store, embedder, manifest)

    response = service.ingest(reset=False)

    assert response.files_processed == 1
    assert response.chunks_indexed == 1
    assert store.deleted_sources == [str(path)]
    assert len(embedder.embedded_texts) == 1
    assert len(store.upserts) == 1
    assert store.upserts[0]["kwargs"] == {
        "embedding_model": "fake-embedding-model",
        "chunk_size": 1200,
        "chunk_overlap": 200,
    }
    assert manifest.is_current(
        source=str(path),
        source_sha256=file_sha256(path),
        qdrant_collection="local_documents",
        embedding_model="fake-embedding-model",
        embedding_vector_size=3,
        chunk_size=1200,
        chunk_overlap=200,
    )


def test_ingest_removes_stale_chunks_for_empty_files(tmp_path: Path):
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    store = FakeStore(point_count=1)
    embedder = FakeEmbedder()
    manifest = _manifest(tmp_path)
    manifest.update(
        source=str(path),
        source_sha256="old-hash",
        qdrant_collection="local_documents",
        embedding_model="fake-embedding-model",
        embedding_vector_size=3,
        chunk_size=1200,
        chunk_overlap=200,
        chunk_count=1,
    )
    service = _service(tmp_path, store, embedder, manifest)

    response = service.ingest(reset=False)

    assert response.files_processed == 0
    assert response.chunks_indexed == 0
    assert response.skipped_files == [str(path)]
    assert store.deleted_sources == [str(path)]
    assert embedder.embedded_texts == []
    assert store.upserts == []
    assert not manifest.is_current(
        source=str(path),
        source_sha256="old-hash",
        qdrant_collection="local_documents",
        embedding_model="fake-embedding-model",
        embedding_vector_size=3,
        chunk_size=1200,
        chunk_overlap=200,
    )


def test_ingest_ignores_manifest_when_qdrant_collection_is_empty(tmp_path: Path):
    path = tmp_path / "notes.txt"
    path.write_text("alpha beta gamma", encoding="utf-8")
    store = FakeStore(point_count=0)
    embedder = FakeEmbedder()
    manifest = _manifest(tmp_path)
    manifest.update(
        source=str(path),
        source_sha256=file_sha256(path),
        qdrant_collection="local_documents",
        embedding_model="fake-embedding-model",
        embedding_vector_size=3,
        chunk_size=1200,
        chunk_overlap=200,
        chunk_count=1,
    )
    service = _service(tmp_path, store, embedder, manifest)

    response = service.ingest(reset=False)

    assert response.files_processed == 1
    assert response.chunks_indexed == 1
    assert len(embedder.embedded_texts) == 1


def test_ingest_removes_chunks_for_deleted_source_files(tmp_path: Path):
    path = tmp_path / "deleted.txt"
    path.write_text("remove this", encoding="utf-8")
    store = FakeStore(point_count=1)
    embedder = FakeEmbedder()
    manifest = _manifest(tmp_path)
    manifest.update(
        source=str(path),
        source_sha256=file_sha256(path),
        qdrant_collection="local_documents",
        embedding_model="fake-embedding-model",
        embedding_vector_size=3,
        chunk_size=1200,
        chunk_overlap=200,
        chunk_count=1,
    )
    path.unlink()
    service = _service(tmp_path, store, embedder, manifest)

    service.ingest(reset=False)

    assert store.deleted_sources == [str(path)]
    assert manifest.sources() == set()


def test_ingest_serializes_concurrent_calls(tmp_path: Path):
    service = _service(tmp_path, FakeStore(), FakeEmbedder(), _manifest(tmp_path))
    entered = Event()
    release = Event()
    counter_lock = Lock()
    active = 0
    maximum_active = 0

    def slow_ingest(*, reset: bool):
        nonlocal active, maximum_active
        with counter_lock:
            active += 1
            maximum_active = max(maximum_active, active)
            entered.set()
        release.wait(timeout=3)
        with counter_lock:
            active -= 1
        return reset

    service._ingest = slow_ingest  # type: ignore[method-assign]
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(service.ingest, reset=False)
        assert entered.wait(timeout=1)
        second = executor.submit(service.ingest, reset=True)
        sleep(0.05)
        assert maximum_active == 1
        release.set()
        assert first.result(timeout=1) is False
        assert second.result(timeout=1) is True


def test_delete_thread_drains_agent_runs_before_removing_thread(tmp_path: Path):
    events = []

    class FakeAgentManager:
        def cancel_thread_runs(self, thread_id: str, *, wait: bool) -> None:
            events.append(("cancel", thread_id, wait))

    class FakeAssistant:
        def delete_thread(self, thread_id: str) -> bool:
            events.append(("delete", thread_id))
            return True

    service = _service(tmp_path, FakeStore(), FakeEmbedder(), _manifest(tmp_path))
    service.agent_manager = FakeAgentManager()
    service.assistant = FakeAssistant()

    assert service.delete_thread("thread-1")
    assert events == [
        ("cancel", "thread-1", True),
        ("delete", "thread-1"),
    ]


def test_chat_trace_uses_persistent_thread_and_keeps_request_session_metadata(
    tmp_path: Path,
    monkeypatch,
):
    service = _service(tmp_path, FakeStore(), FakeEmbedder(), _manifest(tmp_path))
    updates = []
    monkeypatch.setattr(service_module, "update_current_trace", lambda **kwargs: updates.append(kwargs))

    service._update_chat_trace(
        question="hello",
        top_k=4,
        session_id="browser-session",
        thread_id="persistent-thread",
        agent_run_id="run-1",
        source_filenames=["notes.txt"],
        answer_mode="documents",
    )

    assert updates[0]["session_id"] == "persistent-thread"
    assert updates[0]["metadata"]["rag.request_session_id"] == "browser-session"
    assert updates[0]["metadata"]["rag.thread_id"] == "persistent-thread"
    assert updates[0]["metadata"]["rag.agent_run_id"] == "run-1"


def _service(
    tmp_path: Path,
    store: FakeStore,
    embedder: FakeEmbedder,
    manifest: IngestManifest,
) -> RagService:
    settings = Settings(
        _env_file=None,
        qdrant_api_key="secret",
        rag_docs_root=tmp_path,
        rag_manifest_path=manifest.path,
        fastembed_cuda=False,
        fastembed_providers=[],
    )
    return RagService(
        settings=settings,
        store=store,
        embedder=embedder,
        chat_client=FakeChatClient(),
        mlflow_status={},
        manifest=manifest,
    )


def _manifest(tmp_path: Path) -> IngestManifest:
    return IngestManifest(tmp_path / "manifest.json")
