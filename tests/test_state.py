from pathlib import Path

from rag_app.state import AppStateStore


def test_memory_and_threads_survive_store_reconstruction(tmp_path: Path):
    path = tmp_path / "state.sqlite3"
    store = AppStateStore(path)
    thread = store.ensure_thread(title="Persistent thread")
    fact = store.add_memory_fact("Prefers local inference")
    store.close()

    restored = AppStateStore(path)

    assert restored.get_thread(thread.id) == thread
    assert restored.list_memory_facts() == [fact]
    restored.close()


def test_delete_thread_removes_associated_agent_runs(tmp_path: Path):
    store = AppStateStore(tmp_path / "state.sqlite3")
    thread = store.ensure_thread()
    run = store.create_agent_run(
        agent_name="document_research",
        thread_id=thread.id,
        task="summarize notes",
        top_k=4,
    )

    assert store.delete_thread(thread.id)
    assert store.get_thread(thread.id) is None
    assert store.get_agent_run(run.id) is None
    store.close()


def test_reconstruction_marks_active_runs_interrupted(tmp_path: Path):
    path = tmp_path / "state.sqlite3"
    store = AppStateStore(path)
    thread = store.ensure_thread()
    run = store.create_agent_run(
        agent_name="document_research",
        thread_id=thread.id,
        task="summarize notes",
        top_k=4,
    )
    store.update_agent_run(run.id, state="running")
    store.close()

    restored = AppStateStore(path)

    assert restored.get_agent_run(run.id).state == "interrupted"
    restored.close()
