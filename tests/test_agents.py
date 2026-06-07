from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from time import monotonic, sleep

import pytest
from rag_app import agents as agents_module
from rag_app.agents import (
    AgentDefinition,
    AgentExecutionResult,
    AgentManager,
    ApprovalRequired,
    LangGraphAgentRunner,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from rag_app.prompt import PROMPT_TEMPLATES, PromptBundle
from rag_app.schemas import SourceInfo
from rag_app.state import AppStateStore
from typing_extensions import TypedDict


class FakeConversation:
    def __init__(self):
        self.appended: list[tuple[str, str]] = []

    def append_assistant_message(self, thread_id: str, content: str):
        self.appended.append((thread_id, content))

    def retrieve(self, question: str, *, top_k: int):
        return []


class BrokenAppendConversation(FakeConversation):
    def append_assistant_message(self, thread_id: str, content: str):
        raise RuntimeError("append failed")


class FakeChatClient:
    def structured_output(self, messages, schema):
        return schema(queries=["first", "second"])

    def chat_messages(self, messages):
        return "research answer"


def test_document_research_run_completes_and_appends_result(tmp_path: Path):
    manager, state, conversation = _manager(tmp_path)
    thread = state.ensure_thread()

    run = manager.launch_document_research(thread.id, "research notes", 4)
    completed = _wait_for_state(state, run.id, "completed")

    assert completed.result == "No relevant indexed document chunks were found for that research task."
    assert conversation.appended == [(thread.id, completed.result)]
    manager.close()
    state.close()


def test_completed_agent_future_is_released(tmp_path: Path):
    manager, state, _conversation = _manager(tmp_path)
    thread = state.ensure_thread()

    run = manager.launch_document_research(thread.id, "research notes", 4)
    _wait_for_state(state, run.id, "completed")
    _wait_for_condition(lambda: run.id not in manager._futures)

    manager.close()
    state.close()


def test_agent_run_is_failed_when_append_fails(tmp_path: Path):
    conversation = BrokenAppendConversation()
    manager, state, _conversation = _manager(tmp_path, conversation=conversation)
    thread = state.ensure_thread()

    run = manager.launch_document_research(thread.id, "research notes", 4)
    failed = _wait_for_state(state, run.id, "failed")

    assert "failed to append result" in failed.error
    manager.close()
    state.close()


def test_agent_run_trace_is_correlated_to_originating_thread(tmp_path: Path, monkeypatch):
    updates = []
    monkeypatch.setattr(agents_module, "update_current_trace", lambda **kwargs: updates.append(kwargs))
    manager, state, _conversation = _manager(tmp_path)
    thread = state.ensure_thread()

    run = manager.launch_document_research(thread.id, "research notes", 4)
    _wait_for_state(state, run.id, "completed")

    assert updates[0]["session_id"] == thread.id
    assert updates[0]["metadata"]["rag.agent_run_id"] == run.id
    assert updates[0]["metadata"]["rag.thread_id"] == thread.id
    assert updates[-1]["tags"]["rag.agent.state"] == "completed"
    manager.close()
    state.close()


def test_agent_launch_is_idempotent_while_active(tmp_path: Path):
    manager, state, _conversation = _manager(tmp_path)
    thread = state.ensure_thread()
    manager.register_agent(
        AgentDefinition(
            name="waiting",
            read_only=True,
            runner=lambda run, command: ApprovalRequired(payload={"task": run.task}),
        )
    )

    first = manager.launch(agent_name="waiting", thread_id=thread.id, task="same", top_k=4)
    _wait_for_state(state, first.id, "awaiting_approval")
    second = manager.launch(agent_name="waiting", thread_id=thread.id, task="same", top_k=4)

    assert second.id == first.id
    manager.close()
    state.close()


def test_mutating_agent_approval_can_edit_and_resume(tmp_path: Path):
    manager, state, conversation = _manager(tmp_path)
    thread = state.ensure_thread()

    def runner(run, command):
        if command is None:
            return ApprovalRequired(payload={"path": "before.txt"})
        return AgentExecutionResult(
            result="approved mutation",
            sources=[SourceInfo(kind="document", filename="audit.txt")],
        )

    manager.register_agent(AgentDefinition(name="mutator", read_only=False, runner=runner))
    run = manager.launch(agent_name="mutator", thread_id=thread.id, task="write file", top_k=4)
    waiting = _wait_for_state(state, run.id, "awaiting_approval")

    assert waiting.approval.payload == {"path": "before.txt"}
    manager.decide_approval(run.id, decision="edit", edited_payload={"path": "after.txt"})
    completed = _wait_for_state(state, run.id, "completed")

    assert completed.approval.decision["edited_payload"] == {"path": "after.txt"}
    assert conversation.appended == [(thread.id, "approved mutation")]
    manager.close()
    state.close()


def test_mutating_agent_approval_can_be_rejected(tmp_path: Path):
    manager, state, _conversation = _manager(tmp_path)
    thread = state.ensure_thread()
    manager.register_agent(
        AgentDefinition(
            name="mutator",
            read_only=False,
            runner=lambda run, command: ApprovalRequired(payload={"action": "write"}),
        )
    )
    run = manager.launch(agent_name="mutator", thread_id=thread.id, task="write file", top_k=4)
    _wait_for_state(state, run.id, "awaiting_approval")

    rejected = manager.decide_approval(run.id, decision="reject")

    assert rejected.state == "cancelled"
    manager.close()
    state.close()


def test_cancelled_approval_cannot_be_resumed(tmp_path: Path):
    manager, state, _conversation = _manager(tmp_path)
    thread = state.ensure_thread()
    manager.register_agent(
        AgentDefinition(
            name="mutator",
            read_only=False,
            runner=lambda run, command: ApprovalRequired(payload={"action": "write"}),
        )
    )
    run = manager.launch(agent_name="mutator", thread_id=thread.id, task="write file", top_k=4)
    _wait_for_state(state, run.id, "awaiting_approval")

    manager.cancel(run.id)

    with pytest.raises(ValueError, match="no longer awaiting approval"):
        manager.decide_approval(run.id, decision="approve")
    assert state.get_agent_run(run.id).state == "cancelled"
    manager.close()
    state.close()


def test_cancellation_before_completion_prevents_append(tmp_path: Path):
    manager, state, conversation = _manager(tmp_path)
    thread = state.ensure_thread()
    entered = Event()
    release = Event()

    def runner(run, command):
        entered.set()
        release.wait(timeout=3)
        return AgentExecutionResult(result="late result", sources=[])

    manager.register_agent(AgentDefinition(name="blocked", read_only=True, runner=runner))
    run = manager.launch(agent_name="blocked", thread_id=thread.id, task="wait", top_k=4)
    assert entered.wait(timeout=1)

    manager.cancel(run.id)
    release.set()
    cancelled = _wait_for_state(state, run.id, "cancelled")

    assert cancelled.cancel_requested
    assert conversation.appended == []
    manager.close()
    state.close()


def test_queued_cancellation_is_not_overwritten_by_worker_start(tmp_path: Path):
    manager, state, _conversation = _manager(tmp_path)
    thread = state.ensure_thread()
    entered = Event()
    release = Event()

    def runner(run, command):
        entered.set()
        release.wait(timeout=3)
        return AgentExecutionResult(result="done", sources=[])

    manager.register_agent(AgentDefinition(name="blocked", read_only=True, runner=runner))
    first = manager.launch(agent_name="blocked", thread_id=thread.id, task="first", top_k=4)
    assert entered.wait(timeout=1)
    second = manager.launch(agent_name="blocked", thread_id=thread.id, task="second", top_k=4)

    cancelled = manager.cancel(second.id)
    release.set()
    _wait_for_state(state, first.id, "completed")

    assert cancelled.state == "cancelled"
    assert state.get_agent_run(second.id).state == "cancelled"
    manager.close()
    state.close()


def test_close_drains_running_worker_before_state_store_closes(tmp_path: Path):
    manager, state, _conversation = _manager(tmp_path)
    thread = state.ensure_thread()
    entered = Event()
    release = Event()

    def runner(run, command):
        entered.set()
        release.wait(timeout=3)
        return AgentExecutionResult(result="done", sources=[])

    manager.register_agent(AgentDefinition(name="blocked", read_only=True, runner=runner))
    run = manager.launch(agent_name="blocked", thread_id=thread.id, task="wait", top_k=4)
    assert entered.wait(timeout=1)

    with ThreadPoolExecutor(max_workers=1) as executor:
        closing = executor.submit(manager.close)
        sleep(0.05)
        assert not closing.done()
        release.set()
        closing.result(timeout=1)

    assert state.get_agent_run(run.id).state == "cancelled"
    state.close()


def test_thread_run_cancellation_drains_worker_before_deletion(tmp_path: Path):
    manager, state, conversation = _manager(tmp_path)
    thread = state.ensure_thread()
    entered = Event()
    release = Event()

    def runner(run, command):
        entered.set()
        release.wait(timeout=3)
        return AgentExecutionResult(result="done", sources=[])

    manager.register_agent(AgentDefinition(name="blocked", read_only=True, runner=runner))
    manager.launch(agent_name="blocked", thread_id=thread.id, task="wait", top_k=4)
    assert entered.wait(timeout=1)

    with ThreadPoolExecutor(max_workers=1) as executor:
        deleting = executor.submit(manager.cancel_thread_runs, thread.id, wait=True)
        sleep(0.05)
        assert not deleting.done()
        release.set()
        deleting.result(timeout=1)

    assert state.delete_thread(thread.id)
    assert conversation.appended == []
    manager.close()
    state.close()


def test_langgraph_interrupt_adapter_pauses_and_resumes(tmp_path: Path):
    class MutatingState(TypedDict, total=False):
        task: str
        result: str

    def mutate(state: MutatingState):
        decision = interrupt({"action": "write", "task": state["task"]})
        return {"result": f"resumed with {decision['decision']}"}

    graph_builder = StateGraph(MutatingState)
    graph_builder.add_node("mutate", mutate)
    graph_builder.add_edge(START, "mutate")
    graph_builder.add_edge("mutate", END)
    graph = graph_builder.compile(checkpointer=InMemorySaver())

    manager, state, _conversation = _manager(tmp_path)
    thread = state.ensure_thread()
    manager.register_agent(
        AgentDefinition(
            name="langgraph-mutator",
            read_only=False,
            runner=LangGraphAgentRunner(graph),
        )
    )
    run = manager.launch(
        agent_name="langgraph-mutator",
        thread_id=thread.id,
        task="write report",
        top_k=4,
    )
    waiting = _wait_for_state(state, run.id, "awaiting_approval")

    assert waiting.approval.payload["action"] == "write"
    manager.decide_approval(run.id, decision="approve")
    completed = _wait_for_state(state, run.id, "completed")

    assert completed.result == "resumed with approve"
    manager.close()
    state.close()


def _manager(tmp_path: Path, *, conversation: FakeConversation | None = None):
    state = AppStateStore(tmp_path / "state.sqlite3")
    conversation = conversation or FakeConversation()
    manager = AgentManager(
        state_store=state,
        conversation=conversation,
        chat_client=FakeChatClient(),
        prompts=PromptBundle(prompts=dict(PROMPT_TEMPLATES), metadata={}),
        max_concurrent_runs=1,
        research_query_count=3,
    )
    return manager, state, conversation


def _wait_for_state(state: AppStateStore, run_id: str, expected: str):
    deadline = monotonic() + 3
    while monotonic() < deadline:
        run = state.get_agent_run(run_id)
        if run.state == expected:
            return run
        sleep(0.01)
    raise AssertionError(f"Agent run {run_id} did not reach state {expected!r}.")


def _wait_for_condition(condition):
    deadline = monotonic() + 3
    while monotonic() < deadline:
        if condition():
            return
        sleep(0.01)
    raise AssertionError("Condition was not satisfied before timeout.")
