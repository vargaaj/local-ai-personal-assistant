from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import RLock
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from .assistant import ConversationGraph, sources_from_results
from .errors import NoIndexedDocumentsError
from .ollama_client import OllamaChatClient
from .prompt import PromptBundle, build_user_prompt
from .qdrant_store import SearchResult
from .schemas import AgentRunInfo, SourceInfo
from .state import AppStateStore
from .tracing import timed, trace, update_current_trace


class ResearchQueries(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=8)


@dataclass(frozen=True)
class AgentExecutionResult:
    result: str
    sources: list[SourceInfo]


@dataclass(frozen=True)
class ApprovalRequired:
    payload: dict[str, Any]


AgentOutcome = AgentExecutionResult | ApprovalRequired
AgentRunner = Callable[[AgentRunInfo, Command | None], AgentOutcome]


class ResearchGraphState(TypedDict, total=False):
    run_id: str
    task: str
    top_k: int
    queries: list[str]
    results: list[SearchResult]
    answer: str
    sources: list[SourceInfo]


class LangGraphAgentRunner:
    """Adapt an interrupt-capable compiled LangGraph to the agent runtime contract."""

    def __init__(self, graph: Any) -> None:
        self.graph = graph

    def __call__(self, run: AgentRunInfo, command: Command | None) -> AgentOutcome:
        config = {"configurable": {"thread_id": f"agent-{run.id}"}}
        graph_input: Any = command if command is not None else {"task": run.task}
        result = self.graph.invoke(graph_input, config)
        interrupts = result.get("__interrupt__", ())
        if interrupts:
            value = getattr(interrupts[0], "value", interrupts[0])
            payload = value if isinstance(value, dict) else {"request": value}
            return ApprovalRequired(payload=payload)
        sources = [SourceInfo.model_validate(item) for item in result.get("sources", [])]
        return AgentExecutionResult(result=str(result.get("result", "")), sources=sources)


@dataclass(frozen=True)
class AgentDefinition:
    name: str
    read_only: bool
    runner: AgentRunner


class AgentManager:
    def __init__(
        self,
        *,
        state_store: AppStateStore,
        conversation: ConversationGraph,
        chat_client: OllamaChatClient,
        prompts: PromptBundle,
        max_concurrent_runs: int,
        research_query_count: int,
    ) -> None:
        self.state_store = state_store
        self.conversation = conversation
        self.chat_client = chat_client
        self.prompts = prompts
        self.research_query_count = research_query_count
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrent_runs,
            thread_name_prefix="rag-agent",
        )
        self._lock = RLock()
        self._futures: dict[str, Future[Any]] = {}
        self._registry: dict[str, AgentDefinition] = {}
        self._closing = False
        self._research_graph = self._build_research_graph().compile()
        self.register_agent(
            AgentDefinition(
                name="document_research",
                read_only=True,
                runner=self._run_document_research,
            )
        )

    def close(self) -> None:
        with self._lock:
            if self._closing:
                return
            self._closing = True
            self.state_store.cancel_active_agent_runs(include_awaiting_approval=False)
            futures = list(self._futures.values())
        for future in futures:
            future.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)

    def register_agent(self, definition: AgentDefinition) -> None:
        self._registry[definition.name] = definition

    def agent_names(self) -> list[str]:
        return sorted(self._registry)

    def launch(
        self,
        *,
        agent_name: str,
        thread_id: str,
        task: str,
        top_k: int,
    ) -> AgentRunInfo:
        with self._lock:
            if self._closing:
                raise ValueError("Agent runtime is shutting down.")
            if agent_name not in self._registry:
                raise ValueError(f"Unknown agent {agent_name!r}.")
            run = self.state_store.create_agent_run(
                agent_name=agent_name,
                thread_id=thread_id,
                task=task.strip(),
                top_k=top_k,
            )
            self._submit(run.id)
        return run

    def launch_document_research(self, thread_id: str, task: str, top_k: int) -> AgentRunInfo:
        return self.launch(
            agent_name="document_research",
            thread_id=thread_id,
            task=task,
            top_k=top_k,
        )

    def cancel(self, run_id: str) -> AgentRunInfo:
        run = self.state_store.request_agent_cancel(run_id)
        with self._lock:
            future = self._futures.get(run_id)
            if future and run.state == "cancelled":
                future.cancel()
        return run

    def delete(self, run_id: str) -> bool:
        with self._lock:
            future = self._futures.get(run_id)
            if future and not future.done() and not future.cancel():
                raise ValueError("Cancel the agent run before deleting it.")
            if self._futures.get(run_id) is future:
                self._futures.pop(run_id, None)
        return self.state_store.delete_agent_run(run_id)

    def cancel_thread_runs(self, thread_id: str, *, wait: bool) -> None:
        self.state_store.cancel_active_agent_runs(thread_id=thread_id)
        with self._lock:
            futures = [
                future
                for run_id, future in self._futures.items()
                if (
                    (run := self.state_store.get_agent_run(run_id)) is not None
                    and run.thread_id == thread_id
                )
            ]
        for future in futures:
            future.cancel()
        if wait:
            for future in futures:
                try:
                    future.result()
                except Exception:
                    pass

    def decide_approval(
        self,
        run_id: str,
        *,
        decision: Literal["approve", "edit", "reject"],
        edited_payload: dict[str, Any] | None = None,
    ) -> AgentRunInfo:
        with self._lock:
            if self._closing:
                raise ValueError("Agent runtime is shutting down.")
            self.state_store.decide_approval(
                run_id,
                decision=decision,
                edited_payload=edited_payload,
            )
            if decision == "reject":
                run = self.state_store.mark_agent_cancelled(
                    run_id,
                    error="User rejected the requested action.",
                )
                assert run is not None
                return run
            run = self.state_store.queue_agent_run_after_approval(run_id)
            if run is None:
                current = self.state_store.get_agent_run(run_id)
                assert current is not None
                return current
            self._submit(
                run_id,
                command=Command(
                    resume={
                        "decision": decision,
                        "edited_payload": edited_payload,
                    }
                ),
            )
            return run

    def _submit(self, run_id: str, *, command: Command | None = None) -> None:
        with self._lock:
            if self._closing:
                return
            existing = self._futures.get(run_id)
            if existing and not existing.done():
                if command is not None:
                    existing.add_done_callback(
                        lambda _future: self._submit(run_id, command=command)
                    )
                return
            run = self.state_store.get_agent_run(run_id)
            if run is None or run.state not in {"queued", "running"}:
                return
            future = self._executor.submit(self._execute, run_id, command)
            self._futures[run_id] = future
            future.add_done_callback(
                lambda completed, submitted_run_id=run_id: self._forget_future(
                    submitted_run_id,
                    completed,
                )
            )

    def _forget_future(self, run_id: str, future: Future[Any]) -> None:
        with self._lock:
            if self._futures.get(run_id) is future:
                self._futures.pop(run_id, None)

    @trace
    def _execute(self, run_id: str, command: Command | None) -> None:
        run = self.state_store.try_start_agent_run(run_id)
        if run is None:
            return
        definition = self._registry[run.agent_name]
        update_current_trace(
            tags={
                "rag.operation": "agent_run",
                "rag.agent_name": run.agent_name,
            },
            metadata={
                "rag.agent_run_id": run.id,
                "rag.thread_id": run.thread_id,
                "rag.agent.read_only": definition.read_only,
            },
            session_id=run.thread_id,
            request_preview=run.task[:500],
        )
        try:
            outcome = definition.runner(run, command)
            if isinstance(outcome, ApprovalRequired):
                update_current_trace(
                    tags={"rag.agent.state": "awaiting_approval"},
                )
                if self.state_store.create_approval(run_id, outcome.payload) is None:
                    self.state_store.mark_agent_cancelled(run_id)
                return
            completed = self.state_store.complete_agent_run(
                run_id,
                result=outcome.result,
                sources=outcome.sources,
            )
            if completed is None:
                self.state_store.mark_agent_cancelled(run_id)
                return
            try:
                self.conversation.append_assistant_message(run.thread_id, outcome.result)
            except Exception as exc:
                self.state_store.update_agent_run(
                    run_id,
                    state="failed",
                    error=f"Agent completed but failed to append result: {exc}",
                )
                raise
            update_current_trace(
                tags={"rag.agent.state": "completed"},
                response_preview=outcome.result[:500],
            )
        except Exception as exc:
            failed = self.state_store.fail_agent_run(run_id, error=str(exc))
            if failed is None:
                self.state_store.mark_agent_cancelled(run_id)
            update_current_trace(
                tags={"rag.agent.state": "failed"},
                response_preview=str(exc)[:500],
            )

    def _run_document_research(
        self,
        run: AgentRunInfo,
        _command: Command | None,
    ) -> AgentExecutionResult:
        if run.cancel_requested:
            return AgentExecutionResult(result="Document research was cancelled.", sources=[])
        result = self._research_graph.invoke(
            {
                "run_id": run.id,
                "task": run.task,
                "top_k": run.top_k,
            }
        )
        return AgentExecutionResult(
            result=str(result.get("answer", "")),
            sources=list(result.get("sources", [])),
        )

    def _build_research_graph(self) -> StateGraph:
        graph = StateGraph(ResearchGraphState)
        graph.add_node("plan", self._research_plan)
        graph.add_node("retrieve", self._research_retrieve)
        graph.add_node("synthesize", self._research_synthesize)
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "retrieve")
        graph.add_edge("retrieve", "synthesize")
        graph.add_edge("synthesize", END)
        return graph

    def _research_plan(self, state: ResearchGraphState) -> dict[str, Any]:
        return {"queries": self._plan_queries(state["task"])}

    def _research_retrieve(self, state: ResearchGraphState) -> dict[str, Any]:
        results: list[SearchResult] = []
        seen_ids: set[str] = set()
        for query in state["queries"]:
            current = self.state_store.get_agent_run(state["run_id"])
            if current is not None and current.cancel_requested:
                return {"results": []}
            try:
                found = self.conversation.retrieve(query, top_k=state["top_k"])
            except NoIndexedDocumentsError:
                found = []
            for result in found:
                result_id = str(result.id)
                if result_id not in seen_ids:
                    seen_ids.add(result_id)
                    results.append(result)
        return {"results": results}

    def _research_synthesize(self, state: ResearchGraphState) -> dict[str, Any]:
        results = state["results"]
        if not results:
            return {
                "answer": "No relevant indexed document chunks were found for that research task.",
                "sources": [],
            }
        with timed("research.synthesize", sources=len(results)):
            answer = self.chat_client.chat_messages(
                [
                    SystemMessage(content=self.prompts.get("research-synthesis")),
                    HumanMessage(content=build_user_prompt(state["task"], results)),
                ]
            )
        return {"answer": answer, "sources": sources_from_results(results)}

    def _plan_queries(self, task: str) -> list[str]:
        try:
            planned = self.chat_client.structured_output(
                [
                    SystemMessage(content=self.prompts.get("research-planner")),
                    HumanMessage(
                        content=(
                            f"Return {self.research_query_count} distinct search queries. "
                            f"Research task: {task}"
                        )
                    ),
                ],
                ResearchQueries,
            )
            queries = [query.strip() for query in planned.queries if query.strip()]
        except Exception:
            queries = []
        if task not in queries:
            queries.insert(0, task)
        return queries[: self.research_query_count]
