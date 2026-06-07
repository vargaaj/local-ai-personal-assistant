from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from threading import RLock
from typing import Annotated, Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from typing_extensions import TypedDict

from .config import Settings
from .embeddings import EmbeddingProvider
from .errors import NoIndexedDocumentsError, ServiceUnavailableError
from .ollama_client import OllamaChatClient
from .prompt import PromptBundle, build_user_prompt, build_web_user_prompt
from .qdrant_store import QdrantStore, SearchResult
from .schemas import ChatMode, ChatResponse, SourceInfo, ThreadDetail, ThreadMessage
from .state import AppStateStore, open_sqlite
from .tracing import timed, trace, update_current_trace
from .web_search import WebSearchProvider, source_info_from_web_result

logger = logging.getLogger(__name__)


class RouteDecision(BaseModel):
    mode: Literal["general", "documents", "web", "research"]
    reason: str = ""


class AssistantGraphState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    requested_mode: ChatMode
    resolved_mode: Literal["general", "documents", "web", "research", "command"]
    top_k: int
    thread_id: str
    answer: str
    forced_answer: str
    sources: list[dict[str, Any]]
    warnings: list[str]
    agent_run_id: str | None


ResearchLauncher = Callable[[str, str, int], Any]


class ConversationGraph:
    def __init__(
        self,
        *,
        settings: Settings,
        state_store: AppStateStore,
        store: QdrantStore,
        embedder: EmbeddingProvider,
        chat_client: OllamaChatClient,
        web_search: WebSearchProvider,
        prompts: PromptBundle,
    ) -> None:
        from langgraph.checkpoint.sqlite import SqliteSaver

        self.settings = settings
        self.state_store = state_store
        self.store = store
        self.embedder = embedder
        self.chat_client = chat_client
        self.web_search = web_search
        self.prompts = prompts
        self._lock = RLock()
        self._research_launcher: ResearchLauncher | None = None
        self._checkpoint_connection = open_sqlite(settings.rag_state_db_path)
        self.checkpointer = SqliteSaver(self._checkpoint_connection)
        self.graph = self._build_graph().compile(checkpointer=self.checkpointer)

    def close(self) -> None:
        with self._lock:
            self._checkpoint_connection.close()

    def set_research_launcher(self, launcher: ResearchLauncher) -> None:
        self._research_launcher = launcher

    @trace
    def chat(
        self,
        *,
        question: str,
        top_k: int,
        thread_id: str | None,
        mode: ChatMode,
        forced_answer: str | None = None,
    ) -> ChatResponse:
        thread = self.state_store.ensure_thread(thread_id)
        self.state_store.touch_thread(thread.id, question=question)
        inputs: AssistantGraphState = {
            "messages": [HumanMessage(content=question)],
            "question": question,
            "requested_mode": mode,
            "top_k": top_k,
            "thread_id": thread.id,
            "warnings": [],
            "sources": [],
            "agent_run_id": None,
            "forced_answer": forced_answer or "",
        }

        with self._lock, timed("langgraph.chat", thread_id=thread.id, requested_mode=mode):
            result = self.graph.invoke(inputs, self._config(thread.id))
        answer = str(result.get("answer", ""))
        resolved_mode = result.get("resolved_mode", "general")
        if resolved_mode == "command":
            resolved_mode = "general"
        sources = [SourceInfo.model_validate(source) for source in result.get("sources", [])]
        warnings = [str(warning) for warning in result.get("warnings", [])]
        return ChatResponse(
            answer=answer,
            sources=sources,
            thread_id=thread.id,
            resolved_mode=resolved_mode,
            agent_run_id=result.get("agent_run_id"),
            warnings=warnings,
        )

    def thread_detail(self, thread_id: str) -> ThreadDetail | None:
        thread = self.state_store.get_thread(thread_id)
        if thread is None:
            return None
        with self._lock:
            snapshot = self.graph.get_state(self._config(thread_id))
        raw_messages = list((snapshot.values or {}).get("messages", []))
        return ThreadDetail(
            **thread.model_dump(),
            messages=[_thread_message(message) for message in raw_messages],
        )

    def append_assistant_message(self, thread_id: str, content: str) -> None:
        self.state_store.ensure_thread(thread_id)
        with self._lock:
            self.graph.update_state(
                self._config(thread_id),
                {"messages": [AIMessage(content=content)]},
            )
        self.state_store.touch_thread(thread_id)

    def delete_thread(self, thread_id: str) -> bool:
        with self._lock:
            return self.state_store.delete_thread(thread_id)

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AssistantGraphState)
        graph.add_node("route", self._route)
        graph.add_node("general", self._general)
        graph.add_node("documents", self._documents)
        graph.add_node("web", self._web)
        graph.add_node("research", self._research)
        graph.add_node("command", self._command)
        graph.add_edge(START, "route")
        graph.add_conditional_edges(
            "route",
            lambda state: state["resolved_mode"],
            {
                "general": "general",
                "documents": "documents",
                "web": "web",
                "research": "research",
                "command": "command",
            },
        )
        graph.add_edge("general", END)
        graph.add_edge("documents", END)
        graph.add_edge("web", END)
        graph.add_edge("research", END)
        graph.add_edge("command", END)
        return graph

    def _route(self, state: AssistantGraphState) -> dict[str, Any]:
        if state.get("forced_answer"):
            return {"resolved_mode": "command"}

        requested_mode = state.get("requested_mode", "auto")
        warnings = list(state.get("warnings", []))
        if requested_mode != "auto":
            return {"resolved_mode": requested_mode, "warnings": warnings}

        try:
            decision = self.chat_client.structured_output(
                [
                    SystemMessage(content=self.prompts.get("router")),
                    HumanMessage(content=state["question"]),
                ],
                RouteDecision,
            )
            resolved_mode = decision.mode
        except Exception as exc:
            resolved_mode = "general"
            warnings.append(f"Automatic routing failed; using general chat. {exc}")

        if resolved_mode == "web" and not self.web_search.configured:
            resolved_mode = "general"
            warnings.append(
                "Live web search was selected automatically but OLLAMA_API_KEY is not configured; "
                "using general chat."
            )
        return {"resolved_mode": resolved_mode, "warnings": warnings}

    def _general(self, state: AssistantGraphState) -> dict[str, Any]:
        messages = [
            SystemMessage(content=self._local_system_prompt("general")),
            *_conversation_messages(state["messages"]),
        ]
        answer = self.chat_client.chat_messages(messages)
        return _answer_update(answer, warnings=state.get("warnings", []))

    def _documents(self, state: AssistantGraphState) -> dict[str, Any]:
        try:
            results = self.retrieve(state["question"], top_k=state["top_k"])
        except NoIndexedDocumentsError:
            return _answer_update(
                "No indexed documents are available. Run /ingest first.",
                warnings=state.get("warnings", []),
            )
        if not results:
            return _answer_update(
                "No relevant indexed document chunks were found for that question.",
                warnings=state.get("warnings", []),
            )

        sources = sources_from_results(results)
        messages = [
            SystemMessage(content=self._local_system_prompt("documents")),
            HumanMessage(content=build_user_prompt(state["question"], results)),
        ]
        answer = self.chat_client.chat_messages(messages)
        return _answer_update(answer, sources=sources, warnings=state.get("warnings", []))

    def _web(self, state: AssistantGraphState) -> dict[str, Any]:
        if not self.web_search.configured:
            return _answer_update(
                "Live web search is not configured. Set OLLAMA_API_KEY to use Web mode.",
                warnings=[*state.get("warnings", []), "OLLAMA_API_KEY is not configured."],
            )
        try:
            with timed("chat.web_search", query_chars=len(state["question"])):
                results = self.web_search.search(state["question"])
        except Exception as exc:
            logger.warning("Live web search failed: %s", exc)
            warning = "Live web search is temporarily unavailable."
            if state.get("requested_mode") == "auto":
                return {
                    "resolved_mode": "general",
                    **self._general(
                        {
                            **state,
                            "warnings": [*state.get("warnings", []), f"{warning} Using general chat."],
                        }
                    ),
                }
            return _answer_update(
                f"{warning} Try again later or use General mode.",
                warnings=[*state.get("warnings", []), warning],
            )
        sources = [source_info_from_web_result(result) for result in results]
        messages = [
            SystemMessage(content=self.prompts.get("web")),
            HumanMessage(content=build_web_user_prompt(state["question"], results)),
        ]
        answer = self.chat_client.chat_messages(messages)
        return _answer_update(answer, sources=sources, warnings=state.get("warnings", []))

    def _research(self, state: AssistantGraphState) -> dict[str, Any]:
        if self._research_launcher is None:
            raise ServiceUnavailableError("Document research agent runtime is not initialized.")
        run = self._research_launcher(state["thread_id"], state["question"], state["top_k"])
        answer = f"Started document research run {run.id}. The sourced result will be appended here."
        return _answer_update(
            answer,
            agent_run_id=run.id,
            warnings=state.get("warnings", []),
        )

    def _command(self, state: AssistantGraphState) -> dict[str, Any]:
        return _answer_update(
            str(state.get("forced_answer", "")),
            warnings=state.get("warnings", []),
        )

    def _local_system_prompt(self, role: str) -> str:
        facts = self.state_store.list_memory_facts(limit=self.settings.memory_fact_limit)
        if not facts:
            return self.prompts.get(role)
        rendered_facts = "\n".join(f"- {fact.content}" for fact in reversed(facts))
        return f"{self.prompts.get(role)}\n\nExplicitly saved facts:\n{rendered_facts}"

    @trace
    def retrieve(self, question: str, *, top_k: int) -> list[SearchResult]:
        try:
            with timed("chat.count_points"):
                if self.store.count_points() == 0:
                    raise NoIndexedDocumentsError()
            with timed("chat.embed_query", question_chars=len(question)):
                query_vector = self.embedder.embed_query(question)
            with timed("chat.retrieve", top_k=top_k):
                results = self.store.search(query_vector, limit=top_k)
            return relevant_results(
                results,
                minimum_score=self.settings.retrieval_score_threshold,
            )
        except NoIndexedDocumentsError:
            raise
        except Exception as exc:
            raise ServiceUnavailableError(f"Retrieval failed: {exc}") from exc

    @staticmethod
    def _config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}


def sources_from_results(results: Iterable[SearchResult]) -> list[SourceInfo]:
    return [
        SourceInfo(
            kind="document",
            source=str(result.payload.get("source", "")),
            filename=str(result.payload.get("filename", "")),
            chunk_index=int(result.payload.get("chunk_index", 0)),
            score=result.score,
            snippet=_snippet(str(result.payload.get("text", ""))),
        )
        for result in results
    ]


def relevant_results(
    results: Iterable[SearchResult],
    *,
    minimum_score: float,
) -> list[SearchResult]:
    return [
        result
        for result in results
        if result.score is not None and result.score >= minimum_score
    ]


def _answer_update(
    answer: str,
    *,
    sources: Iterable[SourceInfo] = (),
    warnings: list[str] | None = None,
    agent_run_id: str | None = None,
) -> dict[str, Any]:
    update_current_trace(response_preview=answer[:500])
    return {
        "messages": [AIMessage(content=answer)],
        "answer": answer,
        "sources": [source.model_dump(mode="json") for source in sources],
        "warnings": warnings or [],
        "agent_run_id": agent_run_id,
    }


def _conversation_messages(messages: list[BaseMessage], limit: int = 24) -> list[BaseMessage]:
    return list(messages[-limit:])


def _thread_message(message: BaseMessage) -> ThreadMessage:
    role = "assistant"
    if message.type == "human":
        role = "user"
    elif message.type in {"system", "tool"}:
        role = message.type
    return ThreadMessage(
        id=str(message.id) if message.id else None,
        role=role,  # type: ignore[arg-type]
        content=str(message.content),
    )


def _snippet(text: str, limit: int = 240) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."
