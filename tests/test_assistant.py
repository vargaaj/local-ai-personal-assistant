from __future__ import annotations

from pathlib import Path

from rag_app.assistant import ConversationGraph
from rag_app.config import Settings
from rag_app.prompt import PROMPT_TEMPLATES, PromptBundle
from rag_app.qdrant_store import SearchResult
from rag_app.state import AppStateStore


class FakeStore:
    def __init__(self, *, point_count: int = 0, results: list[SearchResult] | None = None):
        self.point_count = point_count
        self.results = results or []

    def count_points(self) -> int:
        return self.point_count

    def search(self, vector: list[float], limit: int):
        return self.results[:limit]


class FakeEmbedder:
    model_name = "fake"
    vector_size = 3

    def embed_query(self, text: str):
        return [1.0, 0.0, 0.0]


class FakeChatClient:
    model = "fake"
    system_prompt_metadata = {}

    def __init__(self, *, route: str = "general", route_error: Exception | None = None):
        self.route = route
        self.route_error = route_error
        self.messages = []

    def structured_output(self, messages, schema):
        if self.route_error:
            raise self.route_error
        return schema(mode=self.route, reason="test")

    def chat_messages(self, messages):
        self.messages.append(messages)
        return "local answer"


class FakeWebSearch:
    def __init__(self, *, configured: bool = True, error: Exception | None = None):
        self.configured = configured
        self.error = error
        self.queries: list[str] = []

    def search(self, query: str):
        self.queries.append(query)
        if self.error:
            raise self.error
        return [{"title": "Example", "url": "https://example.com", "content": "fresh result"}]


def test_general_chat_works_without_indexed_documents(tmp_path: Path):
    graph, state = _graph(tmp_path, store=FakeStore())

    response = graph.chat(question="hello", top_k=4, thread_id=None, mode="general")
    detail = graph.thread_detail(response.thread_id)

    assert response.answer == "local answer"
    assert response.resolved_mode == "general"
    assert [message.role for message in detail.messages] == ["user", "assistant"]
    state.close()
    graph.close()


def test_document_chat_returns_local_sources(tmp_path: Path):
    result = SearchResult(
        id="chunk-1",
        score=0.9,
        payload={"source": "/docs/a.txt", "filename": "a.txt", "chunk_index": 2, "text": "alpha"},
    )
    graph, state = _graph(tmp_path, store=FakeStore(point_count=1, results=[result]))

    response = graph.chat(question="find alpha", top_k=4, thread_id=None, mode="documents")

    assert response.resolved_mode == "documents"
    assert response.sources[0].filename == "a.txt"
    assert response.sources[0].kind == "document"
    state.close()
    graph.close()


def test_document_chat_discards_weak_matches(tmp_path: Path):
    result = SearchResult(
        id="chunk-1",
        score=0.1,
        payload={"source": "/docs/a.txt", "filename": "a.txt", "chunk_index": 2, "text": "alpha"},
    )
    graph, state = _graph(tmp_path, store=FakeStore(point_count=1, results=[result]))

    response = graph.chat(question="unrelated", top_k=4, thread_id=None, mode="documents")

    assert response.answer == "No relevant indexed document chunks were found for that question."
    assert response.sources == []
    state.close()
    graph.close()


def test_web_search_sends_only_current_question_outbound(tmp_path: Path):
    web = FakeWebSearch()
    graph, state = _graph(tmp_path, store=FakeStore(), web=web)
    state.add_memory_fact("PRIVATE SAVED FACT")

    response = graph.chat(question="latest release date", top_k=4, thread_id=None, mode="web")

    assert response.sources[0].url == "https://example.com"
    assert web.queries == ["latest release date"]
    state.close()
    graph.close()


def test_forced_web_search_failure_returns_controlled_message(tmp_path: Path):
    graph, state = _graph(
        tmp_path,
        store=FakeStore(),
        web=FakeWebSearch(error=RuntimeError("provider details")),
    )

    response = graph.chat(question="latest release date", top_k=4, thread_id=None, mode="web")

    assert response.resolved_mode == "web"
    assert response.answer == "Live web search is temporarily unavailable. Try again later or use General mode."
    assert response.warnings == ["Live web search is temporarily unavailable."]
    state.close()
    graph.close()


def test_auto_web_search_failure_falls_back_to_general(tmp_path: Path):
    client = FakeChatClient(route="web")
    graph, state = _graph(
        tmp_path,
        store=FakeStore(),
        web=FakeWebSearch(error=RuntimeError("provider details")),
        client=client,
    )

    response = graph.chat(question="latest release date", top_k=4, thread_id=None, mode="auto")

    assert response.resolved_mode == "general"
    assert response.answer == "local answer"
    assert response.warnings == ["Live web search is temporarily unavailable. Using general chat."]
    state.close()
    graph.close()


def test_auto_router_failure_falls_back_to_general(tmp_path: Path):
    client = FakeChatClient(route_error=RuntimeError("bad structured output"))
    graph, state = _graph(tmp_path, store=FakeStore(), client=client)

    response = graph.chat(question="hello", top_k=4, thread_id=None, mode="auto")

    assert response.resolved_mode == "general"
    assert "Automatic routing failed" in response.warnings[0]
    state.close()
    graph.close()


def test_thread_history_survives_graph_reconstruction(tmp_path: Path):
    graph, state = _graph(tmp_path, store=FakeStore())
    response = graph.chat(question="remember this turn", top_k=4, thread_id=None, mode="general")
    graph.close()
    state.close()

    restored_graph, restored_state = _graph(tmp_path, store=FakeStore())
    detail = restored_graph.thread_detail(response.thread_id)

    assert [message.content for message in detail.messages] == ["remember this turn", "local answer"]
    restored_graph.close()
    restored_state.close()


def _graph(
    tmp_path: Path,
    *,
    store: FakeStore,
    web: FakeWebSearch | None = None,
    client: FakeChatClient | None = None,
):
    settings = Settings(
        _env_file=None,
        qdrant_api_key="secret",
        rag_docs_root=tmp_path,
        rag_state_db_path=tmp_path / "state.sqlite3",
        fastembed_cuda=False,
        fastembed_providers=[],
    )
    state = AppStateStore(settings.rag_state_db_path)
    graph = ConversationGraph(
        settings=settings,
        state_store=state,
        store=store,
        embedder=FakeEmbedder(),
        chat_client=client or FakeChatClient(),
        web_search=web or FakeWebSearch(),
        prompts=PromptBundle(prompts=dict(PROMPT_TEMPLATES), metadata={}),
    )
    return graph, state
