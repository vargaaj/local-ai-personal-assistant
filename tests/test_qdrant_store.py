from types import SimpleNamespace

from rag_app.qdrant_store import QdrantStore


class FakeClient:
    def __init__(self) -> None:
        self.query_kwargs = None

    def collection_exists(self, collection_name: str) -> bool:
        return True

    def query_points(self, **kwargs):
        self.query_kwargs = kwargs
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    id="chunk-1",
                    score=0.9,
                    payload={"filename": "notes.txt"},
                )
            ]
        )


def test_search_passes_score_threshold_to_qdrant():
    store = QdrantStore.__new__(QdrantStore)
    store.collection_name = "local_documents"
    store.vector_size = 3
    store.score_threshold = 0.42
    store.client = FakeClient()

    results = store.search([1.0, 0.0, 0.0], limit=4)

    assert store.client.query_kwargs["score_threshold"] == 0.42
    assert results[0].id == "chunk-1"
