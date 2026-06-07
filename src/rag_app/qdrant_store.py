from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from qdrant_client import QdrantClient, models

from .documents import DocumentChunk
from .ids import point_id_for_chunk
from .tracing import timed, trace


@dataclass(frozen=True)
class SearchResult:
    id: str | int
    score: float | None
    payload: dict[str, Any]


class QdrantStore:
    def __init__(
        self,
        *,
        url: str,
        api_key: str,
        collection_name: str,
        vector_size: int,
        score_threshold: float = 0.35,
        timeout: float = 30.0,
    ) -> None:
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.score_threshold = score_threshold
        self.client = QdrantClient(url=url, api_key=api_key, timeout=timeout)

    @trace
    def health(self) -> dict[str, Any]:
        collections = self.client.get_collections()
        collection_names = [collection.name for collection in collections.collections]
        return {
            "ok": True,
            "collection": self.collection_name,
            "collection_exists": self.collection_name in collection_names,
        }

    def collection_exists(self) -> bool:
        if hasattr(self.client, "collection_exists"):
            return bool(self.client.collection_exists(self.collection_name))
        collections = self.client.get_collections()
        return self.collection_name in {
            collection.name for collection in collections.collections
        }

    @trace
    def ensure_collection(self) -> None:
        if self.collection_exists():
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    @trace
    def reset_collection(self) -> None:
        if self.collection_exists():
            self.client.delete_collection(collection_name=self.collection_name)
        self.ensure_collection()

    @trace
    def delete_source(self, source: str) -> None:
        if not self.collection_exists():
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source",
                            match=models.MatchValue(value=source),
                        )
                    ]
                )
            ),
            wait=True,
        )

    @trace
    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        vectors: list[list[float]],
        *,
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return

        ingested_at = datetime.now(UTC).isoformat()
        source_chunk_count = len(chunks)
        points = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            points.append(
                models.PointStruct(
                    id=point_id_for_chunk(chunk),
                    vector=vector,
                    payload={
                        "source": chunk.source,
                        "filename": chunk.filename,
                        "chunk_index": chunk.chunk_index,
                        "content_sha256": chunk.content_sha256,
                        "source_sha256": chunk.source_sha256,
                        "source_chunk_count": source_chunk_count,
                        "embedding_model": embedding_model,
                        "embedding_vector_size": self.vector_size,
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "text": chunk.text,
                        "ingested_at": ingested_at,
                    },
                )
            )

        with timed("qdrant.upsert", collection=self.collection_name, points=len(points)):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

    @trace
    def count_points(self) -> int:
        if not self.collection_exists():
            return 0
        with timed("qdrant.count", collection=self.collection_name):
            result = self.client.count(
                collection_name=self.collection_name,
                exact=True,
            )
        return int(result.count)

    @trace
    def search(self, vector: list[float], limit: int) -> list[SearchResult]:
        if not self.collection_exists():
            return []

        with timed("qdrant.search", collection=self.collection_name, limit=limit):
            if hasattr(self.client, "query_points"):
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=vector,
                    limit=limit,
                    with_payload=True,
                    score_threshold=self.score_threshold,
                )
                raw_points = getattr(response, "points", response)
            else:
                raw_points = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=vector,
                    limit=limit,
                    with_payload=True,
                    score_threshold=self.score_threshold,
                )

        results = []
        for point in raw_points:
            results.append(
                SearchResult(
                    id=point.id,
                    score=getattr(point, "score", None),
                    payload=dict(point.payload or {}),
                )
            )
        return results
