from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5

from .documents import DocumentChunk


def point_id_for_chunk(chunk: DocumentChunk) -> str:
    raw = f"{chunk.source}:{chunk.chunk_index}:{chunk.content_sha256}"
    return str(uuid5(NAMESPACE_URL, raw))
