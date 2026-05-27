from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ManifestEntry:
    source: str
    source_sha256: str
    qdrant_collection: str
    embedding_model: str
    embedding_vector_size: int
    chunk_size: int
    chunk_overlap: int
    chunk_count: int
    updated_at: str


class IngestManifest:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._records = self._load()

    def is_current(
        self,
        *,
        source: str,
        source_sha256: str,
        qdrant_collection: str,
        embedding_model: str,
        embedding_vector_size: int,
        chunk_size: int,
        chunk_overlap: int,
    ) -> bool:
        record = self._records.get(source)
        if record is None:
            return False

        return (
            record.source_sha256 == source_sha256
            and record.qdrant_collection == qdrant_collection
            and record.embedding_model == embedding_model
            and record.embedding_vector_size == embedding_vector_size
            and record.chunk_size == chunk_size
            and record.chunk_overlap == chunk_overlap
            and record.chunk_count > 0
        )

    def update(
        self,
        *,
        source: str,
        source_sha256: str,
        qdrant_collection: str,
        embedding_model: str,
        embedding_vector_size: int,
        chunk_size: int,
        chunk_overlap: int,
        chunk_count: int,
    ) -> None:
        self._records[source] = ManifestEntry(
            source=source,
            source_sha256=source_sha256,
            qdrant_collection=qdrant_collection,
            embedding_model=embedding_model,
            embedding_vector_size=embedding_vector_size,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_count=chunk_count,
            updated_at=datetime.now(UTC).isoformat(),
        )
        self.save()

    def remove(self, source: str) -> None:
        self._records.pop(source, None)
        self.save()

    def clear(self) -> None:
        self._records.clear()
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "records": {
                source: asdict(record)
                for source, record in sorted(self._records.items())
            },
        }
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(self.path)

    def _load(self) -> dict[str, ManifestEntry]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        records: dict[str, ManifestEntry] = {}
        for source, raw_record in dict(payload.get("records") or {}).items():
            record = _manifest_entry_from_json(raw_record)
            if record is not None:
                records[source] = record
        return records


def _manifest_entry_from_json(value: Any) -> ManifestEntry | None:
    try:
        return ManifestEntry(
            source=str(value["source"]),
            source_sha256=str(value["source_sha256"]),
            qdrant_collection=str(value["qdrant_collection"]),
            embedding_model=str(value["embedding_model"]),
            embedding_vector_size=int(value["embedding_vector_size"]),
            chunk_size=int(value["chunk_size"]),
            chunk_overlap=int(value["chunk_overlap"]),
            chunk_count=int(value["chunk_count"]),
            updated_at=str(value["updated_at"]),
        )
    except (KeyError, TypeError, ValueError):
        return None
