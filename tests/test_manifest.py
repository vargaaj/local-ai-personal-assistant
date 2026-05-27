from pathlib import Path

from rag_app.manifest import IngestManifest


def test_manifest_matches_current_entry(tmp_path: Path):
    manifest = IngestManifest(tmp_path / "manifest.json")

    manifest.update(
        source="/docs/a.txt",
        source_sha256="hash",
        qdrant_collection="local_documents",
        embedding_model="embed-model",
        embedding_vector_size=1024,
        chunk_size=1200,
        chunk_overlap=200,
        chunk_count=2,
    )

    reloaded = IngestManifest(tmp_path / "manifest.json")

    assert reloaded.is_current(
        source="/docs/a.txt",
        source_sha256="hash",
        qdrant_collection="local_documents",
        embedding_model="embed-model",
        embedding_vector_size=1024,
        chunk_size=1200,
        chunk_overlap=200,
    )


def test_manifest_rejects_model_mismatch(tmp_path: Path):
    manifest = IngestManifest(tmp_path / "manifest.json")
    manifest.update(
        source="/docs/a.txt",
        source_sha256="hash",
        qdrant_collection="local_documents",
        embedding_model="old-model",
        embedding_vector_size=1024,
        chunk_size=1200,
        chunk_overlap=200,
        chunk_count=2,
    )

    assert not manifest.is_current(
        source="/docs/a.txt",
        source_sha256="hash",
        qdrant_collection="local_documents",
        embedding_model="embed-model",
        embedding_vector_size=1024,
        chunk_size=1200,
        chunk_overlap=200,
    )


def test_manifest_remove_clears_entry(tmp_path: Path):
    manifest = IngestManifest(tmp_path / "manifest.json")
    manifest.update(
        source="/docs/a.txt",
        source_sha256="hash",
        qdrant_collection="local_documents",
        embedding_model="embed-model",
        embedding_vector_size=1024,
        chunk_size=1200,
        chunk_overlap=200,
        chunk_count=2,
    )

    manifest.remove("/docs/a.txt")

    assert not manifest.is_current(
        source="/docs/a.txt",
        source_sha256="hash",
        qdrant_collection="local_documents",
        embedding_model="embed-model",
        embedding_vector_size=1024,
        chunk_size=1200,
        chunk_overlap=200,
    )
