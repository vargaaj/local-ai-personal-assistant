from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest

from rag_app.uploads import save_uploaded_document


class FakeUpload:
    def __init__(self, name: str, content: bytes = b"hello") -> None:
        self.name = name
        self.content = content

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(self.content)


@pytest.mark.anyio
async def test_save_uploaded_document_sanitizes_name(tmp_path: Path):
    result = await save_uploaded_document(
        FakeUpload("../unsafe name!.txt"),
        docs_root=tmp_path,
    )

    path = result.root
    assert path == tmp_path / "uploads" / "unsafe name_.txt"
    assert result.files == [path]
    assert result.skipped_entries == []
    assert path.read_bytes() == b"hello"


@pytest.mark.anyio
async def test_save_uploaded_document_avoids_overwrite(tmp_path: Path):
    first_result = await save_uploaded_document(
        FakeUpload("notes.md", b"one"),
        docs_root=tmp_path,
    )
    second_result = await save_uploaded_document(
        FakeUpload("notes.md", b"two"),
        docs_root=tmp_path,
    )
    first = first_result.root
    second = second_result.root

    assert first == tmp_path / "uploads" / "notes.md"
    assert second == tmp_path / "uploads" / "notes-1.md"
    assert first.read_bytes() == b"one"
    assert second.read_bytes() == b"two"


@pytest.mark.anyio
async def test_save_uploaded_document_rejects_unsupported_type(tmp_path: Path):
    with pytest.raises(ValueError, match="Unsupported upload type"):
        await save_uploaded_document(FakeUpload("image.png"), docs_root=tmp_path)


@pytest.mark.anyio
async def test_save_uploaded_document_extracts_supported_zip_entries(tmp_path: Path):
    result = await save_uploaded_document(
        FakeUpload("documents.zip", _zip_bytes({
            "notes/readme.md": b"# Notes",
            "plain.txt": b"hello",
            "ignore.png": b"image",
        })),
        docs_root=tmp_path,
    )

    assert result.root == tmp_path / "imports" / "documents"
    assert sorted(path.relative_to(result.root) for path in result.files) == [
        Path("notes/readme.md"),
        Path("plain.txt"),
    ]
    assert result.skipped_entries == ["ignore.png"]
    assert (result.root / "notes" / "readme.md").read_bytes() == b"# Notes"
    assert (result.root / "plain.txt").read_bytes() == b"hello"


@pytest.mark.anyio
async def test_save_uploaded_document_rejects_zip_without_documents(tmp_path: Path):
    with pytest.raises(ValueError, match="did not contain any supported"):
        await save_uploaded_document(
            FakeUpload("images.zip", _zip_bytes({"image.png": b"image"})),
            docs_root=tmp_path,
        )


@pytest.mark.anyio
async def test_save_uploaded_document_rejects_zip_path_traversal(tmp_path: Path):
    with pytest.raises(ValueError, match="parent directory"):
        await save_uploaded_document(
            FakeUpload("bad.zip", _zip_bytes({"../secrets.txt": b"nope"})),
            docs_root=tmp_path,
        )


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return buffer.getvalue()
