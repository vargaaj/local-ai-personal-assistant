from pathlib import Path

import pytest

from rag_app import documents
from rag_app.documents import ParsedDocument, iter_supported_files, parse_document, split_document


def test_iter_supported_files_filters_extensions(tmp_path: Path):
    keep = tmp_path / "notes.md"
    keep.write_text("hello", encoding="utf-8")
    skip = tmp_path / "image.png"
    skip.write_text("skip", encoding="utf-8")

    assert iter_supported_files(tmp_path) == [keep]


def test_parse_text_document(tmp_path: Path):
    path = tmp_path / "notes.txt"
    path.write_text("hello\n", encoding="utf-8")

    parsed = parse_document(path)

    assert parsed.text == "hello"
    assert parsed.path == path
    assert parsed.source_sha256


def test_parse_pdf_document_with_fake_reader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class FakePage:
        def __init__(self, text: str) -> None:
            self.text = text

        def extract_text(self) -> str:
            return self.text

    class FakeReader:
        def __init__(self, _path: str) -> None:
            self.pages = [FakePage("page one"), FakePage("page two")]

    monkeypatch.setattr(documents, "PdfReader", FakeReader)
    path = tmp_path / "sample.pdf"
    path.write_bytes(b"%PDF-test")

    parsed = parse_document(path)

    assert parsed.text == "page one\n\npage two"


def test_split_document_adds_metadata(tmp_path: Path):
    path = tmp_path / "notes.md"
    parsed = ParsedDocument(
        path=path,
        text="alpha beta gamma " * 100,
        source_sha256="source-hash",
    )

    chunks = split_document(parsed, chunk_size=80, chunk_overlap=10)

    assert chunks
    assert chunks[0].source == str(path)
    assert chunks[0].filename == "notes.md"
    assert chunks[0].content_sha256
    assert chunks[0].source_sha256 == "source-hash"
