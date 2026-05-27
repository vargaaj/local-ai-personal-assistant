from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .tracing import trace

SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt"}


@dataclass(frozen=True)
class ParsedDocument:
    path: Path
    text: str
    source_sha256: str


@dataclass(frozen=True)
class DocumentChunk:
    source: str
    filename: str
    chunk_index: int
    text: str
    content_sha256: str
    source_sha256: str


def iter_supported_files(root: Path) -> list[Path]:
    if not root.exists():
        raise ValueError(f"RAG_DOCS_ROOT does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"RAG_DOCS_ROOT must be a directory: {root}")

    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


@trace
def parse_document(path: Path, *, source_sha256: str | None = None) -> ParsedDocument:
    source_sha256 = source_sha256 or file_sha256(path)
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        text = path.read_text(encoding="utf-8", errors="replace")
    elif suffix == ".pdf":
        text = _parse_pdf(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    return ParsedDocument(path=path, text=text.strip(), source_sha256=source_sha256)


def _parse_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    page_texts = []
    for page in reader.pages:
        page_texts.append(page.extract_text() or "")
    return "\n\n".join(page_texts)


@trace
def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@trace
def split_document(
    document: ParsedDocument,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = []
    for index, text in enumerate(splitter.split_text(document.text)):
        clean_text = text.strip()
        if not clean_text:
            continue
        chunks.append(
            DocumentChunk(
                source=str(document.path),
                filename=document.path.name,
                chunk_index=index,
                text=clean_text,
                content_sha256=hashlib.sha256(clean_text.encode("utf-8")).hexdigest(),
                source_sha256=document.source_sha256,
            )
        )
    return chunks
