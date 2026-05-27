from __future__ import annotations

import re
import shutil
import zipfile
from dataclasses import dataclass
from collections.abc import Awaitable
from inspect import isawaitable
from pathlib import Path, PurePosixPath
from typing import Protocol

from .documents import SUPPORTED_SUFFIXES

ARCHIVE_SUFFIXES = {".zip"}
UPLOAD_SUFFIXES = SUPPORTED_SUFFIXES | ARCHIVE_SUFFIXES
MAX_ZIP_UNCOMPRESSED_BYTES = 500 * 1024 * 1024


class UploadedFile(Protocol):
    name: str

    def save(self, path: str | Path) -> Awaitable[None] | None: ...


@dataclass(frozen=True)
class UploadResult:
    root: Path
    files: list[Path]
    skipped_entries: list[str]

    def status_text(self, docs_root: Path) -> str:
        root = self.root.relative_to(docs_root)
        if len(self.files) == 1 and self.root == self.files[0]:
            return f"Saved {root}."

        skipped = ""
        if self.skipped_entries:
            skipped = f" Skipped {len(self.skipped_entries)} unsupported entries."
        return f"Extracted {len(self.files)} files to {root}.{skipped}"


async def save_uploaded_document(upload: UploadedFile, *, docs_root: Path) -> UploadResult:
    filename = _safe_filename(upload.name)
    suffix = Path(filename).suffix.lower()
    if suffix not in UPLOAD_SUFFIXES:
        raise ValueError(
            f"Unsupported upload type {suffix!r}. Supported types: "
            f"{', '.join(sorted(UPLOAD_SUFFIXES))}."
        )

    if suffix in ARCHIVE_SUFFIXES:
        return await _save_zip_upload(upload, filename=filename, docs_root=docs_root)

    upload_dir = docs_root / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = _unique_path(upload_dir / filename)
    saved = upload.save(target)
    if isawaitable(saved):
        await saved
    return UploadResult(root=target, files=[target], skipped_entries=[])


async def _save_zip_upload(
    upload: UploadedFile,
    *,
    filename: str,
    docs_root: Path,
) -> UploadResult:
    imports_dir = docs_root / "imports"
    imports_dir.mkdir(parents=True, exist_ok=True)

    archive_stem = Path(filename).stem or "archive"
    extract_root = _unique_path(imports_dir / _safe_filename(archive_stem))
    archive_path = extract_root.with_suffix(".zip")
    archive_path = _unique_path(imports_dir / "archives" / archive_path.name)
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    saved = upload.save(archive_path)
    if isawaitable(saved):
        await saved

    try:
        files, skipped_entries = _extract_supported_zip_entries(
            archive_path,
            extract_root=extract_root,
        )
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Uploaded file {filename!r} is not a valid zip archive.") from exc

    if not files:
        shutil.rmtree(extract_root, ignore_errors=True)
        raise ValueError(
            "Zip archive did not contain any supported document files "
            f"({', '.join(sorted(SUPPORTED_SUFFIXES))})."
        )

    return UploadResult(
        root=extract_root,
        files=files,
        skipped_entries=skipped_entries,
    )


def _extract_supported_zip_entries(
    archive_path: Path,
    *,
    extract_root: Path,
) -> tuple[list[Path], list[str]]:
    files: list[Path] = []
    skipped_entries: list[str] = []

    with zipfile.ZipFile(archive_path) as archive:
        total_size = sum(info.file_size for info in archive.infolist())
        if total_size > MAX_ZIP_UNCOMPRESSED_BYTES:
            raise ValueError(
                "Zip archive is too large after extraction. "
                f"Limit is {MAX_ZIP_UNCOMPRESSED_BYTES // (1024 * 1024)} MB."
            )

        for info in archive.infolist():
            if info.is_dir():
                continue

            relative_path = _safe_zip_member_path(info.filename)
            if relative_path.suffix.lower() not in SUPPORTED_SUFFIXES:
                skipped_entries.append(info.filename)
                continue

            target = _unique_path(extract_root / relative_path)
            _ensure_within_directory(target, extract_root)
            target.parent.mkdir(parents=True, exist_ok=True)

            with archive.open(info) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            files.append(target)

    return files, skipped_entries


def _safe_zip_member_path(name: str) -> Path:
    normalized = name.replace("\\", "/")
    raw_path = PurePosixPath(normalized)
    if raw_path.is_absolute():
        raise ValueError(f"Zip archive contains an absolute path: {name!r}.")

    parts = []
    for part in raw_path.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            raise ValueError(f"Zip archive contains a parent directory path: {name!r}.")
        parts.append(_safe_filename(part))

    if not parts:
        raise ValueError(f"Zip archive contains an invalid path: {name!r}.")
    return Path(*parts)


def _ensure_within_directory(path: Path, root: Path) -> None:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f"Refusing to extract outside document root: {path}.")


def _safe_filename(name: str) -> str:
    filename = Path(name).name.strip()
    filename = filename.replace("\x00", "")
    filename = re.sub(r"[^A-Za-z0-9._ -]+", "_", filename)
    filename = filename.strip(" .")
    if not filename:
        raise ValueError("Uploaded file must have a filename.")
    if filename in {".", ".."}:
        raise ValueError("Invalid upload filename.")
    return filename


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem or "document"
    suffix = path.suffix
    for index in range(1, 10_000):
        candidate = path.with_name(f"{stem}-{index}{suffix}")
        if not candidate.exists():
            return candidate

    raise ValueError(f"Could not choose a unique filename for {path.name!r}.")
