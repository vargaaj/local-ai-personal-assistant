from __future__ import annotations

from pydantic import BaseModel, Field


class FileError(BaseModel):
    source: str
    error: str


class IngestRequest(BaseModel):
    reset: bool = False


class IngestResponse(BaseModel):
    files_processed: int
    chunks_indexed: int
    skipped_files: list[str]
    parser_errors: list[FileError]


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=6, ge=1, le=20)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)


class SourceInfo(BaseModel):
    source: str
    filename: str
    chunk_index: int
    score: float | None = None
    snippet: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]


class HealthResponse(BaseModel):
    status: str
    qdrant: dict
    ollama: dict
    mlflow: dict
    config: dict
