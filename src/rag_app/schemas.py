from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ChatMode = Literal["auto", "general", "documents", "web", "research"]
ResolvedChatMode = Literal["general", "documents", "web", "research"]
SourceKind = Literal["document", "web"]
AgentRunState = Literal[
    "queued",
    "running",
    "awaiting_approval",
    "completed",
    "failed",
    "cancelled",
    "interrupted",
]
ApprovalDecisionKind = Literal["approve", "edit", "reject"]


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
    thread_id: str | None = Field(default=None, min_length=1, max_length=128)
    mode: ChatMode = "auto"


class SourceInfo(BaseModel):
    kind: SourceKind = "document"
    source: str = ""
    filename: str = ""
    chunk_index: int = 0
    score: float | None = None
    snippet: str | None = None
    title: str | None = None
    url: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    thread_id: str | None = None
    resolved_mode: ResolvedChatMode = "documents"
    agent_run_id: str | None = None
    warnings: list[str] = Field(default_factory=list)


class ThreadCreateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=120)


class ThreadSummary(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class ThreadMessage(BaseModel):
    id: str | None = None
    role: Literal["user", "assistant", "system", "tool"]
    content: str


class ThreadDetail(ThreadSummary):
    messages: list[ThreadMessage]


class MemoryFactCreate(BaseModel):
    content: str = Field(..., min_length=1, max_length=2000)


class MemoryFact(BaseModel):
    id: str
    content: str
    created_at: str


class AgentLaunchRequest(BaseModel):
    agent_name: str = "document_research"
    thread_id: str | None = Field(default=None, min_length=1, max_length=128)
    task: str = Field(..., min_length=1, max_length=10_000)
    top_k: int = Field(default=6, ge=1, le=20)


class ApprovalInfo(BaseModel):
    id: str
    run_id: str
    payload: dict[str, Any]
    state: str
    decision: dict[str, Any] | None = None
    created_at: str
    updated_at: str


class AgentRunInfo(BaseModel):
    id: str
    agent_name: str
    thread_id: str
    task: str
    top_k: int
    state: AgentRunState
    result: str | None = None
    error: str | None = None
    sources: list[SourceInfo] = Field(default_factory=list)
    cancel_requested: bool = False
    approval: ApprovalInfo | None = None
    created_at: str
    updated_at: str


class ApprovalDecisionRequest(BaseModel):
    decision: ApprovalDecisionKind
    edited_payload: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: str
    qdrant: dict
    ollama: dict
    mlflow: dict
    config: dict
