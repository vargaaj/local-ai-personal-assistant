from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from .schemas import AgentRunInfo, ApprovalInfo, MemoryFact, SourceInfo, ThreadSummary


ACTIVE_RUN_STATES = {"queued", "running", "awaiting_approval"}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def open_sqlite(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=5000")
    return connection


class AppStateStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.connection = open_sqlite(path)
        self._lock = RLock()
        self._setup()
        self.interrupt_incomplete_runs()

    def close(self) -> None:
        with self._lock:
            self.connection.close()

    def _setup(self) -> None:
        with self._lock, self.connection:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS app_threads (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS app_memory_facts (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS app_agent_runs (
                    id TEXT PRIMARY KEY,
                    agent_name TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    task TEXT NOT NULL,
                    top_k INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    result TEXT,
                    error TEXT,
                    sources_json TEXT NOT NULL DEFAULT '[]',
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (thread_id) REFERENCES app_threads(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS app_approvals (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL UNIQUE,
                    payload_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    decision_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES app_agent_runs(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_app_threads_updated
                    ON app_threads(updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_app_agent_runs_updated
                    ON app_agent_runs(updated_at DESC);
                """
            )

    def ensure_thread(self, thread_id: str | None = None, *, title: str | None = None) -> ThreadSummary:
        thread_id = thread_id or uuid4().hex
        now = utc_now()
        clean_title = (title or "New chat").strip()[:120] or "New chat"
        with self._lock, self.connection:
            self.connection.execute(
                """
                INSERT INTO app_threads(id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO NOTHING
                """,
                (thread_id, clean_title, now, now),
            )
            row = self.connection.execute(
                "SELECT * FROM app_threads WHERE id = ?",
                (thread_id,),
            ).fetchone()
        assert row is not None
        return _thread_from_row(row)

    def touch_thread(self, thread_id: str, *, question: str | None = None) -> None:
        now = utc_now()
        with self._lock, self.connection:
            row = self.connection.execute(
                "SELECT title FROM app_threads WHERE id = ?",
                (thread_id,),
            ).fetchone()
            if row is None:
                self.ensure_thread(thread_id)
                row = self.connection.execute(
                    "SELECT title FROM app_threads WHERE id = ?",
                    (thread_id,),
                ).fetchone()
            title = str(row["title"])
            if question and title == "New chat":
                title = " ".join(question.split())[:80] or title
            self.connection.execute(
                "UPDATE app_threads SET title = ?, updated_at = ? WHERE id = ?",
                (title, now, thread_id),
            )

    def list_threads(self) -> list[ThreadSummary]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT * FROM app_threads ORDER BY updated_at DESC"
            ).fetchall()
        return [_thread_from_row(row) for row in rows]

    def get_thread(self, thread_id: str) -> ThreadSummary | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM app_threads WHERE id = ?",
                (thread_id,),
            ).fetchone()
        return _thread_from_row(row) if row else None

    def delete_thread(self, thread_id: str) -> bool:
        with self._lock, self.connection:
            deleted = self.connection.execute(
                "DELETE FROM app_threads WHERE id = ?",
                (thread_id,),
            ).rowcount
            self._delete_langgraph_thread(thread_id)
        return bool(deleted)

    def _delete_langgraph_thread(self, thread_id: str) -> None:
        for table in ("writes", "checkpoint_blobs", "checkpoints"):
            if not self._table_exists(table):
                continue
            self.connection.execute(
                f"DELETE FROM {table} WHERE thread_id = ?",  # noqa: S608
                (thread_id,),
            )

    def _table_exists(self, table: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        return row is not None

    def list_memory_facts(self, *, limit: int | None = None) -> list[MemoryFact]:
        sql = "SELECT * FROM app_memory_facts ORDER BY created_at DESC"
        params: tuple[Any, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        with self._lock:
            rows = self.connection.execute(sql, params).fetchall()
        return [
            MemoryFact(id=str(row["id"]), content=str(row["content"]), created_at=str(row["created_at"]))
            for row in rows
        ]

    def add_memory_fact(self, content: str) -> MemoryFact:
        fact = MemoryFact(id=uuid4().hex, content=content.strip(), created_at=utc_now())
        if not fact.content:
            raise ValueError("Memory fact must not be empty.")
        with self._lock, self.connection:
            self.connection.execute(
                "INSERT INTO app_memory_facts(id, content, created_at) VALUES (?, ?, ?)",
                (fact.id, fact.content, fact.created_at),
            )
        return fact

    def delete_memory_fact(self, fact_id: str) -> bool:
        with self._lock, self.connection:
            deleted = self.connection.execute(
                "DELETE FROM app_memory_facts WHERE id = ?",
                (fact_id,),
            ).rowcount
        return bool(deleted)

    def create_agent_run(
        self,
        *,
        agent_name: str,
        thread_id: str,
        task: str,
        top_k: int,
    ) -> AgentRunInfo:
        self.ensure_thread(thread_id)
        with self._lock, self.connection:
            existing = self.connection.execute(
                """
                SELECT * FROM app_agent_runs
                WHERE agent_name = ? AND thread_id = ? AND task = ?
                  AND state IN ('queued', 'running', 'awaiting_approval')
                ORDER BY created_at DESC LIMIT 1
                """,
                (agent_name, thread_id, task),
            ).fetchone()
            if existing:
                return self._run_from_row(existing)

            now = utc_now()
            run_id = uuid4().hex
            self.connection.execute(
                """
                INSERT INTO app_agent_runs(
                    id, agent_name, thread_id, task, top_k, state, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'queued', ?, ?)
                """,
                (run_id, agent_name, thread_id, task, top_k, now, now),
            )
            row = self.connection.execute(
                "SELECT * FROM app_agent_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
        assert row is not None
        return self._run_from_row(row)

    def get_agent_run(self, run_id: str) -> AgentRunInfo | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM app_agent_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
        return self._run_from_row(row) if row else None

    def list_agent_runs(self, *, thread_id: str | None = None) -> list[AgentRunInfo]:
        sql = "SELECT * FROM app_agent_runs"
        params: tuple[Any, ...] = ()
        if thread_id:
            sql += " WHERE thread_id = ?"
            params = (thread_id,)
        sql += " ORDER BY updated_at DESC"
        with self._lock:
            rows = self.connection.execute(sql, params).fetchall()
        return [self._run_from_row(row) for row in rows]

    def update_agent_run(
        self,
        run_id: str,
        *,
        state: str | None = None,
        result: str | None = None,
        error: str | None = None,
        sources: Iterable[SourceInfo] | None = None,
        cancel_requested: bool | None = None,
    ) -> AgentRunInfo:
        current = self.get_agent_run(run_id)
        if current is None:
            raise KeyError(f"Unknown agent run: {run_id}")
        values = {
            "state": state or current.state,
            "result": result if result is not None else current.result,
            "error": error if error is not None else current.error,
            "sources_json": (
                _sources_json(sources) if sources is not None else _sources_json(current.sources)
            ),
            "cancel_requested": (
                int(cancel_requested)
                if cancel_requested is not None
                else int(current.cancel_requested)
            ),
            "updated_at": utc_now(),
        }
        with self._lock, self.connection:
            self.connection.execute(
                """
                UPDATE app_agent_runs
                SET state = :state, result = :result, error = :error,
                    sources_json = :sources_json, cancel_requested = :cancel_requested,
                    updated_at = :updated_at
                WHERE id = :run_id
                """,
                {**values, "run_id": run_id},
            )
        updated = self.get_agent_run(run_id)
        assert updated is not None
        return updated

    def try_start_agent_run(self, run_id: str) -> AgentRunInfo | None:
        with self._lock, self.connection:
            updated = self.connection.execute(
                """
                UPDATE app_agent_runs
                SET state = 'running', updated_at = ?
                WHERE id = ? AND state = 'queued' AND cancel_requested = 0
                """,
                (utc_now(), run_id),
            ).rowcount
        return self.get_agent_run(run_id) if updated else None

    def complete_agent_run(
        self,
        run_id: str,
        *,
        result: str,
        sources: Iterable[SourceInfo],
    ) -> AgentRunInfo | None:
        with self._lock, self.connection:
            updated = self.connection.execute(
                """
                UPDATE app_agent_runs
                SET state = 'completed', result = ?, sources_json = ?, updated_at = ?
                WHERE id = ? AND state = 'running' AND cancel_requested = 0
                """,
                (result, _sources_json(sources), utc_now(), run_id),
            ).rowcount
        return self.get_agent_run(run_id) if updated else None

    def fail_agent_run(self, run_id: str, *, error: str) -> AgentRunInfo | None:
        with self._lock, self.connection:
            updated = self.connection.execute(
                """
                UPDATE app_agent_runs
                SET state = 'failed', error = ?, updated_at = ?
                WHERE id = ? AND state IN ('queued', 'running') AND cancel_requested = 0
                """,
                (error, utc_now(), run_id),
            ).rowcount
        return self.get_agent_run(run_id) if updated else None

    def request_agent_cancel(self, run_id: str) -> AgentRunInfo:
        with self._lock, self.connection:
            current = self.connection.execute(
                "SELECT id FROM app_agent_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            if current is None:
                raise KeyError(f"Unknown agent run: {run_id}")
            self.connection.execute(
                """
                UPDATE app_agent_runs
                SET state = CASE
                        WHEN state IN ('queued', 'awaiting_approval') THEN 'cancelled'
                        ELSE state
                    END,
                    cancel_requested = CASE
                        WHEN state IN ('queued', 'running', 'awaiting_approval') THEN 1
                        ELSE cancel_requested
                    END,
                    updated_at = ?
                WHERE id = ?
                """,
                (utc_now(), run_id),
            )
        run = self.get_agent_run(run_id)
        assert run is not None
        return run

    def mark_agent_cancelled(self, run_id: str, *, error: str | None = None) -> AgentRunInfo | None:
        with self._lock, self.connection:
            updated = self.connection.execute(
                """
                UPDATE app_agent_runs
                SET state = 'cancelled',
                    cancel_requested = 1,
                    error = COALESCE(?, error),
                    updated_at = ?
                WHERE id = ? AND state IN ('queued', 'running', 'awaiting_approval')
                """,
                (error, utc_now(), run_id),
            ).rowcount
        return self.get_agent_run(run_id) if updated else None

    def cancel_active_agent_runs(
        self,
        *,
        thread_id: str | None = None,
        include_awaiting_approval: bool = True,
    ) -> int:
        states = ["queued", "running"]
        if include_awaiting_approval:
            states.append("awaiting_approval")
        placeholders = ", ".join("?" for _ in states)
        sql = f"""
            UPDATE app_agent_runs
            SET state = 'cancelled', cancel_requested = 1, updated_at = ?
            WHERE state IN ({placeholders})
        """
        params: tuple[Any, ...] = (utc_now(), *states)
        if thread_id is not None:
            sql += " AND thread_id = ?"
            params = (*params, thread_id)
        with self._lock, self.connection:
            return self.connection.execute(sql, params).rowcount

    def delete_agent_run(self, run_id: str) -> bool:
        with self._lock, self.connection:
            deleted = self.connection.execute(
                "DELETE FROM app_agent_runs WHERE id = ?",
                (run_id,),
            ).rowcount
        return bool(deleted)

    def interrupt_incomplete_runs(self) -> int:
        now = utc_now()
        with self._lock, self.connection:
            return self.connection.execute(
                """
                UPDATE app_agent_runs
                SET state = 'interrupted',
                    error = COALESCE(error, 'Application restarted before the run completed.'),
                    updated_at = ?
                WHERE state IN ('queued', 'running')
                """,
                (now,),
            ).rowcount

    def create_approval(self, run_id: str, payload: dict[str, Any]) -> ApprovalInfo | None:
        now = utc_now()
        approval_id = uuid4().hex
        with self._lock, self.connection:
            updated = self.connection.execute(
                """
                UPDATE app_agent_runs
                SET state = 'awaiting_approval', updated_at = ?
                WHERE id = ? AND state = 'running' AND cancel_requested = 0
                """,
                (now, run_id),
            ).rowcount
            if not updated:
                return None
            self.connection.execute(
                """
                INSERT INTO app_approvals(id, run_id, payload_json, state, created_at, updated_at)
                VALUES (?, ?, ?, 'pending', ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    state = 'pending',
                    decision_json = NULL,
                    updated_at = excluded.updated_at
                """,
                (approval_id, run_id, json.dumps(payload), now, now),
            )
        approval = self.get_approval_for_run(run_id)
        assert approval is not None
        return approval

    def decide_approval(
        self,
        run_id: str,
        *,
        decision: str,
        edited_payload: dict[str, Any] | None = None,
    ) -> ApprovalInfo:
        decision_payload = {"decision": decision}
        if edited_payload is not None:
            decision_payload["edited_payload"] = edited_payload
        now = utc_now()
        with self._lock, self.connection:
            row = self.connection.execute(
                """
                SELECT approvals.state AS approval_state,
                       runs.state AS run_state,
                       runs.cancel_requested AS cancel_requested
                FROM app_approvals AS approvals
                JOIN app_agent_runs AS runs ON runs.id = approvals.run_id
                WHERE approvals.run_id = ?
                """,
                (run_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Agent run {run_id} has no pending approval.")
            if row["approval_state"] != "pending":
                raise ValueError(f"Approval for agent run {run_id} has already been decided.")
            if row["run_state"] != "awaiting_approval" or row["cancel_requested"]:
                raise ValueError(f"Agent run {run_id} is no longer awaiting approval.")
            self.connection.execute(
                """
                UPDATE app_approvals
                SET state = ?, decision_json = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (decision, json.dumps(decision_payload), now, run_id),
            )
        decided = self.get_approval_for_run(run_id)
        assert decided is not None
        return decided

    def queue_agent_run_after_approval(self, run_id: str) -> AgentRunInfo | None:
        with self._lock, self.connection:
            updated = self.connection.execute(
                """
                UPDATE app_agent_runs
                SET state = 'queued', updated_at = ?
                WHERE id = ? AND state = 'awaiting_approval' AND cancel_requested = 0
                """,
                (utc_now(), run_id),
            ).rowcount
        return self.get_agent_run(run_id) if updated else None

    def get_approval_for_run(self, run_id: str) -> ApprovalInfo | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM app_approvals WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return ApprovalInfo(
            id=str(row["id"]),
            run_id=str(row["run_id"]),
            payload=dict(json.loads(row["payload_json"])),
            state=str(row["state"]),
            decision=json.loads(row["decision_json"]) if row["decision_json"] else None,
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def _run_from_row(self, row: sqlite3.Row) -> AgentRunInfo:
        run_id = str(row["id"])
        return AgentRunInfo(
            id=run_id,
            agent_name=str(row["agent_name"]),
            thread_id=str(row["thread_id"]),
            task=str(row["task"]),
            top_k=int(row["top_k"]),
            state=str(row["state"]),  # type: ignore[arg-type]
            result=str(row["result"]) if row["result"] is not None else None,
            error=str(row["error"]) if row["error"] is not None else None,
            sources=[SourceInfo.model_validate(source) for source in json.loads(row["sources_json"])],
            cancel_requested=bool(row["cancel_requested"]),
            approval=self.get_approval_for_run(run_id),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )


def _thread_from_row(row: sqlite3.Row) -> ThreadSummary:
    return ThreadSummary(
        id=str(row["id"]),
        title=str(row["title"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _sources_json(sources: Iterable[SourceInfo]) -> str:
    return json.dumps([source.model_dump(mode="json") for source in sources])
