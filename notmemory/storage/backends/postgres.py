from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from notmemory.core.exceptions import RollbackError
from notmemory.memory.models import (
    AuditTrail,
    Conflict,
    ConflictReport,
    CycleEvent,
    MemoryEntry,
    RollbackResult,
    TrustLevel,
)
from notmemory.storage.backends.base import BaseStorageBackend

_DDL = [
    """CREATE TABLE IF NOT EXISTS memory_entries (
        id              TEXT PRIMARY KEY,
        bank_id         TEXT NOT NULL,
        content         TEXT NOT NULL,
        context         TEXT,
        source          TEXT,
        trust_level     TEXT NOT NULL DEFAULT 'active',
        confidence      REAL NOT NULL DEFAULT 1.0,
        transaction_id  TEXT NOT NULL,
        hash            TEXT NOT NULL,
        parent_hash     TEXT,
        timestamp       TEXT NOT NULL,
        is_tombstoned   BOOLEAN NOT NULL DEFAULT FALSE
    )""",
    "CREATE INDEX IF NOT EXISTS idx_memory_bank ON memory_entries(bank_id)",
    "CREATE INDEX IF NOT EXISTS idx_memory_txn  ON memory_entries(transaction_id)",
    "CREATE INDEX IF NOT EXISTS idx_memory_fts  ON memory_entries USING GIN (to_tsvector('english', content))",
    """CREATE TABLE IF NOT EXISTS cycle_events (
        event_id        TEXT PRIMARY KEY,
        cycle_id        TEXT NOT NULL,
        parent_event_id TEXT,
        event_type      TEXT NOT NULL,
        model           TEXT,
        tokens_in       INTEGER NOT NULL DEFAULT 0,
        tokens_out      INTEGER NOT NULL DEFAULT 0,
        elapsed_ms      REAL NOT NULL DEFAULT 0.0,
        output          TEXT,
        timestamp       TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_events_cycle ON cycle_events(cycle_id)",
]


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def _entry_payload(entry: MemoryEntry) -> str:
    return json.dumps(
        {
            "id": entry.id,
            "bank_id": entry.bank_id,
            "content": entry.content,
            "transaction_id": entry.transaction_id,
            "timestamp": entry.timestamp.isoformat(),
        },
        sort_keys=True,
    )


class PostgresBackend(BaseStorageBackend):
    def __init__(self, db_url: str, *, hash_chaining: bool = True) -> None:
        self._db_url = db_url
        self._hash_chaining = hash_chaining
        self._engine = create_async_engine(db_url, echo=False)
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

    async def initialize(self) -> None:
        async with self._engine.begin() as conn:
            for stmt in _DDL:
                await conn.execute(text(stmt))

    async def close(self) -> None:
        await self._engine.dispose()

    async def _get_last_hash(self, bank_id: str, session: AsyncSession) -> str | None:
        result = await session.execute(
            text(
                "SELECT hash FROM memory_entries "
                "WHERE bank_id = :b AND is_tombstoned = FALSE "
                "ORDER BY timestamp DESC LIMIT 1"
            ),
            {"b": bank_id},
        )
        row = result.fetchone()
        return row[0] if row else None

    async def write_entry(self, entry: MemoryEntry) -> MemoryEntry:
        async with self._session_factory() as session:
            async with session.begin():
                parent_hash = None
                if self._hash_chaining:
                    parent_hash = await self._get_last_hash(entry.bank_id, session)
                    payload = _entry_payload(entry) + (parent_hash or "")
                    entry = entry.model_copy(
                        update={"hash": _sha256(payload), "parent_hash": parent_hash}
                    )

                await session.execute(
                    text("""
                    INSERT INTO memory_entries
                    (id, bank_id, content, context, source, trust_level, confidence,
                     transaction_id, hash, parent_hash, timestamp, is_tombstoned)
                    VALUES
                    (:id, :bank_id, :content, :context, :source, :trust_level, :confidence,
                     :transaction_id, :hash, :parent_hash, :timestamp, FALSE)
                """),
                    {
                        "id": entry.id,
                        "bank_id": entry.bank_id,
                        "content": json.dumps(entry.content),
                        "context": entry.context,
                        "source": entry.source,
                        "trust_level": entry.trust_level,
                        "confidence": entry.confidence,
                        "transaction_id": entry.transaction_id,
                        "hash": entry.hash,
                        "parent_hash": entry.parent_hash,
                        "timestamp": entry.timestamp,
                    },
                )
        return entry

    async def read_entries(
        self,
        bank_id: str,
        *,
        trust_level: TrustLevel | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryEntry]:
        async with self._session_factory() as session:
            params: dict[str, Any] = {"bank_id": bank_id, "limit": limit, "offset": offset}
            where = "bank_id = :bank_id AND is_tombstoned = FALSE"
            if trust_level:
                where += " AND trust_level = :trust_level"
                params["trust_level"] = trust_level
            result = await session.execute(
                text(
                    f"SELECT * FROM memory_entries WHERE {where} "
                    "ORDER BY timestamp DESC LIMIT :limit OFFSET :offset"
                ),
                params,
            )
            return [self._row_to_entry(r) for r in result.mappings()]

    async def keyword_search(
        self, bank_id: str, query: str, *, limit: int = 20
    ) -> list[MemoryEntry]:
        async with self._session_factory() as session:
            result = await session.execute(
                text("""
                    SELECT * FROM memory_entries
                    WHERE bank_id = :bank_id 
                    AND to_tsvector('english', content) @@ websearch_to_tsquery('english', :query)
                    AND is_tombstoned = FALSE 
                    ORDER BY ts_rank(to_tsvector('english', content), websearch_to_tsquery('english', :query)) DESC 
                    LIMIT :limit
                """),
                {"bank_id": bank_id, "query": query, "limit": limit},
            )
            return [self._row_to_entry(r) for r in result.mappings()]

    async def rollback_transaction(self, transaction_id: str) -> RollbackResult:
        async with self._session_factory() as session:
            async with session.begin():
                result = await session.execute(
                    text(
                        "SELECT COUNT(*) FROM memory_entries "
                        "WHERE transaction_id = :txn AND is_tombstoned = FALSE"
                    ),
                    {"txn": transaction_id},
                )
                count = result.scalar() or 0
                if count == 0:
                    raise RollbackError(
                        f"Transaction '{transaction_id}' not found or already rolled back."
                    )
                await session.execute(
                    text("UPDATE memory_entries SET is_tombstoned = TRUE WHERE transaction_id = :txn"),
                    {"txn": transaction_id},
                )
        return RollbackResult(transaction_id=transaction_id, entries_reversed=count, success=True)

    async def write_cycle_event(self, event: CycleEvent) -> CycleEvent:
        async with self._session_factory() as session:
            async with session.begin():
                await session.execute(
                    text("""
                    INSERT INTO cycle_events
                    (event_id, cycle_id, parent_event_id, event_type, model,
                     tokens_in, tokens_out, elapsed_ms, output, timestamp)
                    VALUES
                    (:event_id, :cycle_id, :parent_event_id, :event_type, :model,
                     :tokens_in, :tokens_out, :elapsed_ms, :output, :timestamp)
                """),
                    {
                        "event_id": event.event_id,
                        "cycle_id": event.cycle_id,
                        "parent_event_id": event.parent_event_id,
                        "event_type": event.event_type,
                        "model": event.model,
                        "tokens_in": event.tokens_in,
                        "tokens_out": event.tokens_out,
                        "elapsed_ms": event.elapsed_ms,
                        "output": json.dumps(event.output) if event.output else None,
                        "timestamp": event.timestamp,
                    },
                )
        return event

    async def get_audit_trail(
        self, cycle_id: str, *, event_types: list[str] | None = None
    ) -> AuditTrail:
        async with self._session_factory() as session:
            params: dict[str, Any] = {"cycle_id": cycle_id}
            where = "cycle_id = :cycle_id"
            if event_types:
                placeholders = ", ".join(f":t{i}" for i in range(len(event_types)))
                where += f" AND event_type IN ({placeholders})"
                for i, t in enumerate(event_types):
                    params[f"t{i}"] = t
            result = await session.execute(
                text(f"SELECT * FROM cycle_events WHERE {where} ORDER BY timestamp ASC"),
                params,
            )
            events = [self._row_to_event(r) for r in result.mappings()]

        total_tokens = sum(e.tokens_in + e.tokens_out for e in events)
        return AuditTrail(
            cycle_id=cycle_id,
            events=events,
            total_tokens=total_tokens,
            started_at=events[0].timestamp if events else None,
            ended_at=events[-1].timestamp if events else None,
        )

    async def detect_conflicts(
        self, bank_id: str, *, trust_level: TrustLevel | None = None
    ) -> ConflictReport:
        entries = await self.read_entries(bank_id, trust_level=trust_level, limit=500)
        conflicts: list[Conflict] = []
        seen: dict[str, str] = {}
        for entry in entries:
            key = json.dumps(entry.content, sort_keys=True)
            if key in seen:
                conflicts.append(
                    Conflict(
                        conflict_type="duplicate",
                        entry_a_id=seen[key],
                        entry_b_id=entry.id,
                        description="Identical content stored twice",
                        severity=0.3,
                    )
                )
            else:
                seen[key] = entry.id
        penalty = sum(c.severity * 20 for c in conflicts)
        return ConflictReport(
            bank_id=bank_id,
            health_score=round(max(0.0, 100.0 - penalty), 2),
            conflicts=conflicts,
        )

    async def verify_hash_chain(self, bank_id: str) -> bool:
        async with self._session_factory() as session:
            result = await session.execute(
                text(
                    "SELECT id, content, transaction_id, timestamp, hash, parent_hash "
                    "FROM memory_entries "
                    "WHERE bank_id = :b AND is_tombstoned = FALSE "
                    "ORDER BY timestamp ASC"
                ),
                {"b": bank_id},
            )
            rows = list(result.mappings())
        prev_hash: str | None = None
        for row in rows:
            entry = MemoryEntry(
                id=row["id"],
                bank_id=bank_id,
                content=json.loads(row["content"]),
                transaction_id=row["transaction_id"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
            )
            expected = _sha256(_entry_payload(entry) + (prev_hash or ""))
            if row["hash"] != expected:
                return False
            prev_hash = row["hash"]
        return True

    async def tombstone_entries(self, bank_id: str, *, entry_ids: list[str] | None = None) -> int:
        async with self._session_factory() as session:
            async with session.begin():
                if entry_ids:
                    placeholders = ", ".join(f":id{i}" for i in range(len(entry_ids)))
                    params: dict[str, Any] = {"bank_id": bank_id}
                    for i, eid in enumerate(entry_ids):
                        params[f"id{i}"] = eid
                    result = await session.execute(
                        text(
                            "UPDATE memory_entries SET is_tombstoned = TRUE "
                            f"WHERE bank_id = :bank_id AND id IN ({placeholders})"
                        ),
                        params,
                    )
                else:
                    result = await session.execute(
                        text(
                            "UPDATE memory_entries SET is_tombstoned = TRUE WHERE bank_id = :bank_id"
                        ),
                        {"bank_id": bank_id},
                    )
                return result.rowcount

    @staticmethod
    def _row_to_entry(row: Any) -> MemoryEntry:
        return MemoryEntry(
            id=row["id"],
            bank_id=row["bank_id"],
            content=row["content"],
            context=row["context"],
            source=row["source"],
            trust_level=row["trust_level"],
            confidence=row["confidence"],
            transaction_id=row["transaction_id"],
            hash=row["hash"],
            parent_hash=row["parent_hash"],
            timestamp=(
               row["timestamp"]
               if isinstance(row["timestamp"], datetime)
               else datetime.fromisoformat(row["timestamp"])
            ),
            is_tombstoned=bool(row["is_tombstoned"]),
        )

    @staticmethod
    def _row_to_event(row: Any) -> CycleEvent:
        return CycleEvent(
            event_id=row["event_id"],
            cycle_id=row["cycle_id"],
            parent_event_id=row["parent_event_id"],
            event_type=row["event_type"],
            model=row["model"],
            tokens_in=row["tokens_in"],
            tokens_out=row["tokens_out"],
            elapsed_ms=row["elapsed_ms"],
            output=json.loads(row["output"]) if row["output"] else None,
            timestamp=datetime.fromisoformat(row["timestamp"]),
        )