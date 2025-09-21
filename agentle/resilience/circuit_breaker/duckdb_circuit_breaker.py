from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional, override, cast

import duckdb

from agentle.resilience.circuit_breaker.circuit_breaker_protocol import (
    CircuitBreakerProtocol,
)


@dataclass
class DuckDBCircuitBreaker(CircuitBreakerProtocol):
    """
    DuckDB-based circuit breaker with a persistent database file.

    - No external services required; state is stored in a DuckDB file.
    - Safe for multiple processes on the same host (single-writer model).
    - Enforces half-open admission limits atomically to prevent stampedes.
    """

    db_path: str = "circuit_breaker.duckdb"
    failure_threshold: int = 5
    recovery_timeout: float = 300.0
    half_open_max_calls: int = 3
    half_open_success_threshold: int = 2
    exponential_backoff_multiplier: float = 1.5
    max_recovery_timeout: float = 1800.0

    _conn: Optional[duckdb.DuckDBPyConnection] = field(
        default=None, init=False, repr=False
    )
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    def _ensure_db_sync(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect(self.db_path)
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS circuits (
                    circuit_id TEXT PRIMARY KEY,
                    failure_count INTEGER NOT NULL DEFAULT 0,
                    last_failure_time DOUBLE NOT NULL DEFAULT 0,
                    is_open INTEGER NOT NULL DEFAULT 0,
                    is_half_open INTEGER NOT NULL DEFAULT 0,
                    recovery_attempts INTEGER NOT NULL DEFAULT 0,
                    half_open_calls INTEGER NOT NULL DEFAULT 0,
                    half_open_successes INTEGER NOT NULL DEFAULT 0,
                    half_open_permits INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            # Ensure column exists for older DBs
            self._conn.execute(
                "ALTER TABLE circuits ADD COLUMN IF NOT EXISTS is_half_open INTEGER DEFAULT 0"
            )
            self._conn.commit()
        return self._conn

    async def _ensure_db(self) -> duckdb.DuckDBPyConnection:
        # Wrap sync connect/init in a thread to avoid blocking event loop
        async with self._lock:
            return await asyncio.to_thread(self._ensure_db_sync)

    def _calc_timeout(self, attempts: int) -> float:
        timeout = self.recovery_timeout * (
            self.exponential_backoff_multiplier**attempts
        )
        return min(timeout, self.max_recovery_timeout)

    def _upsert_row_sync(
        self, conn: duckdb.DuckDBPyConnection, circuit_id: str
    ) -> None:
        conn.execute(
            """
            INSERT INTO circuits(circuit_id)
            SELECT ?
            WHERE NOT EXISTS (SELECT 1 FROM circuits WHERE circuit_id = ?)
            """,
            [circuit_id, circuit_id],
        )

    async def _upsert_row(
        self, conn: duckdb.DuckDBPyConnection, circuit_id: str
    ) -> None:
        await asyncio.to_thread(self._upsert_row_sync, conn, circuit_id)

    @override
    async def is_open(self, circuit_id: str) -> bool:
        conn = await self._ensure_db()
        # Serialize operations within the process; DuckDB serializes writers across processes
        async with self._lock:
            await self._upsert_row(conn, circuit_id)
            # Begin transaction
            await asyncio.to_thread(conn.execute, "BEGIN")
            try:
                cur = await asyncio.to_thread(
                    conn.execute,
                    "SELECT is_open, is_half_open, last_failure_time, recovery_attempts, half_open_calls, half_open_permits FROM circuits WHERE circuit_id = ?",
                    [circuit_id],
                )
                row_any = cur.fetchone()
                # Typed locals
                is_open: int
                is_half_open: int
                last_failure_time: float
                attempts: int
                _half_calls: int
                _permits: int
                if row_any is None:
                    # Should not happen due to upsert, but keep linters happy and be defensive
                    is_open = 0
                    is_half_open = 0
                    last_failure_time = 0.0
                    attempts = 0
                    _half_calls = 0
                    _permits = 0
                else:
                    row = cast(tuple[int, int, float, int, int, int], row_any)
                    (
                        is_open,
                        is_half_open,
                        last_failure_time,
                        attempts,
                        _half_calls,
                        _permits,
                    ) = row
                now: float = float(time.time())

                if is_open:
                    timeout = self._calc_timeout(int(attempts))
                    if now - float(last_failure_time) > float(timeout):
                        # Transition to half-open
                        await asyncio.to_thread(
                            conn.execute,
                            """
                            UPDATE circuits SET
                                is_open=0,
                                is_half_open=1,
                                half_open_calls=0,
                                half_open_successes=0,
                                half_open_permits=?
                            WHERE circuit_id=?
                            """,
                            [max(0, self.half_open_max_calls), circuit_id],
                        )
                        await asyncio.to_thread(conn.execute, "COMMIT")
                        return False
                    await asyncio.to_thread(conn.execute, "COMMIT")
                    return True

                # Half-open admission: allow at most N concurrent calls using counter
                if is_half_open:
                    await asyncio.to_thread(
                        conn.execute,
                        "UPDATE circuits SET half_open_calls = half_open_calls + 1 WHERE circuit_id = ?",
                        [circuit_id],
                    )
                    cur3 = await asyncio.to_thread(
                        conn.execute,
                        "SELECT half_open_calls FROM circuits WHERE circuit_id=?",
                        [circuit_id],
                    )
                    row3_any = cur3.fetchone()
                    if row3_any is None:
                        current_calls = 0
                    else:
                        row3 = cast(tuple[int], row3_any)
                        current_calls = int(row3[0])
                    if current_calls <= self.half_open_max_calls:
                        await asyncio.to_thread(conn.execute, "COMMIT")
                        return False
                    # Revert increment and block
                    await asyncio.to_thread(
                        conn.execute,
                        "UPDATE circuits SET half_open_calls = half_open_calls - 1 WHERE circuit_id = ?",
                        [circuit_id],
                    )
                    await asyncio.to_thread(conn.execute, "COMMIT")
                    return True

                await asyncio.to_thread(conn.execute, "COMMIT")
                return False
            except Exception:
                await asyncio.to_thread(conn.execute, "ROLLBACK")
                raise

    @override
    async def record_success(self, circuit_id: str) -> None:
        conn = await self._ensure_db()
        async with self._lock:
            await self._upsert_row(conn, circuit_id)
            await asyncio.to_thread(conn.execute, "BEGIN")
            try:
                cur = await asyncio.to_thread(
                    conn.execute,
                    "SELECT is_open, is_half_open, half_open_calls, half_open_successes, half_open_permits FROM circuits WHERE circuit_id=?",
                    [circuit_id],
                )
                row_any = cur.fetchone()
                if row_any is None:
                    # Defensive defaults
                    is_open = 0
                    is_half_open = 0
                    half_calls = 0
                    half_success = 0
                    _permits = 0
                else:
                    row = cast(tuple[int, int, int, int, int], row_any)
                    is_open, is_half_open, half_calls, half_success, _permits = row
                if is_open:
                    await asyncio.to_thread(conn.execute, "COMMIT")
                    return

                if is_half_open:
                    half_calls += 1
                    half_success += 1
                    if (
                        half_success >= self.half_open_success_threshold
                        or half_calls >= self.half_open_max_calls
                    ):
                        if half_success >= self.half_open_success_threshold:
                            await asyncio.to_thread(
                                conn.execute,
                                """
                                UPDATE circuits SET
                                    failure_count=0,
                                    last_failure_time=0,
                                    recovery_attempts=0,
                                    half_open_calls=0,
                                    half_open_successes=0,
                                    half_open_permits=0,
                                    is_open=0,
                                    is_half_open=0
                                WHERE circuit_id=?
                                """,
                                [circuit_id],
                            )
                        else:
                            await asyncio.to_thread(
                                conn.execute,
                                """
                                UPDATE circuits SET
                                    is_open=1,
                                    last_failure_time=?,
                                    recovery_attempts=recovery_attempts+1,
                                    half_open_calls=0,
                                    half_open_successes=0,
                                    half_open_permits=0,
                                    is_half_open=0
                                WHERE circuit_id=?
                                """,
                                [time.time(), circuit_id],
                            )
                    else:
                        await asyncio.to_thread(
                            conn.execute,
                            "UPDATE circuits SET half_open_calls=?, half_open_successes=? WHERE circuit_id=?",
                            [half_calls, half_success, circuit_id],
                        )
                    await asyncio.to_thread(conn.execute, "COMMIT")
                    return

                await asyncio.to_thread(
                    conn.execute,
                    "UPDATE circuits SET failure_count=0, last_failure_time=0, recovery_attempts=0 WHERE circuit_id=?",
                    [circuit_id],
                )
                await asyncio.to_thread(conn.execute, "COMMIT")
            except Exception:
                await asyncio.to_thread(conn.execute, "ROLLBACK")
                raise

    @override
    async def record_failure(self, circuit_id: str) -> None:
        conn = await self._ensure_db()
        async with self._lock:
            await self._upsert_row(conn, circuit_id)
            await asyncio.to_thread(conn.execute, "BEGIN")
            try:
                cur = await asyncio.to_thread(
                    conn.execute,
                    "SELECT is_open, is_half_open, failure_count, half_open_calls, half_open_permits FROM circuits WHERE circuit_id=?",
                    [circuit_id],
                )
                row_any = cur.fetchone()
                if row_any is None:
                    # Defensive defaults
                    _is_open = 0
                    is_half_open = 0
                    failure_count = 0
                    _half_calls = 0
                    _permits = 0
                else:
                    row = cast(tuple[int, int, int, int, int], row_any)
                    _is_open, is_half_open, failure_count, _half_calls, _permits = row
                now: float = float(time.time())
                if is_half_open:
                    await asyncio.to_thread(
                        conn.execute,
                        """
                        UPDATE circuits SET
                            is_open=1,
                            failure_count=failure_count+1,
                            last_failure_time=?,
                            recovery_attempts=recovery_attempts+1,
                            half_open_calls=0,
                            half_open_successes=0,
                            half_open_permits=0,
                            is_half_open=0
                        WHERE circuit_id=?
                        """,
                        [now, circuit_id],
                    )
                    await asyncio.to_thread(conn.execute, "COMMIT")
                    return

                failure_count += 1
                open_it = 1 if failure_count >= self.failure_threshold else 0
                await asyncio.to_thread(
                    conn.execute,
                    """
                    UPDATE circuits SET
                        failure_count=?,
                        last_failure_time=?,
                        is_open=CASE WHEN ?=1 THEN 1 ELSE is_open END
                    WHERE circuit_id=?
                    """,
                    [failure_count, now, open_it, circuit_id],
                )
                await asyncio.to_thread(conn.execute, "COMMIT")
            except Exception:
                await asyncio.to_thread(conn.execute, "ROLLBACK")
                raise

    @override
    async def get_failure_count(self, circuit_id: str) -> int:
        conn = await self._ensure_db()
        async with self._lock:
            await self._upsert_row(conn, circuit_id)
            cur = await asyncio.to_thread(
                conn.execute,
                "SELECT failure_count FROM circuits WHERE circuit_id=?",
                [circuit_id],
            )
            row_any = cur.fetchone()
        if row_any is None:
            return 0
        row = cast(tuple[int], row_any)
        return int(row[0])

    @override
    async def reset_circuit(self, circuit_id: str) -> None:
        conn = await self._ensure_db()
        async with self._lock:
            await asyncio.to_thread(
                conn.execute,
                """
                UPDATE circuits SET
                    failure_count=0,
                    last_failure_time=0,
                    is_open=0,
                    is_half_open=0,
                    recovery_attempts=0,
                    half_open_calls=0,
                    half_open_successes=0,
                    half_open_permits=0
                WHERE circuit_id=?
                """,
                [circuit_id],
            )
            await asyncio.to_thread(conn.commit)

    async def close(self) -> None:
        async with self._lock:
            if self._conn is not None:
                await asyncio.to_thread(self._conn.close)
                self._conn = None
