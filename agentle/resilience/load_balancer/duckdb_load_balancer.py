from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import duckdb  # type: ignore

from agentle.generations.providers.base.generation_provider import GenerationProvider
from agentle.resilience.load_balancer.load_balancer_protocol import (
    LoadBalancerProtocol,
)


@dataclass
class DuckDBProviderQuota:
    provider_id: str
    req_per_min: int | None = None
    prompt_tokens_per_min: int | None = None
    completion_tokens_per_min: int | None = None
    weight: int = 1


class DuckDBLoadBalancer(LoadBalancerProtocol):
    """
    DuckDB-backed load balancer with per-provider minute quotas and simple ranking.

    Notes:
    - Uses a sliding 60s window stored in DuckDB for admission checks.
    - Safe for multi-process single-host scenarios (DuckDB serializes writers).
    - Not a distributed coordinator; for multi-host, prefer Redis/Postgres.
    """

    def __init__(self, db_path: str = "load_balancer.duckdb") -> None:
        self.db_path = db_path
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        self._lock = asyncio.Lock()

    def _ensure_db_sync(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect(self.db_path)
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS quotas (
                    provider_id TEXT,
                    model TEXT,
                    req_per_min INTEGER,
                    prompt_tokens_per_min INTEGER,
                    completion_tokens_per_min INTEGER,
                    weight INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY(provider_id, model)
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    provider_id TEXT,
                    model TEXT,
                    ts DOUBLE
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS token_events (
                    provider_id TEXT,
                    model TEXT,
                    ts DOUBLE,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER
                );
                """
            )
            self._conn.commit()
        return self._conn

    async def _ensure_db(self) -> duckdb.DuckDBPyConnection:
        async with self._lock:
            return await asyncio.to_thread(self._ensure_db_sync)

    async def set_quota(
        self, quota: DuckDBProviderQuota, model: str | None = None
    ) -> None:
        conn = await self._ensure_db()
        async with self._lock:
            await asyncio.to_thread(
                conn.execute,
                """
                INSERT INTO quotas(provider_id, model, req_per_min, prompt_tokens_per_min, completion_tokens_per_min, weight)
                VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(provider_id, model) DO UPDATE SET
                    req_per_min=excluded.req_per_min,
                    prompt_tokens_per_min=excluded.prompt_tokens_per_min,
                    completion_tokens_per_min=excluded.completion_tokens_per_min,
                    weight=excluded.weight
                """,
                [
                    quota.provider_id,
                    model or "__any__",
                    quota.req_per_min,
                    quota.prompt_tokens_per_min,
                    quota.completion_tokens_per_min,
                    quota.weight,
                ],
            )
            await asyncio.to_thread(conn.commit)

    async def rank_providers(
        self,
        providers: Sequence[GenerationProvider],
        *,
        model: str | None = None,
    ) -> Sequence[GenerationProvider]:
        conn = await self._ensure_db()
        window_start = time.time() - 60.0
        async with self._lock:
            # Fetch quotas for all providers at once
            pids = [p.circuit_identity for p in providers]
            if not pids:
                return list(providers)
            placeholders = ",".join(["?"] * len(pids))
            # Prefer model-specific quotas; fall back to provider-level (__any__)
            cur = await asyncio.to_thread(
                conn.execute,
                f"""
                SELECT provider_id, model, req_per_min, prompt_tokens_per_min, completion_tokens_per_min, weight
                FROM quotas
                WHERE provider_id IN ({placeholders}) AND model IN (?, '__any__')
                """,
                [*pids, model or "__any__"],
            )
            qmap: dict[
                Tuple[str, str], tuple[Optional[int], Optional[int], Optional[int], int]
            ] = {}
            for row in cur.fetchall():
                pid, mdl = row[0], row[1]
                rpm = int(row[2]) if row[2] is not None else None
                ptm = int(row[3]) if row[3] is not None else None
                ctm = int(row[4]) if row[4] is not None else None
                weight = int(row[5])
                qmap[(pid, mdl)] = (rpm, ptm, ctm, weight)

            # Recent counts per (provider, model_key)
            cur2 = await asyncio.to_thread(
                conn.execute,
                f"""
                SELECT provider_id, model, COUNT(*)
                FROM events
                WHERE ts >= ? AND provider_id IN ({placeholders}) AND model IN (?, '__any__')
                GROUP BY provider_id, model
                """,
                [window_start, *pids, model or "__any__"],
            )
            counts: dict[Tuple[str, str], int] = {
                (row[0], row[1]): int(row[2]) for row in cur2.fetchall()
            }

            # Token aggregates per (provider, model_key)
            cur3 = await asyncio.to_thread(
                conn.execute,
                f"""
                SELECT provider_id, model, COALESCE(SUM(prompt_tokens),0), COALESCE(SUM(completion_tokens),0)
                FROM token_events
                WHERE ts >= ? AND provider_id IN ({placeholders}) AND model IN (?, '__any__')
                GROUP BY provider_id, model
                """,
                [window_start, *pids, model or "__any__"],
            )
            tokens: dict[Tuple[str, str], tuple[int, int]] = {
                (row[0], row[1]): (int(row[2]), int(row[3])) for row in cur3.fetchall()
            }

            def score(p: GenerationProvider) -> tuple[int, float, int]:
                pid = p.circuit_identity
                mdl_key = model or "__any__"
                # pick model-specific quota first
                q = qmap.get((pid, mdl_key)) or qmap.get((pid, "__any__"))
                if q is None:
                    # unlimited
                    return (1, 1.0, 1)
                rpm, ptm, ctm, weight = q
                headrooms: list[float] = []
                # req/min
                if rpm is not None and rpm > 0:
                    used = counts.get((pid, mdl_key), 0)
                    remaining = max(0, rpm - used)
                    headrooms.append(remaining / float(rpm))
                # tokens
                tok = tokens.get((pid, mdl_key), (0, 0))
                if ptm is not None and ptm > 0:
                    remaining_pt = max(0, ptm - tok[0])
                    headrooms.append(remaining_pt / float(ptm))
                if ctm is not None and ctm > 0:
                    remaining_ct = max(0, ctm - tok[1])
                    headrooms.append(remaining_ct / float(ctm))
                if not headrooms:
                    return (1, 1.0, weight)
                min_headroom = min(headrooms)
                return (1 if min_headroom > 0.0 else 0, min_headroom, weight)

            ordered = sorted(providers, key=lambda p: score(p), reverse=True)
            return ordered

    async def acquire(self, provider_id: str, *, model: str | None = None) -> bool:
        conn = await self._ensure_db()
        now = time.time()
        window_start = now - 60.0
        async with self._lock:
            await asyncio.to_thread(conn.execute, "BEGIN")
            try:
                mdl_key = model or "__any__"
                # Prefer model-specific quota first
                cur = await asyncio.to_thread(
                    conn.execute,
                    """
                    SELECT req_per_min FROM quotas
                    WHERE provider_id=? AND model IN (?, '__any__')
                    ORDER BY CASE WHEN model=? THEN 0 ELSE 1 END
                    LIMIT 1
                    """,
                    [provider_id, mdl_key, mdl_key],
                )
                row = cur.fetchone()
                rpm_val = row[0] if row is not None else None
                rpm = int(rpm_val) if isinstance(rpm_val, (int, float)) else -1
                if rpm >= 0:
                    cur2 = await asyncio.to_thread(
                        conn.execute,
                        "SELECT COUNT(*) FROM events WHERE provider_id=? AND model IN (?, '__any__') AND ts >= ?",
                        [provider_id, mdl_key, window_start],
                    )
                    used_row = cur2.fetchone()
                    used_val = used_row[0] if used_row is not None else None
                    used = int(used_val) if isinstance(used_val, (int, float)) else 0
                    if used >= rpm:
                        await asyncio.to_thread(conn.execute, "ROLLBACK")
                        return False

                await asyncio.to_thread(
                    conn.execute,
                    "INSERT INTO events(provider_id, model, ts) VALUES(?, ?, ?)",
                    [provider_id, mdl_key, now],
                )
                await asyncio.to_thread(conn.execute, "COMMIT")
                return True
            except Exception:
                await asyncio.to_thread(conn.execute, "ROLLBACK")
                raise

    async def record_result(
        self,
        provider_id: str,
        *,
        model: str | None = None,
        success: bool = True,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> None:
        if not prompt_tokens and not completion_tokens:
            return None
        conn = await self._ensure_db()
        async with self._lock:
            await asyncio.to_thread(
                conn.execute,
                "INSERT INTO token_events(provider_id, model, ts, prompt_tokens, completion_tokens) VALUES(?, ?, ?, ?, ?)",
                [
                    provider_id,
                    model or "__any__",
                    time.time(),
                    int(prompt_tokens or 0),
                    int(completion_tokens or 0),
                ],
            )
            await asyncio.to_thread(conn.commit)
        return None
