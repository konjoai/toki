"""
Attack success rate tracking — SQLite-backed per-attempt log.

Each row in the ``attacks`` table is one test execution: prompt hash,
attack type, optional mutation strategy, outcome, model, and latency.
The tracker exposes aggregate queries that power ``GET /api/attack_stats``:
per-strategy success rates, per-attack-type rates, and a daily trend.

Distinct from :mod:`toki.regression`, which compares aggregate runs
against a stored baseline. This module is the *long-running event log*
that the regression layer (and the dashboard) can read from.
"""

from __future__ import annotations

import enum
import hashlib
import logging
import sqlite3
import time
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public enums / dataclasses
# ---------------------------------------------------------------------------


class AttackResult(enum.Enum):
    """Discrete outcomes — only these three are written to disk."""

    SUCCESS = "success"   # attack landed (the model was BYPASSED)
    FAILURE = "failure"   # attack blocked (the model REFUSED)
    ERROR = "error"       # something went wrong before scoring

    @classmethod
    def parse(cls, value: object) -> "AttackResult":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.strip().lower())
            except ValueError as exc:
                raise ValueError(f"unknown attack result: {value!r}") from exc
        raise ValueError(f"unsupported result type: {type(value).__name__}")


@dataclass(frozen=True)
class AttackRecord:
    """One immutable row from the attacks table."""

    id: int
    timestamp: str
    prompt_hash: str
    attack_type: str
    mutant_strategy: str | None
    result: str
    model: str | None
    latency_ms: float | None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS attacks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    prompt_hash     TEXT NOT NULL,
    attack_type     TEXT NOT NULL,
    mutant_strategy TEXT,
    result          TEXT NOT NULL CHECK (result IN ('success', 'failure', 'error')),
    model           TEXT,
    latency_ms      REAL
);

CREATE INDEX IF NOT EXISTS attacks_by_time   ON attacks(timestamp);
CREATE INDEX IF NOT EXISTS attacks_by_type   ON attacks(attack_type);
CREATE INDEX IF NOT EXISTS attacks_by_model  ON attacks(model);
"""


def _hash_prompt(prompt: str) -> str:
    """Deterministic short hash — privacy-friendly identity of a prompt."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# AttackTracker
# ---------------------------------------------------------------------------


class AttackTracker:
    """SQLite-backed event log of every attack attempt.

    ``db_path`` may be ``":memory:"`` for tests.
    """

    def __init__(self, db_path: str | Path = "toki/db/attack_history.db") -> None:
        self._db_path = ":memory:" if str(db_path) == ":memory:" else Path(db_path)
        if isinstance(self._db_path, Path):
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # In-memory connections must be kept alive — share one connection.
        self._memory_conn: sqlite3.Connection | None = None
        if str(db_path) == ":memory:":
            self._memory_conn = sqlite3.connect(":memory:")
            self._memory_conn.row_factory = sqlite3.Row
            self._memory_conn.executescript(_SCHEMA)
        else:
            with self._open() as cx:
                cx.executescript(_SCHEMA)

    # ------- connection lifecycle -------

    @contextmanager
    def _open(self) -> Iterator[sqlite3.Connection]:
        if self._memory_conn is not None:
            yield self._memory_conn
            return
        cx = sqlite3.connect(self._db_path)
        cx.row_factory = sqlite3.Row
        try:
            yield cx
            cx.commit()
        finally:
            cx.close()

    # ------- write path -------

    def record(
        self,
        attack_type: str,
        prompt: str,
        result: AttackResult | str,
        *,
        strategy: str | None = None,
        model: str | None = None,
        latency_ms: float | None = None,
        timestamp: str | None = None,
    ) -> int:
        """Insert one attack attempt. Returns the new row id."""
        if not attack_type:
            raise ValueError("attack_type required")
        # Empty strings are valid (the edge-case generator uses them as
        # an attack vector). Require str only.
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        res = AttackResult.parse(result)
        ts = timestamp or _utc_now_iso()
        with self._open() as cx:
            cur = cx.execute(
                """INSERT INTO attacks
                   (timestamp, prompt_hash, attack_type, mutant_strategy,
                    result, model, latency_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    ts,
                    _hash_prompt(prompt),
                    attack_type,
                    strategy,
                    res.value,
                    model,
                    float(latency_ms) if latency_ms is not None else None,
                ),
            )
            return int(cur.lastrowid or 0)

    # ------- read path -------

    def _build_filters(
        self,
        days: int | None,
        attack_type: str | None,
        model: str | None,
    ) -> tuple[str, list[object]]:
        clauses: list[str] = []
        params: list[object] = []
        if days is not None and days > 0:
            cutoff = (
                datetime.now(timezone.utc) - timedelta(days=days)
            ).isoformat(timespec="seconds")
            clauses.append("timestamp >= ?")
            params.append(cutoff)
        if attack_type:
            clauses.append("attack_type = ?")
            params.append(attack_type)
        if model:
            clauses.append("model = ?")
            params.append(model)
        sql = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        return sql, params

    def stats(
        self,
        *,
        days: int | None = 7,
        attack_type: str | None = None,
        model: str | None = None,
    ) -> dict:
        """Aggregate stats: totals, by-strategy, by-attack-type, daily trend."""
        where, params = self._build_filters(days, attack_type, model)
        with self._open() as cx:
            total_row = cx.execute(
                f"SELECT COUNT(*) AS n, "
                f"SUM(CASE WHEN result='success' THEN 1 ELSE 0 END) AS s "
                f"FROM attacks{where}",
                params,
            ).fetchone()
            total = int(total_row["n"] or 0)
            successes = int(total_row["s"] or 0)
            success_rate = (successes / total) if total else 0.0

            by_strategy = self._group_rates(
                cx,
                f"SELECT mutant_strategy AS k, COUNT(*) AS n, "
                f"SUM(CASE WHEN result='success' THEN 1 ELSE 0 END) AS s "
                f"FROM attacks{where} GROUP BY mutant_strategy",
                params,
            )
            by_attack_type = self._group_rates(
                cx,
                f"SELECT attack_type AS k, COUNT(*) AS n, "
                f"SUM(CASE WHEN result='success' THEN 1 ELSE 0 END) AS s "
                f"FROM attacks{where} GROUP BY attack_type",
                params,
            )
            trend = self._daily_trend(cx, where, params)

        return {
            "total_attempts": total,
            "successes": successes,
            "success_rate": round(success_rate, 4),
            "by_strategy": by_strategy,
            "by_attack_type": by_attack_type,
            "trend": trend,
            "filters": {
                "days": days,
                "attack_type": attack_type,
                "model": model,
            },
        }

    @staticmethod
    def _group_rates(
        cx: sqlite3.Connection, sql: str, params: list[object]
    ) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for row in cx.execute(sql, params):
            key = row["k"] if row["k"] is not None else "(none)"
            n = int(row["n"] or 0)
            s = int(row["s"] or 0)
            out[key] = {
                "attempts": n,
                "successes": s,
                "success_rate": round(s / n, 4) if n else 0.0,
            }
        return out

    @staticmethod
    def _daily_trend(
        cx: sqlite3.Connection, where: str, params: list[object]
    ) -> list[dict]:
        sql = (
            "SELECT substr(timestamp, 1, 10) AS day, COUNT(*) AS n, "
            "SUM(CASE WHEN result='success' THEN 1 ELSE 0 END) AS s "
            f"FROM attacks{where} GROUP BY day ORDER BY day"
        )
        out: list[dict] = []
        for row in cx.execute(sql, params):
            n = int(row["n"] or 0)
            s = int(row["s"] or 0)
            out.append(
                {
                    "date": row["day"],
                    "attempts": n,
                    "successes": s,
                    "success_rate": round(s / n, 4) if n else 0.0,
                }
            )
        return out

    def classify_categories(
        self,
        *,
        days: int | None = 30,
        intermittent_band: tuple[float, float] = (0.05, 0.95),
        min_attempts: int = 5,
    ) -> dict[str, dict]:
        """Partition attack types into always-blocked / newly-bypassing /
        intermittent buckets — used by the dashboard to surface what's
        worth attention. ``intermittent_band`` is (lo, hi) on success rate.

        - "always_blocked"  success_rate < lo
        - "newly_bypassing" success_rate > hi  (and any success in last 24h)
        - "intermittent"    lo <= success_rate <= hi
        """
        lo, hi = intermittent_band
        if not (0.0 <= lo <= hi <= 1.0):
            raise ValueError("intermittent_band must satisfy 0 <= lo <= hi <= 1")
        where, params = self._build_filters(days, None, None)
        recent_cutoff = (
            datetime.now(timezone.utc) - timedelta(days=1)
        ).isoformat(timespec="seconds")
        with self._open() as cx:
            rows = cx.execute(
                f"SELECT attack_type, COUNT(*) AS n, "
                f"SUM(CASE WHEN result='success' THEN 1 ELSE 0 END) AS s, "
                f"SUM(CASE WHEN result='success' AND timestamp >= ? "
                f"         THEN 1 ELSE 0 END) AS recent_s "
                f"FROM attacks{where} GROUP BY attack_type",
                [recent_cutoff, *params],
            ).fetchall()
        out: dict[str, dict] = {}
        for row in rows:
            n = int(row["n"] or 0)
            s = int(row["s"] or 0)
            recent_s = int(row["recent_s"] or 0)
            rate = (s / n) if n else 0.0
            if n < min_attempts:
                bucket = "insufficient_data"
            elif rate < lo:
                bucket = "always_blocked"
            elif rate > hi and recent_s > 0:
                bucket = "newly_bypassing"
            else:
                bucket = "intermittent"
            out[row["attack_type"]] = {
                "bucket": bucket,
                "attempts": n,
                "success_rate": round(rate, 4),
                "recent_successes": recent_s,
            }
        return out

    def fetch(
        self,
        *,
        days: int | None = None,
        attack_type: str | None = None,
        model: str | None = None,
        result: str | None = None,
        limit: int | None = None,
    ) -> list[AttackRecord]:
        """Yield raw rows matching the given filters. Used by the exporter."""
        where, params = self._build_filters(days, attack_type, model)
        if result:
            res = AttackResult.parse(result).value
            joiner = " AND " if where else " WHERE "
            where = where + joiner + "result = ?"
            params.append(res)
        sql = f"SELECT * FROM attacks{where} ORDER BY timestamp ASC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        with self._open() as cx:
            return [
                AttackRecord(
                    id=int(r["id"]),
                    timestamp=str(r["timestamp"]),
                    prompt_hash=str(r["prompt_hash"]),
                    attack_type=str(r["attack_type"]),
                    mutant_strategy=r["mutant_strategy"],
                    result=str(r["result"]),
                    model=r["model"],
                    latency_ms=r["latency_ms"],
                )
                for r in cx.execute(sql, params).fetchall()
            ]

    def count(
        self,
        *,
        days: int | None = None,
        attack_type: str | None = None,
        model: str | None = None,
        result: str | None = None,
    ) -> int:
        """Count rows matching the same filter shape as :meth:`fetch`."""
        where, params = self._build_filters(days, attack_type, model)
        if result:
            res = AttackResult.parse(result).value
            joiner = " AND " if where else " WHERE "
            where = where + joiner + "result = ?"
            params.append(res)
        with self._open() as cx:
            (n,) = cx.execute(
                f"SELECT COUNT(*) FROM attacks{where}", params
            ).fetchone()
            return int(n or 0)

    # ------- convenience helpers -------

    def record_many(self, rows: Iterable[dict]) -> int:
        """Bulk insert. Each dict must include keys accepted by ``record``."""
        n = 0
        for row in rows:
            self.record(**row)
            n += 1
        return n

    def close(self) -> None:
        if self._memory_conn is not None:
            self._memory_conn.close()
            self._memory_conn = None


def now_ms() -> float:
    """Monotonic ms timestamp — useful when wiring latency into ``record``."""
    return time.perf_counter() * 1000.0
