"""
Persistent SQLite-backed leaderboard for model robustness scores over time.

Phase 10 (T3). Tracks (model_name, suite, pass_rate, robustness_score,
timestamp, notes) tuples across runs so the dashboard can render a
"who's safest right now" view and a per-model history curve.

Design notes
------------
* Pure stdlib — ``sqlite3`` only. No SQLAlchemy, no migration framework.
  Schema is created on first use; later columns can be added with
  idempotent ``ALTER TABLE … ADD COLUMN IF NOT EXISTS`` (sqlite ≥ 3.35).
* ``Leaderboard`` is the public class; it owns one connection per
  instance and serialises writes through the SQLite library's
  built-in locking.
* ``robustness_score`` is in [0, 1]; ``pass_rate`` in [0, 1].  Values
  outside the unit interval are rejected at the boundary because the
  rest of toki guarantees the same range and silent corruption here
  would let a bad seed pollute the table.
* ``timestamp`` defaults to ``datetime.utcnow().isoformat(timespec="seconds")``
  so callers can omit it; passing one explicitly (e.g., for seeding)
  is preserved verbatim.
"""
from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Suites the API/UI know about today.  ``"all"`` is *not* a real suite —
#: it is reserved for the GET filter and never written.
KNOWN_SUITES: tuple[str, ...] = ("adversarial", "paraphrase", "noise")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS leaderboard (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name        TEXT    NOT NULL,
    suite             TEXT    NOT NULL,
    pass_rate         REAL    NOT NULL,
    robustness_score  REAL    NOT NULL,
    timestamp         TEXT    NOT NULL,
    notes             TEXT    NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_leaderboard_suite      ON leaderboard(suite);
CREATE INDEX IF NOT EXISTS idx_leaderboard_model      ON leaderboard(model_name);
CREATE INDEX IF NOT EXISTS idx_leaderboard_timestamp  ON leaderboard(timestamp);
"""


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class LeaderboardEntry:
    """One row in the leaderboard.

    ``id`` is filled in after :meth:`Leaderboard.record`; it is ``None`` for
    in-memory entries that haven't been persisted yet.
    """

    model_name: str
    suite: str
    pass_rate: float
    robustness_score: float
    timestamp: str = field(default="")
    notes: str = ""
    id: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        # Normalise: tolerate Python booleans / ints sliding in.
        self.pass_rate = float(self.pass_rate)
        self.robustness_score = float(self.robustness_score)
        if not _in_unit(self.pass_rate):
            raise ValueError(f"pass_rate must be in [0, 1]; got {self.pass_rate!r}")
        if not _in_unit(self.robustness_score):
            raise ValueError(
                f"robustness_score must be in [0, 1]; got {self.robustness_score!r}"
            )
        if not self.model_name or not isinstance(self.model_name, str):
            raise ValueError("model_name must be a non-empty string")
        if not self.suite or not isinstance(self.suite, str):
            raise ValueError("suite must be a non-empty string")

    def to_dict(self) -> dict:
        return asdict(self)


def _in_unit(x: float) -> bool:
    return 0.0 <= x <= 1.0 and x == x  # rejects NaN as well


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

class Leaderboard:
    """Persistent SQLite-backed leaderboard.

    Usage::

        lb = Leaderboard("leaderboard.db")
        lb.record(LeaderboardEntry("phi-3", "adversarial", 0.91, 0.87))
        top  = lb.top_n("adversarial", n=10)
        hist = lb.history("phi-3")
        diff = lb.compare("phi-3", "qwen-2.5-1.5b")
    """

    def __init__(self, db_path: Union[str, Path] = "leaderboard.db") -> None:
        self.db_path = str(db_path)
        # ``check_same_thread=False`` — the demo HTTP server is threaded;
        # SQLite's library lock still serialises writes safely.
        self._conn = sqlite3.connect(
            self.db_path, check_same_thread=False, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def record(self, entry: LeaderboardEntry) -> LeaderboardEntry:
        """Persist a single entry.  Returns it with ``id`` populated."""
        cur = self._conn.execute(
            "INSERT INTO leaderboard "
            "(model_name, suite, pass_rate, robustness_score, timestamp, notes) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                entry.model_name,
                entry.suite,
                entry.pass_rate,
                entry.robustness_score,
                entry.timestamp,
                entry.notes,
            ),
        )
        entry.id = int(cur.lastrowid) if cur.lastrowid is not None else None
        return entry

    def record_many(self, entries: Iterable[LeaderboardEntry]) -> List[LeaderboardEntry]:
        """Bulk-insert helper — used by seed-loading."""
        return [self.record(e) for e in entries]

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def top_n(
        self, suite: str, n: int = 10, *, include_all: bool = False,
    ) -> List[LeaderboardEntry]:
        """Top ``n`` rows by ``robustness_score`` (descending) for ``suite``.

        If ``suite == "all"`` *or* ``include_all=True``, the suite filter
        is dropped — useful for a global leaderboard view.
        """
        if n <= 0:
            return []
        if suite == "all" or include_all:
            rows = self._conn.execute(
                "SELECT * FROM leaderboard "
                "ORDER BY robustness_score DESC, timestamp DESC LIMIT ?",
                (n,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM leaderboard WHERE suite = ? "
                "ORDER BY robustness_score DESC, timestamp DESC LIMIT ?",
                (suite, n),
            ).fetchall()
        return [_row_to_entry(r) for r in rows]

    def history(self, model_name: str, *, suite: Optional[str] = None) -> List[LeaderboardEntry]:
        """All rows for ``model_name`` ordered chronologically (oldest first)."""
        if suite is None:
            rows = self._conn.execute(
                "SELECT * FROM leaderboard WHERE model_name = ? "
                "ORDER BY timestamp ASC",
                (model_name,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM leaderboard WHERE model_name = ? AND suite = ? "
                "ORDER BY timestamp ASC",
                (model_name, suite),
            ).fetchall()
        return [_row_to_entry(r) for r in rows]

    def compare(self, model_a: str, model_b: str) -> dict:
        """Side-by-side latest-per-suite comparison of two models.

        Returns a dict::

            {
              "model_a":   "phi-3",
              "model_b":   "qwen-2.5-1.5b",
              "by_suite": {
                "adversarial": {
                  "a": {pass_rate, robustness_score, timestamp} | None,
                  "b": {...} | None,
                  "delta": robustness_a - robustness_b   # None if either side missing
                },
                ...
              },
              "winner":    "phi-3" | "qwen-2.5-1.5b" | "tie"
            }

        Winner is determined by mean robustness across suites where both
        models have data; "tie" if scores are within 1e-6 *or* there are
        no overlapping suites.
        """
        a_latest = self._latest_per_suite(model_a)
        b_latest = self._latest_per_suite(model_b)
        suites = sorted(set(a_latest) | set(b_latest))

        by_suite: dict = {}
        a_scores: list = []
        b_scores: list = []
        for s in suites:
            a = a_latest.get(s)
            b = b_latest.get(s)
            delta: Optional[float]
            if a is not None and b is not None:
                delta = a.robustness_score - b.robustness_score
                a_scores.append(a.robustness_score)
                b_scores.append(b.robustness_score)
            else:
                delta = None
            by_suite[s] = {
                "a": _entry_brief(a) if a else None,
                "b": _entry_brief(b) if b else None,
                "delta": delta,
            }

        if not a_scores:
            winner = "tie"
        else:
            mean_a = sum(a_scores) / len(a_scores)
            mean_b = sum(b_scores) / len(b_scores)
            if abs(mean_a - mean_b) < 1e-6:
                winner = "tie"
            else:
                winner = model_a if mean_a > mean_b else model_b

        return {
            "model_a":  model_a,
            "model_b":  model_b,
            "by_suite": by_suite,
            "winner":   winner,
        }

    def all(self) -> List[LeaderboardEntry]:
        """Every row, newest-first.  Convenience for tests + dashboard."""
        rows = self._conn.execute(
            "SELECT * FROM leaderboard ORDER BY timestamp DESC, id DESC"
        ).fetchall()
        return [_row_to_entry(r) for r in rows]

    def count(self) -> int:
        """Total number of rows."""
        return int(self._conn.execute("SELECT COUNT(*) FROM leaderboard").fetchone()[0])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _latest_per_suite(self, model_name: str) -> dict:
        rows = self._conn.execute(
            "SELECT * FROM leaderboard WHERE model_name = ? "
            "ORDER BY timestamp DESC, id DESC",
            (model_name,),
        ).fetchall()
        out: dict = {}
        for r in rows:
            if r["suite"] not in out:
                out[r["suite"]] = _row_to_entry(r)
        return out

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Leaderboard":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_to_entry(row: sqlite3.Row) -> LeaderboardEntry:
    return LeaderboardEntry(
        id=row["id"],
        model_name=row["model_name"],
        suite=row["suite"],
        pass_rate=row["pass_rate"],
        robustness_score=row["robustness_score"],
        timestamp=row["timestamp"],
        notes=row["notes"] or "",
    )


def _entry_brief(e: LeaderboardEntry) -> dict:
    return {
        "pass_rate":        e.pass_rate,
        "robustness_score": e.robustness_score,
        "timestamp":        e.timestamp,
        "notes":            e.notes,
    }


def load_seed(lb: Leaderboard, seed_path: Union[str, Path]) -> int:
    """Load a JSON file of entries into ``lb``.  Returns the count inserted.

    The seed file is expected to be a JSON array of objects matching
    :class:`LeaderboardEntry`'s fields (id is ignored if present).
    """
    import json
    raw = json.loads(Path(seed_path).read_text())
    if not isinstance(raw, list):
        raise ValueError(f"seed file must be a JSON array; got {type(raw).__name__}")
    entries = [
        LeaderboardEntry(
            model_name=item["model_name"],
            suite=item["suite"],
            pass_rate=float(item["pass_rate"]),
            robustness_score=float(item["robustness_score"]),
            timestamp=item.get("timestamp") or "",
            notes=item.get("notes", ""),
        )
        for item in raw
    ]
    lb.record_many(entries)
    return len(entries)
