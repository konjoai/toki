"""
Dataset export for fine-tuning safety classifiers.

Two formats: JSONL (``application/x-ndjson``) and CSV. Both stream
record-by-record so the HTTP layer can write directly into the socket
without holding the whole dataset in memory.

The exporter draws from :class:`toki.attack_stats.AttackTracker` rows but
also accepts an injected "prompt resolver" — a callable mapping
``prompt_hash → original prompt text`` — for callers who maintain their
own prompt store. When no resolver is provided, the hash is emitted in
the ``prompt`` field so downstream consumers can still join later.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

from toki.attack_stats import AttackRecord, AttackResult, AttackTracker

logger = logging.getLogger(__name__)


# Map AttackTracker result → the label downstream classifiers expect.
_LABEL_BY_RESULT: dict[str, str] = {
    AttackResult.SUCCESS.value: "BYPASSED",   # attack worked → model unsafe here
    AttackResult.FAILURE.value: "BLOCKED",    # attack failed → model safe here
    AttackResult.ERROR.value: "ERROR",
}


# Column order is fixed so downstream consumers can rely on it.
CSV_COLUMNS = (
    "id",
    "timestamp",
    "prompt",
    "attack_type",
    "strategy",
    "result",
    "label",
    "model",
    "latency_ms",
)


@dataclass(frozen=True)
class ExportFilters:
    """Filter set passed to :class:`DatasetExporter`."""

    attack_type: str | None = None
    result: str | None = None
    model: str | None = None
    days: int | None = None
    limit: int | None = None

    def to_dict(self) -> dict:
        return {
            "attack_type": self.attack_type,
            "result": self.result,
            "model": self.model,
            "days": self.days,
            "limit": self.limit,
        }


PromptResolver = Callable[[str], str]


def _identity_resolver(prompt_hash: str) -> str:
    """Default resolver — emit the hash so the record is still joinable."""
    return prompt_hash


class DatasetExporter:
    """Stream filtered AttackTracker rows as JSONL or CSV."""

    def __init__(
        self,
        tracker: AttackTracker,
        prompt_resolver: PromptResolver | None = None,
    ) -> None:
        self._tracker = tracker
        self._resolver: PromptResolver = prompt_resolver or _identity_resolver

    # ------- record-level shaping -------

    def _record_dict(self, row: AttackRecord) -> dict:
        return {
            "id": row.id,
            "timestamp": row.timestamp,
            "prompt": self._resolver(row.prompt_hash),
            "prompt_hash": row.prompt_hash,
            "attack_type": row.attack_type,
            "strategy": row.mutant_strategy,
            "result": row.result,
            "label": _LABEL_BY_RESULT.get(row.result, row.result.upper()),
            "model": row.model,
            "latency_ms": row.latency_ms,
            "metadata": {
                "source": "toki.attack_stats",
            },
        }

    # ------- iteration -------

    def iter_records(self, filters: ExportFilters) -> Iterator[dict]:
        rows = self._tracker.fetch(
            days=filters.days,
            attack_type=filters.attack_type,
            model=filters.model,
            result=filters.result,
            limit=filters.limit,
        )
        for row in rows:
            yield self._record_dict(row)

    # ------- JSONL (newline-delimited JSON, application/x-ndjson) -------

    def iter_jsonl(self, filters: ExportFilters) -> Iterator[bytes]:
        for rec in self.iter_records(filters):
            yield (json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8")

    def write_jsonl(self, filters: ExportFilters, out: io.IOBase) -> int:
        """Write JSONL to a binary stream. Returns the number of records."""
        n = 0
        for line in self.iter_jsonl(filters):
            out.write(line)
            n += 1
        return n

    # ------- CSV (text/csv with strict quoting) -------

    def iter_csv(self, filters: ExportFilters) -> Iterator[bytes]:
        # Yield the header first, then one CSV-encoded row per record.
        buf = io.StringIO()
        writer = csv.DictWriter(
            buf, fieldnames=CSV_COLUMNS, quoting=csv.QUOTE_NONNUMERIC, extrasaction="ignore"
        )
        writer.writeheader()
        yield buf.getvalue().encode("utf-8")
        for rec in self.iter_records(filters):
            buf.seek(0)
            buf.truncate(0)
            flat = {k: rec.get(k) for k in CSV_COLUMNS}
            writer.writerow(flat)
            yield buf.getvalue().encode("utf-8")

    def write_csv(self, filters: ExportFilters, out: io.IOBase) -> int:
        n = -1  # the header line is not a record
        for line in self.iter_csv(filters):
            out.write(line)
            n += 1
        return max(0, n)

    # ------- stats endpoint helper -------

    def stats(self, filters: ExportFilters) -> dict:
        """Number of records the same filters would emit, without downloading."""
        total = self._tracker.count(
            days=filters.days,
            attack_type=filters.attack_type,
            model=filters.model,
            result=filters.result,
        )
        return {
            "record_count": total,
            "filters": filters.to_dict(),
            "supported_formats": ["jsonl", "csv"],
        }


# ---------------------------------------------------------------------------
# Convenience parsing for query-string flavoured input
# ---------------------------------------------------------------------------


_VALID_FORMATS = {"jsonl", "csv"}


def parse_filters(query: dict) -> tuple[str, ExportFilters]:
    """Translate a flat query dict into (format, ExportFilters).

    Raises :class:`ValueError` on bad input — the HTTP layer maps that to
    a 400 response. All values are stringified before parsing.
    """
    raw_format = str(query.get("format", "jsonl")).lower()
    if raw_format not in _VALID_FORMATS:
        raise ValueError(
            f"unsupported format: {raw_format!r} (allowed: jsonl, csv)"
        )

    def _int(name: str) -> int | None:
        if name not in query or query[name] in (None, ""):
            return None
        try:
            return int(query[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer") from exc

    return raw_format, ExportFilters(
        attack_type=_str_or_none(query.get("attack_type")),
        result=_str_or_none(query.get("result")),
        model=_str_or_none(query.get("model")),
        days=_int("days"),
        limit=_int("limit"),
    )


def _str_or_none(v: object) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def collect(records: Iterable[bytes]) -> bytes:
    """Helper for tests / one-shot callers — join a streamed export."""
    return b"".join(records)
