"""
Safety regression CI gate.

Two operations:

* ``Baseline.from_summary(...)`` snapshots a current evaluator summary into
  a structured ``Baseline`` that can be saved to disk.
* ``RegressionReport.compare(baseline, current, tolerance=0.02)`` diffs a
  fresh summary against a stored baseline and reports any category that
  *regressed* (got worse) by more than ``tolerance``.

The report carries enough structure to drive a CI gate: ``report.failed``
is true iff any category's drop exceeds tolerance, and the CLI surfaces
this as a non-zero exit code.

No external dependencies — pure stdlib.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Optional


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

@dataclass
class Baseline:
    """A frozen-in-time pass/score snapshot keyed by category.

    ``per_category`` maps category → pass_rate in [0, 1]. ``overall`` is the
    mean across all categories present. ``created`` is an ISO-8601 timestamp.
    ``meta`` is a free-form dict callers can use to record seed, model name,
    test size, git SHA, etc.
    """

    overall: float
    per_category: dict[str, float]
    created: str = ""
    meta: dict = field(default_factory=dict)
    schema: str = "toki.regression.v1"

    def __post_init__(self) -> None:
        if not self.created:
            self.created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    @classmethod
    def from_summary(
        cls,
        summary: Mapping,
        meta: Optional[Mapping] = None,
    ) -> "Baseline":
        """Build a baseline from a :meth:`RobustnessEvaluator.summary` dict.

        Accepts both the canonical key ``"by_category"`` (pass rate per
        category as produced by the real evaluator) and a flat
        ``{category: rate}`` dict for callers building baselines from
        custom shapes.
        """
        if "by_category" in summary:
            per_cat = {str(k): float(v) for k, v in summary["by_category"].items()}
        else:
            per_cat = {str(k): float(v) for k, v in summary.items()
                       if isinstance(v, (int, float)) and 0.0 <= v <= 1.0}
        overall = float(summary.get("mean_score", _mean(per_cat.values())))
        return cls(
            overall=overall,
            per_category=per_cat,
            meta=dict(meta or {}),
        )

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))
        return p

    @classmethod
    def load(cls, path: str | Path) -> "Baseline":
        data = json.loads(Path(path).read_text())
        data.pop("schema", None)
        return cls(**data, schema="toki.regression.v1")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class CategoryDelta:
    category: str
    baseline: float
    current: float
    delta: float           # current - baseline; negative = regression
    regressed: bool        # delta < -tolerance


@dataclass
class RegressionReport:
    """Outcome of comparing a current summary to a baseline."""

    tolerance: float
    overall_delta: float
    regressed: list[CategoryDelta]
    improved: list[CategoryDelta]
    unchanged: list[CategoryDelta]
    missing_from_current: list[str]    # categories the baseline had that current does not
    new_in_current: list[str]          # categories present now but not baselined
    worst_delta: Optional[CategoryDelta] = None

    @property
    def failed(self) -> bool:
        return bool(self.regressed)

    def exit_code(self) -> int:
        return 1 if self.failed else 0

    def as_dict(self) -> dict:
        return {
            "tolerance":         self.tolerance,
            "failed":            self.failed,
            "overall_delta":     self.overall_delta,
            "regressed":         [asdict(d) for d in self.regressed],
            "improved":          [asdict(d) for d in self.improved],
            "unchanged":         [asdict(d) for d in self.unchanged],
            "missing_from_current": list(self.missing_from_current),
            "new_in_current":       list(self.new_in_current),
            "worst_delta":          asdict(self.worst_delta) if self.worst_delta else None,
        }

    def to_markdown(self) -> str:
        lines = []
        lines.append(f"# Safety regression report ({'❌ FAILED' if self.failed else '✅ OK'})")
        lines.append("")
        lines.append(f"Tolerance: **{self.tolerance:.2%}**   Overall Δ: **{self.overall_delta:+.4f}**")
        lines.append("")
        if self.regressed:
            lines.append("## Regressions")
            lines.append("| category | baseline | current | delta |")
            lines.append("|----------|---------:|--------:|------:|")
            for d in self.regressed:
                lines.append(f"| {d.category} | {d.baseline:.4f} | {d.current:.4f} | {d.delta:+.4f} |")
            lines.append("")
        if self.improved:
            lines.append("## Improvements")
            for d in self.improved:
                lines.append(f"- **{d.category}** {d.delta:+.4f}")
            lines.append("")
        if self.missing_from_current:
            lines.append("## Categories missing from current run")
            for c in self.missing_from_current:
                lines.append(f"- {c}")
            lines.append("")
        if self.new_in_current:
            lines.append("## New categories (no baseline)")
            for c in self.new_in_current:
                lines.append(f"- {c}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


def compare(
    baseline: Baseline,
    current_summary: Mapping,
    tolerance: float = 0.02,
) -> RegressionReport:
    """Compare a current summary to a baseline.

    A category is **regressed** iff ``current - baseline < -tolerance``.
    A category is **improved** iff ``current - baseline >  tolerance``.
    Otherwise it is **unchanged**.

    ``tolerance`` must be in ``[0, 1]``.
    """
    if not 0.0 <= tolerance <= 1.0:
        raise ValueError("tolerance must be in [0, 1]")

    current = Baseline.from_summary(current_summary)

    base_keys = set(baseline.per_category)
    cur_keys = set(current.per_category)

    regressed: list[CategoryDelta] = []
    improved:  list[CategoryDelta] = []
    unchanged: list[CategoryDelta] = []
    worst: Optional[CategoryDelta] = None

    for cat in sorted(base_keys & cur_keys):
        b = float(baseline.per_category[cat])
        c = float(current.per_category[cat])
        delta = c - b
        node = CategoryDelta(
            category=cat,
            baseline=b,
            current=c,
            delta=delta,
            regressed=(delta < -tolerance),
        )
        if delta < -tolerance:   regressed.append(node)
        elif delta >  tolerance: improved.append(node)
        else:                    unchanged.append(node)
        if worst is None or node.delta < worst.delta:
            worst = node

    return RegressionReport(
        tolerance=tolerance,
        overall_delta=current.overall - baseline.overall,
        regressed=regressed,
        improved=improved,
        unchanged=unchanged,
        missing_from_current=sorted(base_keys - cur_keys),
        new_in_current=sorted(cur_keys - base_keys),
        worst_delta=worst,
    )


# Public alias to make `from toki.regression import RegressionReport, compare`
# read naturally — but also expose `RegressionReport.compare` as a classmethod
# for fluent use (this is the form documented in PLAN.md).
RegressionReport.compare = staticmethod(compare)  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean(values: Iterable[float]) -> float:
    vs = list(values)
    return sum(vs) / len(vs) if vs else 0.0
