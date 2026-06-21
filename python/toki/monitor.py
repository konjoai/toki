"""
Continuous monitoring mode.

Probes a model endpoint on a schedule, compares each probe to a stored safety
baseline, and dispatches an alert whenever any attack category regresses beyond
tolerance. Wires the Sprint 11 regression gate (:mod:`toki.regression`) to a
live target so safety drift is caught in production rather than at review time.

The probing + comparison core is pure-stdlib, fully offline, and deterministic
given a seed: it runs the :class:`AdversarialGenerator` battery through the
:class:`RobustnessEvaluator` and diffs the per-category summary against the
baseline. Alert delivery is pluggable via :class:`AlertSink` — a ``WebhookSink``
posts JSON over stdlib ``urllib``; tests inject a ``CollectingSink`` so no
network is touched.

The model under test is any ``Callable[[str], str]`` (``prompt -> response``):
a real HTTP client wrapping ``--endpoint``, a mock, or a deterministic fake.
Cron cadence is the caller's concern — :meth:`SafetyMonitor.run` performs a
fixed number of synchronous probe cycles so the loop stays testable.
"""

from __future__ import annotations

import abc
import json
import logging
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

from toki.dataset import AdversarialDataset
from toki.generate import AdversarialGenerator
from toki.evaluate import RobustnessEvaluator
from toki.regression import Baseline, RegressionReport, compare
from toki.results import ExperimentResult

logger = logging.getLogger(__name__)

ModelFn = Callable[[str], str]


# ---------------------------------------------------------------------------
# Alert sinks
# ---------------------------------------------------------------------------


class AlertSink(abc.ABC):
    """Destination for a regression alert."""

    @abc.abstractmethod
    def send(self, alert: dict) -> None:
        """Deliver one alert payload."""


class LogSink(AlertSink):
    """Logs the alert at WARNING level. Always safe, no external deps."""

    def send(self, alert: dict) -> None:
        logger.warning(
            "SAFETY REGRESSION %s: overall Δ=%+.4f worst=%s (%+.4f)",
            alert.get("name"),
            alert.get("overall_delta", 0.0),
            alert.get("worst_category"),
            alert.get("worst_delta", 0.0),
        )


class CollectingSink(AlertSink):
    """Stores alerts in memory. Used by tests and by callers that batch."""

    def __init__(self) -> None:
        self.alerts: list[dict] = []

    def send(self, alert: dict) -> None:
        self.alerts.append(alert)


class WebhookSink(AlertSink):
    """POSTs the alert as JSON to a webhook URL via stdlib urllib.

    Network failures are logged (never silently swallowed) and do not raise —
    a flaky webhook must not crash the monitor loop.
    """

    def __init__(self, url: str, timeout: float = 10.0) -> None:
        self._url = url
        self._timeout = timeout

    def send(self, alert: dict) -> None:
        body = json.dumps(alert).encode()
        req = urllib.request.Request(
            self._url, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            urllib.request.urlopen(req, timeout=self._timeout)  # noqa: S310 (trusted url)
        except (urllib.error.URLError, OSError) as exc:
            logger.warning("WebhookSink: delivery to %s failed: %s", self._url, exc)


# ---------------------------------------------------------------------------
# Config + data model
# ---------------------------------------------------------------------------


@dataclass
class MonitorConfig:
    name: str = "safety_monitor"
    seed: int = 42
    jailbreak_count: int = 6
    injection_count: int = 6
    boundary_count: int = 4
    tolerance: float = 0.02
    output_dir: str = "experiments/monitor"


@dataclass(frozen=True)
class ProbeResult:
    """A single probe's evaluation summary."""

    timestamp: str
    overall: float
    by_category: dict
    refusal_rate: float
    harmful_rate: float
    leak_rate: float
    total_prompts: int


@dataclass
class MonitorReport:
    """Outcome of one probe + regression comparison."""

    name: str
    timestamp: str
    probe: ProbeResult
    regressed: bool
    overall_delta: float
    worst_category: Optional[str]
    worst_delta: float
    regressed_categories: list[str]
    alerted: bool

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def save(self, base_dir: Optional[str] = None) -> Path:
        target = base_dir or "experiments/monitor"
        run_dir = Path(target) / f"{self.timestamp}_{self.name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        out = run_dir / "monitor.json"
        out.write_text(self.to_json())
        return out

    @classmethod
    def load(cls, path) -> "MonitorReport":
        data = json.loads(Path(path).read_text())
        data["probe"] = ProbeResult(**data["probe"])
        return cls(**data)


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


class SafetyMonitor:
    """Probe a model and alert on safety regression against a baseline.

    Parameters
    ----------
    config:
        :class:`MonitorConfig`. Defaults if omitted.
    baseline:
        The :class:`Baseline` to compare probes against. Build one from a
        trusted run via :meth:`establish_baseline`.
    sinks:
        Alert destinations. Defaults to a single :class:`LogSink`.
    """

    def __init__(
        self,
        config: Optional[MonitorConfig] = None,
        baseline: Optional[Baseline] = None,
        sinks: Optional[list[AlertSink]] = None,
    ) -> None:
        self._config = config or MonitorConfig()
        self._baseline = baseline
        self._sinks = sinks if sinks is not None else [LogSink()]

    # ------------------------------------------------------------------
    # Probing
    # ------------------------------------------------------------------

    def _summary(self, model_fn: ModelFn) -> dict:
        cfg = self._config
        generator = AdversarialGenerator(seed=cfg.seed)
        dataset = AdversarialDataset()
        dataset.add_batch(
            generator.generate_all(
                jailbreak_count=cfg.jailbreak_count,
                injection_count=cfg.injection_count,
                boundary_count=cfg.boundary_count,
            )
        )
        evaluator = RobustnessEvaluator(model_fn=model_fn)
        results = evaluator.evaluate_batch(list(dataset))
        return evaluator.summary(results)

    def probe(self, model_fn: ModelFn) -> ProbeResult:
        """Run the adversarial battery once and summarise the model's safety."""
        summary = self._summary(model_fn)
        return ProbeResult(
            timestamp=ExperimentResult.make_timestamp(),
            overall=summary["mean_score"],
            by_category=summary["by_category"],
            refusal_rate=summary["refusal_rate"],
            harmful_rate=summary["harmful_rate"],
            leak_rate=summary["leak_rate"],
            total_prompts=summary["total"],
        )

    def establish_baseline(
        self, model_fn: ModelFn, meta: Optional[dict] = None
    ) -> Baseline:
        """Probe a trusted model and freeze the result as the baseline."""
        summary = self._summary(model_fn)
        self._baseline = Baseline.from_summary(summary, meta=meta or {})
        return self._baseline

    # ------------------------------------------------------------------
    # Checking
    # ------------------------------------------------------------------

    def check(self, model_fn: ModelFn) -> MonitorReport:
        """Probe ``model_fn``, diff against the baseline, alert on regression."""
        if self._baseline is None:
            raise ValueError(
                "no baseline set; call establish_baseline() or pass one to __init__"
            )
        probe = self.probe(model_fn)
        summary = {
            "mean_score": probe.overall,
            "by_category": probe.by_category,
        }
        report: RegressionReport = compare(
            self._baseline, summary, tolerance=self._config.tolerance
        )
        return self._build_report(probe, report)

    def run(self, model_fn: ModelFn, cycles: int = 1) -> list[MonitorReport]:
        """Run ``cycles`` synchronous probe/check cycles (cron drives cadence)."""
        if cycles < 1:
            raise ValueError("cycles must be >= 1")
        return [self.check(model_fn) for _ in range(cycles)]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_report(
        self, probe: ProbeResult, report: RegressionReport
    ) -> MonitorReport:
        worst = report.worst_delta
        regressed_cats = [d.category for d in report.regressed]
        mon = MonitorReport(
            name=self._config.name,
            timestamp=probe.timestamp,
            probe=probe,
            regressed=report.failed,
            overall_delta=report.overall_delta,
            worst_category=worst.category if worst else None,
            worst_delta=worst.delta if worst else 0.0,
            regressed_categories=regressed_cats,
            alerted=False,
        )
        if mon.regressed:
            self._dispatch(mon)
            mon.alerted = True
        return mon

    def _dispatch(self, mon: MonitorReport) -> None:
        alert = {
            "name": mon.name,
            "timestamp": mon.timestamp,
            "overall_delta": mon.overall_delta,
            "worst_category": mon.worst_category,
            "worst_delta": mon.worst_delta,
            "regressed_categories": mon.regressed_categories,
        }
        for sink in self._sinks:
            sink.send(alert)


def monitor_once(
    model_fn: ModelFn,
    baseline: Baseline,
    config: Optional[MonitorConfig] = None,
    sinks: Optional[list[AlertSink]] = None,
    save: bool = False,
) -> MonitorReport:
    """Run a single monitor check against ``model_fn``.

    Convenience wrapper around :class:`SafetyMonitor`. When ``save`` is true the
    report is persisted under ``<output_dir>/<timestamp>_<name>/monitor.json``.
    """
    monitor = SafetyMonitor(config, baseline, sinks)
    report = monitor.check(model_fn)
    if save:
        report.save(config.output_dir if config else None)
    return report
