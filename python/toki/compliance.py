"""
Compliance certification report.

Maps toki's adversarial-testing coverage onto formal AI-governance control
frameworks and emits a tamper-evident certification report: per-control
status (covered / partial / gap), the toki attack categories that evidence
each control, an overall coverage score, and a SHA-256 manifest over the
evidence so a report cannot be silently edited after signing.

Supported frameworks
--------------------
  * NIST AI RMF 1.0       — MEASURE function, adversarial-robustness controls
  * OWASP Agentic Top 10  — ASI01–ASI10 (2026)
  * ISO/IEC 42001:2023    — AI management-system Annex A controls
  * EU AI Act             — Article 15 (accuracy, robustness, cybersecurity)

Pure-stdlib, fully offline, deterministic. Evidence is any mapping of toki
attack category → test count (e.g. from an ``AdversarialDataset`` or a
``RedTeamResult``); ``compliance_from_dataset`` derives it for you.
"""

from __future__ import annotations

import enum
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

from toki.results import ExperimentResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frameworks
# ---------------------------------------------------------------------------


class Framework(str, enum.Enum):
    """A supported compliance control framework."""

    NIST_AI_RMF = "nist_ai_rmf"
    OWASP_AGENTIC = "owasp_agentic"
    ISO_42001 = "iso_42001"
    EU_AI_ACT = "eu_ai_act"


# Control status values.
COVERED = "covered"
PARTIAL = "partial"
GAP = "gap"


@dataclass(frozen=True)
class Control:
    """One control in a framework, plus the toki categories that evidence it."""

    control_id: str
    title: str
    categories: tuple[str, ...]  # toki attack categories demonstrating this control


# Control catalogs. Each control lists the toki attack categories whose test
# coverage constitutes evidence that the control has been exercised.
_NIST_AI_RMF: tuple[Control, ...] = (
    Control(
        "MEASURE-2.2",
        "Evaluate inputs for robustness (load / malformed)",
        ("boundary", "edge_case"),
    ),
    Control(
        "MEASURE-2.5",
        "Robustness against adversarial inputs",
        ("jailbreak", "injection", "encoding"),
    ),
    Control(
        "MEASURE-2.6",
        "Evaluate for unintended / indirect behavior",
        ("indirect", "multiturn"),
    ),
    Control("MEASURE-2.7", "Evaluate AI for scope containment (agency)", ("agentic",)),
    Control(
        "MEASURE-2.11", "Evaluate across languages / representations", ("encoding",)
    ),
)

_OWASP_AGENTIC: tuple[Control, ...] = (
    Control("ASI01", "Agent authorization & control hijacking", ("agentic",)),
    Control("ASI02", "Prompt injection (direct)", ("jailbreak", "injection")),
    Control("ASI03", "Indirect / cross-domain prompt injection", ("indirect",)),
    Control("ASI04", "Tool misuse & excessive agency", ("agentic",)),
    Control("ASI05", "Memory & context manipulation", ("multiturn", "indirect")),
    Control("ASI06", "Input obfuscation & encoding evasion", ("encoding",)),
    Control("ASI07", "Resource exhaustion / unbounded input", ("boundary",)),
    Control("ASI08", "Edge-case & malformed-input handling", ("edge_case",)),
)

_ISO_42001: tuple[Control, ...] = (
    Control(
        "A.6.2.4",
        "AI system verification & validation",
        ("jailbreak", "injection", "edge_case", "boundary"),
    ),
    Control(
        "A.8.3",
        "Robustness & resilience testing",
        ("encoding", "indirect", "multiturn"),
    ),
    Control("A.9.2", "Responsible-use & misuse evaluation", ("jailbreak", "agentic")),
)

_EU_AI_ACT: tuple[Control, ...] = (
    Control(
        "Art15-Accuracy",
        "Accuracy across representative inputs",
        ("edge_case", "boundary"),
    ),
    Control(
        "Art15-Robustness",
        "Robustness against errors & adversarial use",
        ("jailbreak", "injection", "multiturn"),
    ),
    Control(
        "Art15-Cybersecurity",
        "Resilience to manipulation attempts",
        ("encoding", "indirect", "agentic"),
    ),
)

_CATALOGS: dict[str, tuple[Control, ...]] = {
    Framework.NIST_AI_RMF.value: _NIST_AI_RMF,
    Framework.OWASP_AGENTIC.value: _OWASP_AGENTIC,
    Framework.ISO_42001.value: _ISO_42001,
    Framework.EU_AI_ACT.value: _EU_AI_ACT,
}


def get_catalog(framework: str | Framework) -> tuple[Control, ...]:
    """Return the control catalog for a framework name or enum."""
    key = framework.value if isinstance(framework, Framework) else str(framework)
    if key not in _CATALOGS:
        raise ValueError(f"unknown framework {key!r}; valid: {sorted(_CATALOGS)}")
    return _CATALOGS[key]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ControlStatus:
    """Per-control assessment outcome."""

    control_id: str
    title: str
    status: str  # covered | partial | gap
    evidence_categories: list[str]  # mapped categories that have tests
    missing_categories: list[str]  # mapped categories still untested
    test_count: int  # total tests across the mapped categories


@dataclass
class ComplianceReport:
    """A signed compliance certification over toki's testing coverage."""

    framework: str
    timestamp: str
    toki_version: str
    min_tests: int
    n_controls: int
    n_covered: int
    n_partial: int
    n_gap: int
    coverage_score: float  # (covered + 0.5*partial) / n_controls
    certified: bool  # no gaps remain
    controls: list[ControlStatus]
    manifest_sha256: str  # tamper-evident digest over the evidence

    # -- serialization -------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def to_markdown(self) -> str:
        badge = "✅ CERTIFIED" if self.certified else "⚠️ GAPS PRESENT"
        lines = [
            f"# Compliance Report — {self.framework}",
            "",
            f"**Status:** {badge}  ·  **Coverage:** {self.coverage_score:.0%}  "
            f"·  toki v{self.toki_version}  ·  {self.timestamp}",
            "",
            f"Covered {self.n_covered} · Partial {self.n_partial} · "
            f"Gap {self.n_gap}  (of {self.n_controls} controls, "
            f"min {self.min_tests} test(s)/category)",
            "",
            "| Control | Title | Status | Evidence | Missing | Tests |",
            "|---------|-------|--------|----------|---------|-------|",
        ]
        for c in self.controls:
            lines.append(
                f"| {c.control_id} | {c.title} | {c.status.upper()} | "
                f"{', '.join(c.evidence_categories) or '—'} | "
                f"{', '.join(c.missing_categories) or '—'} | {c.test_count} |"
            )
        lines += ["", f"Manifest SHA-256: `{self.manifest_sha256}`", ""]
        return "\n".join(lines)

    def to_html(self) -> str:
        badge = "CERTIFIED" if self.certified else "GAPS PRESENT"
        rows = "".join(
            f"<tr class='{c.status}'><td>{c.control_id}</td><td>{c.title}</td>"
            f"<td>{c.status.upper()}</td>"
            f"<td>{', '.join(c.evidence_categories) or '—'}</td>"
            f"<td>{', '.join(c.missing_categories) or '—'}</td>"
            f"<td>{c.test_count}</td></tr>"
            for c in self.controls
        )
        return (
            "<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>Compliance — {self.framework}</title><style>"
            "body{background:#0d1117;color:#c9d1d9;font-family:system-ui;margin:2rem}"
            "table{border-collapse:collapse;width:100%}td,th{border:1px solid #30363d;"
            "padding:6px 10px;text-align:left}.covered{color:#3fb950}"
            ".partial{color:#d29922}.gap{color:#f85149}</style></head><body>"
            f"<h1>Compliance Report — {self.framework}</h1>"
            f"<p><b>{badge}</b> · Coverage {self.coverage_score:.0%} · "
            f"toki v{self.toki_version} · {self.timestamp}</p>"
            "<table><tr><th>Control</th><th>Title</th><th>Status</th>"
            "<th>Evidence</th><th>Missing</th><th>Tests</th></tr>"
            f"{rows}</table>"
            f"<p>Manifest SHA-256: <code>{self.manifest_sha256}</code></p>"
            "</body></html>"
        )

    def save(self, base_dir: str = "experiments/compliance") -> tuple[Path, Path]:
        run_dir = Path(base_dir) / f"{self.timestamp}_{self.framework}"
        run_dir.mkdir(parents=True, exist_ok=True)
        json_path = run_dir / "compliance.json"
        html_path = run_dir / "compliance.html"
        json_path.write_text(self.to_json())
        html_path.write_text(self.to_html())
        return json_path, html_path

    @classmethod
    def load(cls, path) -> "ComplianceReport":
        data = json.loads(Path(path).read_text())
        data["controls"] = [ControlStatus(**c) for c in data["controls"]]
        return cls(**data)


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------


def _assess_control(
    control: Control, counts: Mapping[str, int], min_tests: int
) -> ControlStatus:
    present = [c for c in control.categories if counts.get(c, 0) >= min_tests]
    missing = [c for c in control.categories if counts.get(c, 0) < min_tests]
    if not missing:
        status = COVERED
    elif present:
        status = PARTIAL
    else:
        status = GAP
    return ControlStatus(
        control_id=control.control_id,
        title=control.title,
        status=status,
        evidence_categories=present,
        missing_categories=missing,
        test_count=sum(counts.get(c, 0) for c in control.categories),
    )


def _manifest(framework: str, statuses: list[ControlStatus]) -> str:
    """Tamper-evident SHA-256 over the (control, status, test_count) evidence."""
    payload = json.dumps(
        {
            "framework": framework,
            "evidence": [[s.control_id, s.status, s.test_count] for s in statuses],
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _toki_version() -> str:
    # Lazy import: compliance is imported while toki/__init__ is still executing.
    try:
        from toki import __version__

        return __version__
    except ImportError:  # pragma: no cover - defensive
        logger.warning("could not resolve toki.__version__")
        return "unknown"


def assess_compliance(
    framework: str | Framework,
    category_counts: Mapping[str, int],
    min_tests: int = 1,
) -> ComplianceReport:
    """Assess testing coverage against a framework's controls.

    Parameters
    ----------
    framework:
        A :class:`Framework` or its string value.
    category_counts:
        Mapping of toki attack category → number of tests run.
    min_tests:
        Tests a category must have for it to count as evidence (default 1).
    """
    if min_tests < 1:
        raise ValueError("min_tests must be >= 1")
    catalog = get_catalog(framework)
    fw = framework.value if isinstance(framework, Framework) else str(framework)

    statuses = [_assess_control(c, category_counts, min_tests) for c in catalog]
    n_covered, n_partial, n_gap = _tally(statuses)
    n = len(statuses)
    score = (n_covered + 0.5 * n_partial) / n if n else 0.0

    return ComplianceReport(
        framework=fw,
        timestamp=ExperimentResult.make_timestamp(),
        toki_version=_toki_version(),
        min_tests=min_tests,
        n_controls=n,
        n_covered=n_covered,
        n_partial=n_partial,
        n_gap=n_gap,
        coverage_score=score,
        certified=n_gap == 0,
        controls=statuses,
        manifest_sha256=_manifest(fw, statuses),
    )


def _tally(statuses: list[ControlStatus]) -> tuple[int, int, int]:
    """Return (n_covered, n_partial, n_gap) over a list of control statuses."""
    n_covered = sum(1 for s in statuses if s.status == COVERED)
    n_partial = sum(1 for s in statuses if s.status == PARTIAL)
    n_gap = sum(1 for s in statuses if s.status == GAP)
    return n_covered, n_partial, n_gap


def count_categories(prompts: Iterable) -> dict[str, int]:
    """Tally toki attack categories from any iterable of prompts.

    Each item must expose a ``.category`` attribute (e.g.
    :class:`AdversarialPrompt`). Unknown / missing categories are tallied
    under their raw value so nothing is silently dropped.
    """
    counts: dict[str, int] = {}
    for p in prompts:
        cat = getattr(p, "category", None) or "unknown"
        counts[cat] = counts.get(cat, 0) + 1
    return counts


def compliance_from_dataset(
    framework: str | Framework,
    dataset,
    min_tests: int = 1,
    save: bool = False,
) -> ComplianceReport:
    """Assess a dataset (or any iterable of prompts) against a framework."""
    report = assess_compliance(framework, count_categories(dataset), min_tests)
    if save:
        report.save()
    return report
