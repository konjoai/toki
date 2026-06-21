"""Tests for toki.monitor — continuous safety-regression monitoring."""

from __future__ import annotations

import json

import pytest

from toki.compare import baseline_mixed, baseline_safe, baseline_unsafe
from toki.monitor import (
    CollectingSink,
    LogSink,
    MonitorConfig,
    MonitorReport,
    ProbeResult,
    SafetyMonitor,
    WebhookSink,
    monitor_once,
)
from toki.regression import Baseline


def _small_cfg(**kw):
    return MonitorConfig(jailbreak_count=4, injection_count=4, boundary_count=2, **kw)


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------


def test_probe_returns_summary_fields():
    mon = SafetyMonitor(_small_cfg())
    probe = mon.probe(baseline_safe)
    assert isinstance(probe, ProbeResult)
    assert 0.0 <= probe.overall <= 1.0
    assert probe.total_prompts > 0
    assert probe.by_category


def test_safe_model_scores_higher_than_unsafe():
    mon = SafetyMonitor(_small_cfg())
    assert mon.probe(baseline_safe).overall > mon.probe(baseline_unsafe).overall


def test_probe_is_deterministic_for_seed():
    a = SafetyMonitor(_small_cfg(seed=7)).probe(baseline_mixed)
    b = SafetyMonitor(_small_cfg(seed=7)).probe(baseline_mixed)
    assert a.overall == b.overall
    assert a.by_category == b.by_category


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


def test_establish_baseline_returns_baseline():
    mon = SafetyMonitor(_small_cfg())
    base = mon.establish_baseline(baseline_safe, meta={"ref": "safe"})
    assert isinstance(base, Baseline)
    assert base.per_category
    assert base.meta["ref"] == "safe"


def test_check_without_baseline_raises():
    mon = SafetyMonitor(_small_cfg())
    with pytest.raises(ValueError):
        mon.check(baseline_safe)


# ---------------------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------------------


def test_no_regression_when_model_unchanged():
    mon = SafetyMonitor(_small_cfg())
    mon.establish_baseline(baseline_safe)
    report = mon.check(baseline_safe)
    assert report.regressed is False
    assert report.alerted is False
    assert abs(report.overall_delta) < 1e-9


def test_regression_detected_when_model_degrades():
    sink = CollectingSink()
    mon = SafetyMonitor(_small_cfg(), sinks=[sink])
    mon.establish_baseline(baseline_safe)  # trusted baseline
    report = mon.check(baseline_unsafe)  # endpoint got much worse
    assert report.regressed is True
    assert report.alerted is True
    assert report.overall_delta < 0
    assert report.regressed_categories
    assert len(sink.alerts) == 1


def test_alert_payload_has_expected_fields():
    sink = CollectingSink()
    mon = SafetyMonitor(_small_cfg(), sinks=[sink])
    mon.establish_baseline(baseline_safe)
    mon.check(baseline_unsafe)
    alert = sink.alerts[0]
    assert {
        "name",
        "timestamp",
        "overall_delta",
        "worst_category",
        "worst_delta",
        "regressed_categories",
    } <= set(alert)


def test_tolerance_suppresses_small_drift():
    # a tiny degradation under tolerance should not alert
    sink = CollectingSink()
    mon = SafetyMonitor(_small_cfg(tolerance=1.0), sinks=[sink])
    mon.establish_baseline(baseline_safe)
    report = mon.check(baseline_unsafe)
    assert report.regressed is False
    assert sink.alerts == []


def test_run_multiple_cycles():
    sink = CollectingSink()
    mon = SafetyMonitor(_small_cfg(), sinks=[sink])
    mon.establish_baseline(baseline_safe)
    reports = mon.run(baseline_unsafe, cycles=3)
    assert len(reports) == 3
    assert all(r.regressed for r in reports)
    assert len(sink.alerts) == 3


def test_run_zero_cycles_raises():
    mon = SafetyMonitor(_small_cfg())
    mon.establish_baseline(baseline_safe)
    with pytest.raises(ValueError):
        mon.run(baseline_safe, cycles=0)


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------


def test_default_sink_is_log_sink():
    mon = SafetyMonitor(_small_cfg())
    assert any(isinstance(s, LogSink) for s in mon._sinks)


def test_multiple_sinks_all_receive_alert():
    s1, s2 = CollectingSink(), CollectingSink()
    mon = SafetyMonitor(_small_cfg(), sinks=[s1, s2])
    mon.establish_baseline(baseline_safe)
    mon.check(baseline_unsafe)
    assert len(s1.alerts) == 1 and len(s2.alerts) == 1


def test_log_sink_emits_warning(caplog):
    import logging

    mon = SafetyMonitor(_small_cfg(), sinks=[LogSink()])
    mon.establish_baseline(baseline_safe)
    with caplog.at_level(logging.WARNING):
        mon.check(baseline_unsafe)
    assert any("SAFETY REGRESSION" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Convenience + persistence
# ---------------------------------------------------------------------------


def test_monitor_once_with_explicit_baseline():
    ref = SafetyMonitor(_small_cfg()).establish_baseline(baseline_safe)
    sink = CollectingSink()
    report = monitor_once(baseline_unsafe, ref, _small_cfg(), sinks=[sink])
    assert report.regressed is True
    assert len(sink.alerts) == 1


def test_monitor_once_save_persists(tmp_path):
    ref = SafetyMonitor(_small_cfg()).establish_baseline(baseline_safe)
    cfg = _small_cfg(output_dir=str(tmp_path))
    report = monitor_once(
        baseline_unsafe, ref, cfg, sinks=[CollectingSink()], save=True
    )
    saved = tmp_path / f"{report.timestamp}_{report.name}" / "monitor.json"
    assert saved.exists()


def test_webhook_sink_unreachable_logs_and_does_not_raise(caplog):
    import logging

    # Port 1 on localhost refuses connections — exercises the failure path
    # offline and deterministically; the monitor must not crash.
    sink = WebhookSink("http://127.0.0.1:1/alert", timeout=0.5)
    mon = SafetyMonitor(_small_cfg(), sinks=[sink])
    mon.establish_baseline(baseline_safe)
    with caplog.at_level(logging.WARNING):
        report = mon.check(baseline_unsafe)
    assert report.alerted is True
    assert any("WebhookSink" in r.message for r in caplog.records)


def test_save_and_load_roundtrip(tmp_path):
    mon = SafetyMonitor(_small_cfg(output_dir=str(tmp_path)))
    mon.establish_baseline(baseline_safe)
    report = mon.check(baseline_unsafe)
    out = report.save(str(tmp_path))
    assert out.exists()

    loaded = MonitorReport.load(out)
    assert loaded.name == report.name
    assert loaded.regressed == report.regressed
    assert isinstance(loaded.probe, ProbeResult)
    assert loaded.probe == report.probe


def test_to_json_valid():
    mon = SafetyMonitor(_small_cfg())
    mon.establish_baseline(baseline_safe)
    data = json.loads(mon.check(baseline_safe).to_json())
    assert data["name"] == "safety_monitor"
    assert "probe" in data
