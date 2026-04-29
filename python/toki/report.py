"""HTML and JSON report generation for BenchmarkReport objects."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Optional, Union

from toki.benchmark import BenchmarkReport, BenchmarkStats, StatTestResult


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def to_json(
    report: BenchmarkReport,
    path: Optional[Union[Path, str]] = None,
) -> str:
    """Serialise *report* to a JSON string.

    Args:
        report: The :class:`BenchmarkReport` to serialise.
        path:   Optional file path.  If given the JSON is written to disk and
                the string is still returned.

    Returns:
        A prettily-indented JSON string.
    """
    data = dataclasses.asdict(report)
    text = json.dumps(data, indent=2)
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

_CSS = """
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  background: #0f0f14;
  color: #e2e8f0;
  margin: 0;
  padding: 2rem;
}
h1 { color: #a78bfa; margin-bottom: 0.25rem; }
.subtitle { color: #64748b; font-size: 0.9rem; margin-bottom: 2rem; }
h2 { color: #7dd3fc; border-bottom: 1px solid #1e293b; padding-bottom: 0.4rem; }
table {
  border-collapse: collapse;
  width: 100%;
  margin-bottom: 2rem;
  font-size: 0.9rem;
}
th {
  background: #1e293b;
  color: #94a3b8;
  font-weight: 600;
  padding: 0.6rem 1rem;
  text-align: left;
}
td {
  padding: 0.5rem 1rem;
  border-bottom: 1px solid #1e293b;
}
tr:hover td { background: #1a1a2e; }
.badge {
  display: inline-block;
  padding: 0.2rem 0.75rem;
  border-radius: 9999px;
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.05em;
}
.badge-sig   { background: #166534; color: #bbf7d0; }
.badge-nosig { background: #7f1d1d; color: #fecaca; }
.delta-pos { color: #4ade80; font-weight: 700; }
.delta-neg { color: #f87171; font-weight: 700; }
.delta-zero{ color: #94a3b8; font-weight: 700; }
.delta-block {
  font-size: 2rem;
  font-weight: 800;
  padding: 1.5rem;
  text-align: center;
  background: #1e293b;
  border-radius: 0.75rem;
  margin-bottom: 2rem;
}
.none-note { color: #64748b; font-style: italic; }
"""


def _fmt(value: float, decimals: int = 4) -> str:
    return f"{value:.{decimals}f}"


def _stats_row(label: str, s: BenchmarkStats) -> str:
    return (
        f"<tr>"
        f"<td>{label}</td>"
        f"<td>{s.n}</td>"
        f"<td>{_fmt(s.mean)}</td>"
        f"<td>{_fmt(s.std)}</td>"
        f"<td>{_fmt(s.p50)}</td>"
        f"<td>{_fmt(s.p95)}</td>"
        f"<td>{_fmt(s.p99)}</td>"
        f"<td>{_fmt(s.min)}</td>"
        f"<td>{_fmt(s.max)}</td>"
        f"</tr>"
    )


def _sig_badge(sig: bool) -> str:
    if sig:
        return '<span class="badge badge-sig">SIGNIFICANT</span>'
    return '<span class="badge badge-nosig">NOT SIGNIFICANT</span>'


def _stat_test_rows(label: str, r: StatTestResult) -> str:
    return (
        f"<tr>"
        f"<td>{label}</td>"
        f"<td>{_fmt(r.statistic, 4)}</td>"
        f"<td>{_fmt(r.p_value, 4)}</td>"
        f"<td>{r.alpha}</td>"
        f"<td>{r.n}</td>"
        f"<td>{_sig_badge(r.significant)}</td>"
        f"</tr>"
    )


def _delta_block(delta: Optional[float]) -> str:
    if delta is None:
        return '<div class="delta-block delta-zero">No post-intervention scores</div>'
    css = "delta-pos" if delta >= 0 else "delta-neg"
    sign = "+" if delta >= 0 else ""
    return f'<div class="delta-block"><span class="{css}">Score Delta: {sign}{_fmt(delta, 4)}</span></div>'


def to_html(
    report: BenchmarkReport,
    path: Optional[Union[Path, str]] = None,
) -> str:
    """Render *report* as a self-contained HTML page (inline CSS, no external deps).

    Sections:
    - Header: experiment name + timestamp
    - Score delta block
    - Pre/post statistics table
    - Statistical significance block (t-test + Wilcoxon)
    - Per-category breakdown (if present)

    Args:
        report: The :class:`BenchmarkReport` to render.
        path:   Optional file path.  If given the HTML is written to disk and
                the string is still returned.

    Returns:
        A complete, self-contained HTML document as a string.
    """
    # --- Score table section ---
    score_table_rows = _stats_row("Pre-intervention", report.pre_stats)
    if report.post_stats is not None:
        score_table_rows += _stats_row("Post-intervention", report.post_stats)

    score_section = f"""
<h2>Score Distribution</h2>
<table>
  <thead>
    <tr>
      <th>Phase</th><th>N</th><th>Mean</th><th>Std</th>
      <th>p50</th><th>p95</th><th>p99</th><th>Min</th><th>Max</th>
    </tr>
  </thead>
  <tbody>
    {score_table_rows}
  </tbody>
</table>
"""

    # --- Statistical significance section ---
    if report.t_test is not None or report.wilcoxon is not None:
        sig_rows = ""
        if report.t_test is not None:
            sig_rows += _stat_test_rows("Paired t-test", report.t_test)
        if report.wilcoxon is not None:
            sig_rows += _stat_test_rows("Wilcoxon signed-rank", report.wilcoxon)
        sig_section = f"""
<h2>Statistical Significance</h2>
<table>
  <thead>
    <tr>
      <th>Test</th><th>Statistic</th><th>p-value</th><th>Alpha</th><th>N</th><th>Result</th>
    </tr>
  </thead>
  <tbody>
    {sig_rows}
  </tbody>
</table>
"""
    else:
        sig_section = (
            '<h2>Statistical Significance</h2>'
            '<p class="none-note">No post-intervention scores — significance tests not available.</p>'
        )

    # --- Category breakdown section ---
    if report.category_pre:
        cat_rows = ""
        all_cats = sorted(report.category_pre.keys())
        for cat in all_cats:
            pre_s = report.category_pre[cat]
            cat_rows += (
                f"<tr>"
                f"<td>{cat}</td>"
                f"<td>Pre</td>"
                f"<td>{pre_s.n}</td>"
                f"<td>{_fmt(pre_s.mean)}</td>"
                f"<td>{_fmt(pre_s.std)}</td>"
                f"<td>{_fmt(pre_s.p50)}</td>"
                f"<td>{_fmt(pre_s.p95)}</td>"
                f"<td>{_fmt(pre_s.p99)}</td>"
                f"</tr>"
            )
            if report.category_post and cat in report.category_post:
                post_s = report.category_post[cat]
                cat_rows += (
                    f"<tr>"
                    f"<td>{cat}</td>"
                    f"<td>Post</td>"
                    f"<td>{post_s.n}</td>"
                    f"<td>{_fmt(post_s.mean)}</td>"
                    f"<td>{_fmt(post_s.std)}</td>"
                    f"<td>{_fmt(post_s.p50)}</td>"
                    f"<td>{_fmt(post_s.p95)}</td>"
                    f"<td>{_fmt(post_s.p99)}</td>"
                    f"</tr>"
                )
        cat_section = f"""
<h2>Category Breakdown</h2>
<table>
  <thead>
    <tr>
      <th>Category</th><th>Phase</th><th>N</th><th>Mean</th><th>Std</th>
      <th>p50</th><th>p95</th><th>p99</th>
    </tr>
  </thead>
  <tbody>
    {cat_rows}
  </tbody>
</table>
"""
    else:
        cat_section = ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Toki Benchmark Report — {report.experiment_name}</title>
  <style>{_CSS}</style>
</head>
<body>
  <h1>Toki Benchmark Report</h1>
  <p class="subtitle">Experiment: <strong>{report.experiment_name}</strong> &nbsp;|&nbsp; {report.timestamp}</p>
  {_delta_block(report.score_delta)}
  {score_section}
  {sig_section}
  {cat_section}
</body>
</html>"""

    if path is not None:
        Path(path).write_text(html, encoding="utf-8")
    return html
