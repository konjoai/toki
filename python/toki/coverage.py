"""
Coverage map + blind-spot dashboard.

Answers the question "are we done testing?" by accounting for *what's covered*
across four axes: attack **category**, **severity**, **language**, **encoding**.

A blind spot is any axis bucket whose share falls below a configurable
``blind_threshold`` (default 5%). The returned :class:`CoverageMap` also
contains pre-computed radar-polygon coordinates so the demo UI can render an
SVG spider chart without doing any math.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Iterable, Mapping, Optional

from toki.dataset import AdversarialDataset
from toki.generate import AdversarialPrompt


# ---------------------------------------------------------------------------
# Axis definitions
# ---------------------------------------------------------------------------

CATEGORY_AXIS: tuple[str, ...] = (
    "jailbreak", "injection", "edge_case", "boundary",
    "encoding", "indirect", "agentic", "multiturn", "multiagent",
)
SEVERITY_AXIS: tuple[str, ...] = ("low", "medium", "high", "critical")
LANGUAGE_AXIS: tuple[str, ...] = ("en", "es", "fr", "de", "other")
ENCODING_AXIS: tuple[str, ...] = ("plain", "base64", "rot13", "unicode_zwsp", "chat_template")

# Default severity inferred from category when an attack doesn't carry a
# `severity` attribute of its own. These are conservative — a jailbreak
# attempt is "high" by default; boundary spam is "low".
_DEFAULT_SEVERITY: dict[str, str] = {
    "jailbreak":  "high",
    "injection":  "high",
    "edge_case":  "medium",
    "boundary":   "low",
    "encoding":   "high",
    "indirect":   "critical",
    "agentic":    "critical",
    "multiturn":  "critical",
    "multiagent": "critical",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CoverageMap:
    """Per-axis counts + computed gaps + radar coordinates.

    ``axes`` maps axis name → ordered list of bucket counts (same order as the
    constants above). ``shares`` maps the same buckets to fractions in [0, 1].
    ``blind_spots`` is a flat list of ``"axis.bucket"`` strings whose share
    falls strictly below ``blind_threshold``.

    ``radar_points`` is a list of ``{"axis","bucket","share","x","y"}``
    dicts. ``x`` and ``y`` are SVG coordinates in a 200×200 viewBox centred at
    (100, 100) with full-radius 90 — the demo UI can drop them straight into
    a ``<polygon>``.
    """

    total: int
    axes: dict[str, dict[str, int]]
    shares: dict[str, dict[str, float]]
    blind_threshold: float
    blind_spots: list[str]
    radar_points: list[dict]
    radar_polygon: str            # "x1,y1 x2,y2 ..." ready for SVG
    consistency_unreliable: int = 0  # set by callers who attach consistency info

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _category_for(prompt: AdversarialPrompt) -> str:
    cat = (prompt.category or "").lower()
    if cat in CATEGORY_AXIS:
        return cat
    # Fall back to the closest axis name we know about.
    if "inject" in cat:    return "injection"
    if "jail"   in cat:    return "jailbreak"
    if "edge"   in cat:    return "edge_case"
    if "bound"  in cat:    return "boundary"
    if "encode" in cat:    return "encoding"
    if "indir"  in cat:    return "indirect"
    if "multiagent" in cat or "multi_agent" in cat or "inter_agent" in cat:    return "multiagent"
    if "agent"  in cat:    return "agentic"
    if "multi"  in cat or "turn" in cat:    return "multiturn"
    return "edge_case"   # safe fallback


def _severity_for(prompt: AdversarialPrompt) -> str:
    # Honour any explicit attribute the prompt may carry (multilingual.py adds
    # `severity` via subclass for its 50-case battery).
    sev = getattr(prompt, "severity", None)
    if sev and sev in SEVERITY_AXIS:
        return sev
    return _DEFAULT_SEVERITY.get(_category_for(prompt), "medium")


def _language_for(prompt: AdversarialPrompt) -> str:
    lang = getattr(prompt, "language", None)
    if lang and lang in LANGUAGE_AXIS:
        return lang
    text = (prompt.text or "").lower()
    # Cheap heuristic for the 4 supported languages — covers the multilingual
    # battery's actual phrasing without dragging in a real langdetect dep.
    if any(w in text for w in (" und ", " nicht ", " bitte ", " ist ", " sie ")):
        return "de"
    if any(w in text for w in (" et ", " ne pas ", " vous ", " s'il vous ", "ignorez")):
        return "fr"
    if any(w in text for w in (" no ", " que ", " usted ", "ignora ", " hola ")):
        return "es"
    return "en"


def _encoding_for(prompt: AdversarialPrompt) -> str:
    enc = getattr(prompt, "encoding", None)
    if enc and enc in ENCODING_AXIS:
        return enc
    text = prompt.text or ""
    # Zero-width chars
    if "​" in text or "‌" in text or "‍" in text or "﻿" in text:
        return "unicode_zwsp"
    return "plain"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_coverage(
    source: Iterable[AdversarialPrompt] | AdversarialDataset,
    blind_threshold: float = 0.05,
) -> CoverageMap:
    """Compute a :class:`CoverageMap` from a dataset or iterable of prompts.

    Parameters
    ----------
    source:
        Any iterable of :class:`AdversarialPrompt` — typically an
        :class:`AdversarialDataset`.
    blind_threshold:
        Minimum share required for a bucket to not be flagged a blind spot.
        Defaults to 5% (0.05). Set to 0 to disable blind-spot flagging.
    """
    if not 0.0 <= blind_threshold <= 1.0:
        raise ValueError("blind_threshold must be in [0, 1]")

    prompts = list(source)
    total = len(prompts)

    axes: dict[str, dict[str, int]] = {
        "category": {b: 0 for b in CATEGORY_AXIS},
        "severity": {b: 0 for b in SEVERITY_AXIS},
        "language": {b: 0 for b in LANGUAGE_AXIS},
        "encoding": {b: 0 for b in ENCODING_AXIS},
    }
    for p in prompts:
        axes["category"][_category_for(p)] += 1
        axes["severity"][_severity_for(p)] += 1
        axes["language"][_language_for(p)] += 1
        axes["encoding"][_encoding_for(p)] += 1

    # Shares per axis-bucket (normalised within each axis).
    shares: dict[str, dict[str, float]] = {}
    for axis_name, counts in axes.items():
        s = sum(counts.values()) or 1
        shares[axis_name] = {b: counts[b] / s for b in counts}

    # Blind spots are bucket shares < threshold, BUT we don't flag a bucket
    # that has zero shares when the axis is also empty — that's no signal.
    blind_spots: list[str] = []
    if total > 0 and blind_threshold > 0:
        for axis_name, counts in axes.items():
            for b in counts:
                if shares[axis_name][b] < blind_threshold:
                    blind_spots.append(f"{axis_name}.{b}")

    radar_points, radar_polygon = _build_radar(shares)

    return CoverageMap(
        total=total,
        axes=axes,
        shares=shares,
        blind_threshold=blind_threshold,
        blind_spots=blind_spots,
        radar_points=radar_points,
        radar_polygon=radar_polygon,
    )


def _build_radar(shares: Mapping[str, Mapping[str, float]]) -> tuple[list[dict], str]:
    """Flatten per-axis bucket shares into one radar polygon.

    Each axis contributes its buckets in declaration order. Buckets are
    placed evenly around a 360° circle. The radial distance for each bucket
    is normalised to the *max share within its axis* so each axis's tallest
    bucket reaches the rim — this gives a much more readable shape than a
    raw share normalisation when some axes have many buckets.
    """
    flat: list[tuple[str, str, float]] = []
    for axis_name in ("category", "severity", "language", "encoding"):
        bucket_shares = shares.get(axis_name, {})
        if not bucket_shares:
            continue
        axis_max = max(bucket_shares.values()) or 1.0
        for bucket, share in bucket_shares.items():
            flat.append((axis_name, bucket, share / axis_max))

    if not flat:
        return ([], "")

    n = len(flat)
    cx, cy, r_full = 100.0, 100.0, 90.0
    points: list[dict] = []
    polyparts: list[str] = []
    for i, (axis_name, bucket, normalised) in enumerate(flat):
        # Start at top (-π/2), go clockwise.
        angle = -math.pi / 2 + (i / n) * 2 * math.pi
        r = max(2.0, normalised * r_full)   # tiny spike for zero so chart stays a polygon
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        points.append({
            "axis": axis_name,
            "bucket": bucket,
            "share_norm": round(normalised, 4),
            "x": round(x, 2),
            "y": round(y, 2),
        })
        polyparts.append(f"{x:.2f},{y:.2f}")
    return points, " ".join(polyparts)


def label_positions(radar_points: list[dict], radius_extra: float = 20.0) -> list[dict]:
    """Return label coordinates pushed outward from each radar vertex.

    Useful for the demo UI when laying axis labels around the spider chart.
    """
    cx, cy = 100.0, 100.0
    out: list[dict] = []
    for p in radar_points:
        dx, dy = p["x"] - cx, p["y"] - cy
        d = math.hypot(dx, dy) or 1.0
        scale = (d + radius_extra) / d
        out.append({
            **p,
            "lx": round(cx + dx * scale, 2),
            "ly": round(cy + dy * scale, 2),
        })
    return out
