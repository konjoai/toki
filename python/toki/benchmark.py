"""Statistical benchmark analysis — pure stdlib, no scipy/numpy required."""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional

from toki.results import ExperimentResult


# ---------------------------------------------------------------------------
# BenchmarkStats
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkStats:
    n: int
    mean: float
    std: float
    p50: float
    p95: float
    p99: float
    min: float
    max: float


def _percentile(sorted_data: List[float], p: float) -> float:
    """Return the p-th percentile (0 <= p <= 100) of an already-sorted list.

    Uses the nearest-rank method so no interpolation is needed and the result
    is always a value that was actually observed in the dataset.
    """
    n = len(sorted_data)
    if n == 0:
        raise ValueError("Cannot compute percentile of an empty list")
    if n == 1:
        return sorted_data[0]
    # Index via ceiling of (p/100 * n) — 1-based → 0-based
    idx = max(0, math.ceil(p / 100.0 * n) - 1)
    return sorted_data[min(idx, n - 1)]


def compute_stats(scores: List[float]) -> BenchmarkStats:
    """Compute descriptive statistics for a list of scores.

    p50 / p95 / p99 are computed via a pure sorted-list percentile (no scipy).
    std uses ``statistics.stdev`` for n > 1, else 0.0.
    """
    if not scores:
        raise ValueError("scores must be non-empty")

    n = len(scores)
    mean = statistics.mean(scores)
    std = statistics.stdev(scores) if n > 1 else 0.0
    sorted_s = sorted(scores)
    return BenchmarkStats(
        n=n,
        mean=mean,
        std=std,
        p50=_percentile(sorted_s, 50),
        p95=_percentile(sorted_s, 95),
        p99=_percentile(sorted_s, 99),
        min=sorted_s[0],
        max=sorted_s[-1],
    )


# ---------------------------------------------------------------------------
# Statistical significance tests — pure math / statistics
# ---------------------------------------------------------------------------

@dataclass
class StatTestResult:
    test_name: str      # "paired_t_test" or "wilcoxon"
    statistic: float
    p_value: float
    significant: bool   # p_value < alpha
    alpha: float
    n: int


def _normal_cdf(z: float) -> float:
    """Standard normal CDF via erfc.  Φ(z) = 0.5 * erfc(-z / sqrt(2))."""
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def _regularized_incomplete_beta(x: float, a: float, b: float, max_iter: int = 200) -> float:
    """Regularized incomplete beta function I_x(a, b) via continued-fraction expansion.

    Used to compute the CDF of the t-distribution for small n (n ≤ 30).
    Reference: Abramowitz and Stegun §26.5 / Numerical Recipes §6.4.
    """
    if x < 0.0 or x > 1.0:
        raise ValueError(f"x must be in [0, 1], got {x}")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    # For symmetry use the identity: I_x(a,b) = 1 - I_{1-x}(b,a) when x > (a+1)/(a+b+2)
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _regularized_incomplete_beta(1.0 - x, b, a, max_iter)

    # Log of the beta function via log-gamma
    ln_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    # Front factor
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - ln_beta) / a

    # Lentz's continued-fraction method
    TINY = 1e-30
    f = TINY
    C = f
    D = 0.0
    for m in range(max_iter):
        for j in (0, 1):
            if m == 0 and j == 0:
                d = 1.0
            elif j == 0:
                # even step
                d = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
            else:
                # odd step
                d = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
            D = 1.0 + d * D
            if abs(D) < TINY:
                D = TINY
            D = 1.0 / D
            C = 1.0 + d / C
            if abs(C) < TINY:
                C = TINY
            delta = C * D
            f *= delta
            if abs(delta - 1.0) < 1e-10:
                return front * f
    return front * f


def _t_dist_two_tailed_p(t: float, df: int) -> float:
    """Two-tailed p-value for Student's t with *df* degrees of freedom.

    For df > 30 we use the normal approximation (|t| → z); otherwise we use
    the regularized incomplete beta function:
        p = 2 * I_{df/(df+t^2)}(df/2, 0.5)
    """
    if math.isinf(t) or math.isnan(t):
        return 0.0
    abs_t = abs(t)
    if df > 30:
        # Normal approximation
        p = 2.0 * (1.0 - _normal_cdf(abs_t))
        return max(0.0, min(1.0, p))
    # Exact via I_x(a, b)
    x = df / (df + abs_t ** 2)
    p = 2.0 * _regularized_incomplete_beta(x, df / 2.0, 0.5)
    return max(0.0, min(1.0, p))


def paired_t_test(
    before: List[float],
    after: List[float],
    alpha: float = 0.05,
) -> StatTestResult:
    """Paired t-test for before/after score vectors.

    t = mean(d) / (std(d) / sqrt(n))
    df = n - 1
    Two-tailed p-value uses the t-distribution (normal approximation for n > 30).

    Args:
        before: scores before intervention.
        after:  scores after intervention; must be same length as *before*.
        alpha:  significance threshold (default 0.05).

    Returns:
        StatTestResult with statistic, p_value, and significant flag.
    """
    if len(before) != len(after):
        raise ValueError("before and after must have the same length")
    n = len(before)
    if n < 2:
        raise ValueError("paired_t_test requires n >= 2")

    diffs = [a - b for b, a in zip(before, after)]
    mean_d = statistics.mean(diffs)
    std_d = statistics.stdev(diffs)

    if std_d == 0.0:
        if mean_d == 0.0:
            # All differences are exactly zero → no change; t undefined, treat p=1
            t_stat = 0.0
            p_val = 1.0
        else:
            # All differences are identical and non-zero → perfect separation.
            # t → ∞, p → 0; use a finite sentinel that is unambiguously significant.
            t_stat = float("inf")
            p_val = 0.0
    else:
        t_stat = mean_d / (std_d / math.sqrt(n))
        p_val = _t_dist_two_tailed_p(t_stat, n - 1)

    return StatTestResult(
        test_name="paired_t_test",
        statistic=t_stat,
        p_value=p_val,
        significant=p_val < alpha,
        alpha=alpha,
        n=n,
    )


def wilcoxon_test(
    before: List[float],
    after: List[float],
    alpha: float = 0.05,
) -> StatTestResult:
    """Wilcoxon signed-rank test (normal approximation for p-value).

    Algorithm:
    1. Compute differences d_i = after_i - before_i; drop zeros.
    2. Rank |d_i| (average-rank ties).
    3. W+ = sum of ranks where d_i > 0; W- = sum where d_i < 0.
    4. W = min(W+, W-).
    5. z = (W - n*(n+1)/4) / sqrt(n*(n+1)*(2n+1)/24).
    6. p = 2 * (1 - Φ(|z|)).

    Args:
        before: scores before intervention.
        after:  scores after intervention; must be same length as *before*.
        alpha:  significance threshold (default 0.05).

    Returns:
        StatTestResult with statistic=W, p_value, and significant flag.
    """
    if len(before) != len(after):
        raise ValueError("before and after must have the same length")

    diffs = [a - b for b, a in zip(before, after)]
    # Remove zero differences (tied pairs contribute nothing)
    nonzero = [(d, abs(d)) for d in diffs if d != 0.0]
    n = len(nonzero)

    if n == 0:
        # All differences are zero → no evidence of change
        return StatTestResult(
            test_name="wilcoxon",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            alpha=alpha,
            n=len(before),
        )

    # Rank by absolute value; handle ties with average ranks
    sorted_by_abs = sorted(range(n), key=lambda i: nonzero[i][1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        # Find extent of tie group
        while j < n and nonzero[sorted_by_abs[j]][1] == nonzero[sorted_by_abs[i]][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0  # average of 1-based ranks i+1 .. j
        for k in range(i, j):
            ranks[sorted_by_abs[k]] = avg_rank
        i = j

    w_plus = sum(ranks[i] for i in range(n) if nonzero[i][0] > 0)
    w_minus = sum(ranks[i] for i in range(n) if nonzero[i][0] < 0)
    W = min(w_plus, w_minus)

    # Normal approximation
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    if var_w == 0.0:
        z = 0.0
    else:
        z = (W - mean_w) / math.sqrt(var_w)

    p_val = 2.0 * (1.0 - _normal_cdf(abs(z)))
    p_val = max(0.0, min(1.0, p_val))

    return StatTestResult(
        test_name="wilcoxon",
        statistic=W,
        p_value=p_val,
        significant=p_val < alpha,
        alpha=alpha,
        n=n,
    )


# ---------------------------------------------------------------------------
# BenchmarkReport + generate_report
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkReport:
    experiment_name: str
    timestamp: str
    pre_stats: BenchmarkStats
    post_stats: Optional[BenchmarkStats]
    t_test: Optional[StatTestResult]
    wilcoxon: Optional[StatTestResult]
    score_delta: Optional[float]           # post_mean - pre_mean, or None
    category_pre: Dict[str, BenchmarkStats]
    category_post: Optional[Dict[str, BenchmarkStats]]


def generate_report(
    result: ExperimentResult,
    pre_scores: List[float],
    post_scores: Optional[List[float]] = None,
    category_pre: Optional[Dict[str, List[float]]] = None,
    category_post: Optional[Dict[str, List[float]]] = None,
) -> BenchmarkReport:
    """Build a :class:`BenchmarkReport` from an experiment result and raw score lists.

    Args:
        result:       Completed :class:`ExperimentResult`.
        pre_scores:   Raw per-prompt scores before intervention.
        post_scores:  Raw per-prompt scores after intervention (optional).
        category_pre:  Per-category score lists before intervention (optional).
        category_post: Per-category score lists after intervention (optional).

    Returns:
        A fully populated :class:`BenchmarkReport`.
    """
    pre_stats = compute_stats(pre_scores)

    post_stats: Optional[BenchmarkStats] = None
    t_test: Optional[StatTestResult] = None
    wilcoxon: Optional[StatTestResult] = None
    score_delta: Optional[float] = None

    if post_scores is not None and len(post_scores) > 0:
        post_stats = compute_stats(post_scores)
        score_delta = post_stats.mean - pre_stats.mean
        if len(pre_scores) == len(post_scores) and len(pre_scores) >= 2:
            t_test = paired_t_test(pre_scores, post_scores)
            wilcoxon = wilcoxon_test(pre_scores, post_scores)

    cat_pre_stats: Dict[str, BenchmarkStats] = {}
    if category_pre:
        for cat, scores in category_pre.items():
            if scores:
                cat_pre_stats[cat] = compute_stats(scores)

    cat_post_stats: Optional[Dict[str, BenchmarkStats]] = None
    if category_post:
        cat_post_stats = {}
        for cat, scores in category_post.items():
            if scores:
                cat_post_stats[cat] = compute_stats(scores)

    return BenchmarkReport(
        experiment_name=result.name,
        timestamp=result.timestamp,
        pre_stats=pre_stats,
        post_stats=post_stats,
        t_test=t_test,
        wilcoxon=wilcoxon,
        score_delta=score_delta,
        category_pre=cat_pre_stats,
        category_post=cat_post_stats,
    )
