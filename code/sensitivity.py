"""Outlier and scale sensitivity of the mean-versus-rank discordance.

The headline finding is that a t-test and a Mann-Whitney test on the same
trial data can differ by nine orders of magnitude in significance.  The
obvious objection is that a handful of extreme observations produce this,
so the effect would vanish under any sensible outlier handling.

This module tests that objection directly.  For each stratum it recomputes
both tests under seven treatments of the data: untouched, symmetric
trimming at two levels, winsorising at two levels, an explicit cap on the
extreme tail, a log-ratio transform that removes the skew by construction,
and van der Waerden normal scores.

Trimming and winsorising use pooled within-stratum quantiles so that both
arms are cut at the same points, which avoids manufacturing a difference
between them.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from heterogeneity import _log10_p_gap, cohens_d, mann_whitney_auc  # noqa: E402


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def _pooled_bounds(y_c: np.ndarray, y_t: np.ndarray, pct: float) -> Tuple[float, float]:
    pooled = np.concatenate([y_c, y_t])
    return float(np.percentile(pooled, pct)), float(np.percentile(pooled, 100 - pct))


def t_raw(y_c: np.ndarray, y_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return y_c, y_t


def make_trim(pct: float) -> Callable:
    def _f(y_c: np.ndarray, y_t: np.ndarray):
        lo, hi = _pooled_bounds(y_c, y_t, pct)
        return y_c[(y_c >= lo) & (y_c <= hi)], y_t[(y_t >= lo) & (y_t <= hi)]
    return _f


def make_winsor(pct: float) -> Callable:
    def _f(y_c: np.ndarray, y_t: np.ndarray):
        lo, hi = _pooled_bounds(y_c, y_t, pct)
        return np.clip(y_c, lo, hi), np.clip(y_t, lo, hi)
    return _f


def make_cap(threshold: float) -> Callable:
    """Drop observations beyond a fixed value on the original PIEF scale.

    Applied to oriented data, so the cap is expressed as the oriented
    equivalent of the raw threshold.
    """
    def _f(y_c: np.ndarray, y_t: np.ndarray):
        return y_c[y_c >= -threshold], y_t[y_t >= -threshold]
    return _f


def t_logratio(y_c: np.ndarray, y_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Log of the post/pre episode ratio.

    Oriented PIEF is -PIEF, so raw = -y and ratio = 1 + raw/100.  Complete
    resolution sits at ratio 0, so the ratio is floored at 0.01, treating
    a full response as a reduction to one percent of baseline.  The result
    is negated again to keep larger meaning better.
    """
    def _tr(y):
        ratio = 1.0 + (-y) / 100.0
        return -np.log(np.maximum(ratio, 0.01))
    return _tr(y_c), _tr(y_t)


def t_normal_scores(y_c: np.ndarray, y_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Van der Waerden scores: rank the pooled sample, then map to normal quantiles.

    A t-test on these scores is asymptotically equivalent to a rank test,
    so it shows what the mean-based machinery achieves once the scale
    problem is removed entirely.
    """
    pooled = np.concatenate([y_c, y_t])
    n = len(pooled)
    ranks = stats.rankdata(pooled)
    scores = stats.norm.ppf(ranks / (n + 1))
    return scores[: len(y_c)], scores[len(y_c):]


def t_log_shift(y_c: np.ndarray, y_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Log transform after shifting the pooled sample to be positive.

    The general-purpose skew remover, applicable to any outcome.  Unlike
    :func:`t_logratio` it makes no assumption that the endpoint is a
    percent change.
    """
    pooled = np.concatenate([y_c, y_t])
    shift = 0.0
    lo = pooled.min()
    if lo <= 0:
        shift = -lo + 0.01 * max(np.ptp(pooled), 1.0)
    return np.log(y_c + shift), np.log(y_t + shift)


# Transforms valid for any continuous endpoint.
TRANSFORMS: List[Tuple[str, Callable]] = [
    ("raw", t_raw),
    ("trim 1%", make_trim(1.0)),
    ("trim 5%", make_trim(5.0)),
    ("winsor 1%", make_winsor(1.0)),
    ("winsor 5%", make_winsor(5.0)),
    ("log-shift", t_log_shift),
    ("normal scores", t_normal_scores),
]

# Extra transforms that assume the endpoint is a percent change bounded
# below at -100%, as the Lilly PIEF variable is.  Do NOT apply these to a
# general outcome: on an unbounded endpoint such as birthweight the
# log-ratio transform is meaningless and produces spurious significance.
PERCENT_CHANGE_TRANSFORMS: List[Tuple[str, Callable]] = [
    ("drop >500%", make_cap(500.0)),
    ("log-ratio", t_logratio),
]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(y_c: np.ndarray, y_t: np.ndarray) -> Dict[str, float]:
    """Both tests plus the discordance metrics for one processed sample."""
    out: Dict[str, float] = {
        "n": len(y_c) + len(y_t),
        "auc": float("nan"),
        "d": float("nan"),
        "t_p": float("nan"),
        "mw_p": float("nan"),
        "gap": float("nan"),
        "skew": float("nan"),
        "kurt": float("nan"),
        "var_ratio": float("nan"),
    }
    if len(y_c) < 3 or len(y_t) < 3:
        return out
    out["auc"] = mann_whitney_auc(y_c, y_t)
    out["d"] = cohens_d(y_c, y_t)
    out["t_p"] = float(stats.ttest_ind(y_t, y_c, equal_var=False).pvalue)
    out["mw_p"] = float(stats.mannwhitneyu(y_t, y_c, alternative="two-sided").pvalue)
    out["gap"] = _log10_p_gap(out["mw_p"], out["t_p"])
    pooled = np.concatenate([y_c, y_t])
    out["skew"] = float(stats.skew(pooled))
    out["kurt"] = float(stats.kurtosis(pooled))
    # Ratio of arm standard deviations.  The second-order expansion in the
    # supplement predicts that the transformed mean contrast picks up a term
    # proportional to the difference in arm variances, so this is the
    # quantity theory says should drive the mean-based instability.
    s_c, s_t = np.std(y_c, ddof=1), np.std(y_t, ddof=1)
    out["var_ratio"] = float(max(s_t, s_c) / min(s_t, s_c)) if min(s_t, s_c) > 0 else float("nan")
    return out


def run_sensitivity(
    df: pd.DataFrame,
    outcome: str,
    arm: str,
    stratum: str,
    control_label: object,
    higher_is_better: bool,
    percent_change: bool = False,
) -> pd.DataFrame:
    """Sensitivity table: one row per stratum and transform.

    Set `percent_change=True` only when the endpoint is a percent change
    bounded below at -100%, which enables the two extra transforms in
    :data:`PERCENT_CHANGE_TRANSFORMS`.
    """
    transforms = TRANSFORMS + (PERCENT_CHANGE_TRANSFORMS if percent_change else [])
    work = df[[outcome, arm, stratum]].dropna().copy()
    sign = 1.0 if higher_is_better else -1.0
    work["_y"] = sign * work[outcome].to_numpy(dtype=float)
    work["_trt"] = (work[arm] != control_label).astype(int)

    rows = []
    for lev in sorted(work[stratum].unique()):
        block = work[work[stratum] == lev]
        y_c0 = block.loc[block["_trt"] == 0, "_y"].to_numpy()
        y_t0 = block.loc[block["_trt"] == 1, "_y"].to_numpy()
        for name, fn in transforms:
            y_c, y_t = fn(y_c0, y_t0)
            rec = evaluate(y_c, y_t)
            rec.update({"stratum": lev, "transform": name})
            rows.append(rec)
    cols = ["stratum", "transform", "n", "auc", "d", "t_p", "mw_p", "gap", "skew", "kurt"]
    return pd.DataFrame(rows)[cols]
