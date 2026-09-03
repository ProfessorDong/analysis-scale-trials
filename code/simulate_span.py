"""Calibration and characterisation of the span statistic.

Two questions, in order of importance.

CALIBRATION. The span is the log10 range of an estimator's p-value across
the specification menu. A sceptical reader should ask whether the
mean-based span is larger for a mechanical reason rather than a
substantive one: perhaps a t-test simply reaches more extreme p-values, so
its log range is wider whatever the data look like. If that were so the
whole comparison would be confounded and the paper would be wrong.

The test is a setting where the specification menu should not matter:
normal arms with equal variance, where every member of the menu is close to
a linear rescaling and no summary is disturbed by it. If the two spans
agree there, the statistic is fair and any divergence elsewhere is a
property of the data rather than of the measure.

CHARACTERISATION. Having established that, we vary skewness, variance
ratio, effect size and sample size to describe when the mean-based span
grows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from sensitivity import TRANSFORMS  # noqa: E402
from sweep import NON_DELETING, _span  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def spans(a: np.ndarray, b: np.ndarray) -> tuple[float, float, bool, bool]:
    """Spans and verdict-change indicators across the sample-preserving menu.

    The verdict-change indicator is the quantity the paper actually leans
    on, so it needs its own null calibration: how often does an estimator's
    conclusion at alpha = 0.05 change across the menu when the menu ought
    to be irrelevant?
    """
    tp, mp = [], []
    for name, fn in TRANSFORMS:
        if name not in NON_DELETING:
            continue
        try:
            x, z = fn(a, b)
            t = stats.ttest_ind(z, x, equal_var=False).pvalue
            m = stats.mannwhitneyu(z, x, alternative="two-sided").pvalue
            if np.isfinite(t) and np.isfinite(m):
                tp.append(t)
                mp.append(m)
        except Exception:
            continue
    if len(tp) < 3:
        return float("nan"), float("nan"), False, False
    t_flip = len({p < 0.05 for p in tp}) > 1
    m_flip = len({p < 0.05 for p in mp}) > 1
    return _span(tp), _span(mp), t_flip, m_flip


# Control-arm moments for each generating family.  The treated arm is built
# to a target mean and standard deviation rather than by scaling a variate,
# so that `shift` is a standardised mean difference and `vr` a ratio of
# standard deviations, independently, in every family.
#
# The earlier version multiplied the treated exponential variate by vr and
# widened the lognormal's log-scale sigma, both of which move the mean as
# well as the spread: at vr = 2.5 the nominally null cell had a treated mean
# 2.5 times the control's.  The grid was not orthogonal and the "no effect"
# condition was not null, which made the exponential and contaminated rows
# of the simulation table run backwards -- not because sensitivity fell, but
# because the effect became too large, or the variance too great, for any
# specification to sit near the threshold.
LN_SIG = 0.6
CONTAM_SD = float(np.sqrt(0.95 * 1.0 + 0.05 * 8.0 ** 2))

CONTROL = {
    "normal": (0.0, 1.0),
    "lognormal": (float(np.exp(LN_SIG ** 2 / 2)),
                  float(np.sqrt((np.exp(LN_SIG ** 2) - 1) * np.exp(LN_SIG ** 2)))),
    "gamma": (1.0, 1.0),
    "contaminated": (0.0, CONTAM_SD),
}


def draw(dist: str, n: int, shift: float, vr: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """Control and treated samples with prescribed mean and dispersion.

    `shift` is the mean difference in control standard deviations; `vr` is
    the treated-to-control ratio of standard deviations.  The two are
    varied independently, so a cell with shift = 0 has equal means whatever
    the variance ratio.
    """
    mu_c, sd_c = CONTROL[dist]
    mu_t, sd_t = mu_c + shift * sd_c, vr * sd_c

    if dist == "normal":
        a = rng.normal(mu_c, sd_c, n)
        b = rng.normal(mu_t, sd_t, n)
    elif dist == "lognormal":
        # Exact moment matching: sigma^2 = log(1 + s^2/m^2), mu = log m - sigma^2/2.
        def ln(m, s, k):
            v = np.log1p((s / m) ** 2)
            return rng.lognormal(np.log(m) - v / 2, np.sqrt(v), k)
        a, b = ln(mu_c, sd_c, n), ln(mu_t, sd_t, n)
    elif dist == "gamma":
        # Gamma with shape m^2/s^2 and scale s^2/m.  At mean = sd = 1 this is
        # exactly Exp(1), so the equal-spread column is the exponential case;
        # unlike scaling an exponential variate, it can widen the spread
        # without dragging the mean along with it.
        def ga(m, s, k):
            return rng.gamma(m * m / (s * s), (s * s) / m, k)
        a, b = ga(mu_c, sd_c, n), ga(mu_t, sd_t, n)
    elif dist == "contaminated":
        # 95% normal, 5% from a component eight times as dispersed.  Scaling
        # both components by the same factor leaves the mean untouched.
        def cm(m, scale, k):
            heavy = rng.random(k) >= 0.95
            return np.where(heavy, rng.normal(m, 8.0 * scale, k),
                            rng.normal(m, 1.0 * scale, k))
        a, b = cm(mu_c, 1.0, n), cm(mu_t, vr, n)
    else:
        raise ValueError(dist)
    return a, b


def run(reps: int = 2000, seed: int = 20260829) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    grid = [
        (dist, n, shift, vr)
        for dist in ("normal", "lognormal", "gamma", "contaminated")
        for n in (50, 200, 800)
        for shift in (0.0, 0.2, 0.5)
        for vr in (1.0, 1.5, 2.5)
    ]
    for dist, n, shift, vr in grid:
        ts, ms, tf, mf = [], [], [], []
        for _ in range(reps):
            a, b = draw(dist, n, shift, vr, rng)
            t, m, t_fl, m_fl = spans(a, b)
            if np.isfinite(t) and np.isfinite(m):
                ts.append(t); ms.append(m); tf.append(t_fl); mf.append(m_fl)
        if not ts:
            continue
        ts, ms = np.array(ts), np.array(ms)
        tf, mf = np.array(tf), np.array(mf)
        rows.append({
            "dist": dist, "n": n, "shift": shift, "var_ratio": vr,
            "reps": len(ts),
            "t_span_median": float(np.median(ts)),
            "mw_span_median": float(np.median(ms)),
            "excess_median": float(np.median(ts - ms)),
            "frac_excess_positive": float((ts > ms).mean()),
            "t_flip_rate": float(tf.mean()),
            "mw_flip_rate": float(mf.mean()),
            # Binomial standard error of the flip rate, so the table can be
            # rounded to a precision the replicate count actually supports.
            "t_flip_se": float(np.sqrt(tf.mean() * (1 - tf.mean()) / len(tf))),
            "mw_flip_se": float(np.sqrt(mf.mean() * (1 - mf.mean()) / len(mf))),
        })
        print(f"  {dist:13s} n={n:4d} shift={shift:.1f} vr={vr:.1f}  "
              f"t={np.median(ts):7.3f}  mw={np.median(ms):7.4f}  "
              f"flip t={tf.mean():5.1%} mw={mf.mean():5.1%}", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = run()
    dest = ROOT / "results" / "simulation_spans.csv"
    df.to_csv(dest, index=False)
    print(f"\nwrote {dest}: {len(df)} cells")
