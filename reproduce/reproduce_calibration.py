"""Reproduce the null calibration and the mechanism prediction.

Requires no data at all. Two results from the manuscript:

  1. CALIBRATION. Under normal arms with equal variance the specification
     menu should be nearly irrelevant, so this is where a sceptic should
     check whether the mean-based span is larger for a mechanical reason.
     It is not zero: the mean-based verdict still changes in roughly 5% of
     replicates against roughly 0% rank-based. That is the phenomenon in
     miniature and it establishes the floor against which observed rates
     must be read.

  2. MECHANISM. Equation (3) predicts the transformed mean contrast
     acquires a term proportional to the difference in arm variances, so
     instability should grow as the arms differ more in dispersion.
     It does, at fixed shape; tail weight acts on top of that and is
     not captured by the second-order term.

Reduced replicate count by default so this runs in about a minute; pass
--full for the 400 replicates used in the manuscript.
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy import stats

from reproduce_worked_example import MENU


def spans_and_flips(a, b):
    tp, mp = [], []
    for _, fn in MENU:
        x, z = fn(a, b)
        tp.append(stats.ttest_ind(z, x, equal_var=False).pvalue)
        mp.append(stats.mannwhitneyu(z, x, alternative="two-sided").pvalue)
    span = lambda v: np.log10(max(v)) - np.log10(min(v))
    return (span(tp), span(mp),
            len({p < 0.05 for p in tp}) > 1,
            len({p < 0.05 for p in mp}) > 1)


# The treated arm is built to a prescribed mean and standard deviation rather
# than by rescaling a variate.  Widening a lognormal by inflating its log-scale
# sigma also moves its mean, which would confound dispersion with effect size
# and make the "no effect" condition not null.  Moment matching keeps `shift`
# a standardized mean difference and `sd_ratio` a ratio of standard deviations,
# independently.
LN_SIG = 0.6
CONTROL = {"normal": (0.0, 1.0),
           "lognormal": (float(np.exp(LN_SIG ** 2 / 2)),
                         float(np.sqrt((np.exp(LN_SIG ** 2) - 1)
                                       * np.exp(LN_SIG ** 2))))}


def draw(dist, n, shift, vr, rng):
    mu_c, sd_c = CONTROL[dist]
    mu_t, sd_t = mu_c + shift * sd_c, vr * sd_c
    if dist == "normal":
        return rng.normal(mu_c, sd_c, n), rng.normal(mu_t, sd_t, n)
    if dist == "lognormal":
        def ln(m, s, k):
            v = np.log1p((s / m) ** 2)
            return rng.lognormal(np.log(m) - v / 2, np.sqrt(v), k)
        return ln(mu_c, sd_c, n), ln(mu_t, sd_t, n)
    raise ValueError(dist)


def cell(dist, n, shift, vr, reps, rng):
    t_fl, m_fl, ex = [], [], []
    for _ in range(reps):
        a, b = draw(dist, n, shift, vr, rng)
        ts, ms, tf, mf = spans_and_flips(a, b)
        if np.isfinite(ts) and np.isfinite(ms):
            t_fl.append(tf); m_fl.append(mf); ex.append(ts - ms)
    return 100 * np.mean(t_fl), 100 * np.mean(m_fl), float(np.median(ex))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="400 replicates per cell, as in the manuscript")
    args = ap.parse_args()
    reps = 400 if args.full else 120
    rng = np.random.default_rng(20260829)

    print(f"{reps} replicates per cell, n = 200 per arm\n")
    print("1. CALIBRATION: normal arms, equal dispersion, no treatment effect")
    print("   The specification menu should be nearly irrelevant here.\n")
    t, m, e = cell("normal", 200, 0.0, 1.0, reps, rng)
    print(f"   mean-based verdict changed: {t:5.1f}%")
    print(f"   rank-based verdict changed: {m:5.1f}%")
    print(f"   median excess span:         {e:5.3f}")
    print("\n   The mean-based floor is non-zero.  Observed trial rates of")
    print("   roughly 9% must be read against this, not against zero.\n")

    print("2. MECHANISM: verdict-change rate by ratio of arm standard\n"
          "   deviations, at a standardized effect of 0.2\n")
    print(f"   {'distribution':14s} {'sd ratio':>10s} {'mean-based':>12s} {'rank-based':>12s}")
    print("   " + "-" * 50)
    for dist in ("normal", "lognormal"):
        for vr in (1.0, 1.5, 2.5):
            t, m, _ = cell(dist, 200, 0.2, vr, reps, rng)
            print(f"   {dist:14s} {vr:10.1f} {t:11.1f}% {m:11.1f}%")
    print("\n   At fixed shape the mean-based rate grows as the arms differ")
    print("   more in dispersion, which is the second-order term of equation (3)")
    print("   behaving as derived.  Tail weight acts on top of it: compare the")
    print("   two distributions at equal dispersion.  The rank-based rate stays")
    print("   at its floor throughout.")


if __name__ == "__main__":
    main()
