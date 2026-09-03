"""Fully reproducible worked example: Beat the Blues.

Downloads a publicly redistributable randomized trial, applies the
pre-specified specification menu, and prints the table behind Figure 2a of
the manuscript. No credentials, no data use agreement, no local data files.

The trial is Beat the Blues, a randomized comparison of computerised
cognitive behavioural therapy against treatment as usual, distributed in
the R package HSAUR and mirrored publicly as CSV. The endpoint is the Beck
Depression Inventory at four months; lower is better, so it is negated.

Expected output: the mean-based P-value crosses the conventional 0.05
threshold across the menu (0.029 under winsorising, 0.353 on a log scale)
while the rank-based P-value moves by less than 0.02 orders of magnitude.

Usage:  python reproduce_worked_example.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

URL = ("https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/"
       "master/csv/HSAUR/BtheB.csv")


# --- the pre-specified specification menu (five, sample-preserving) --------

def _pooled_bounds(a, b, pct):
    pooled = np.concatenate([a, b])
    return np.percentile(pooled, pct), np.percentile(pooled, 100 - pct)


def as_recorded(a, b):
    return a, b


def winsorize(pct):
    def f(a, b):
        lo, hi = _pooled_bounds(a, b, pct)
        return np.clip(a, lo, hi), np.clip(b, lo, hi)
    return f


def log_shift(a, b):
    pooled = np.concatenate([a, b])
    lo = pooled.min()
    c = 0.0 if lo > 0 else -lo + 0.01 * max(np.ptp(pooled), 1.0)
    return np.log(a + c), np.log(b + c)


def normal_scores(a, b):
    pooled = np.concatenate([a, b])
    r = stats.rankdata(pooled)
    z = stats.norm.ppf(r / (len(pooled) + 1))
    return z[:len(a)], z[len(a):]


MENU = [
    ("as recorded", as_recorded),
    ("winsorize 1%", winsorize(1.0)),
    ("winsorize 5%", winsorize(5.0)),
    ("log scale", log_shift),
    ("normal scores", normal_scores),
]


def main() -> None:
    df = pd.read_csv(URL)
    sub = df[["treatment", "bdi.4m"]].dropna()

    # Lower BDI is better, so orient the outcome as larger-is-better.
    control = -sub.loc[sub.treatment == "TAU", "bdi.4m"].to_numpy(float)
    treated = -sub.loc[sub.treatment != "TAU", "bdi.4m"].to_numpy(float)

    print(f"Beat the Blues, BDI at 4 months. "
          f"n = {len(control)} control, {len(treated)} treated\n")
    print(f"{'specification':16s} {'mean-based P':>14s} {'rank-based P':>14s}"
          f" {'verdict (mean)':>16s}")
    print("-" * 64)

    tp, mp = [], []
    for name, fn in MENU:
        x, z = fn(control, treated)
        t = stats.ttest_ind(z, x, equal_var=False).pvalue
        m = stats.mannwhitneyu(z, x, alternative="two-sided").pvalue
        tp.append(t)
        mp.append(m)
        verdict = "significant" if t < 0.05 else "not significant"
        print(f"{name:16s} {t:14.4f} {m:14.4f} {verdict:>16s}")

    span = lambda v: np.log10(max(v)) - np.log10(min(v))
    print("-" * 64)
    print(f"{'span (orders)':16s} {span(tp):14.3f} {span(mp):14.3f}")
    print(f"\nMean-based verdict changes across the menu: "
          f"{len({p < 0.05 for p in tp}) > 1}")
    print(f"Rank-based verdict changes across the menu: "
          f"{len({p < 0.05 for p in mp}) > 1}")


if __name__ == "__main__":
    main()
