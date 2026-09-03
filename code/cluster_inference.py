"""Trial-clustered inference for the span comparison.

Comparisons are not independent: 28 trials contribute 2,088 of them, and
three trials contribute 56%. Treating each comparison as an independent
observation, as a naive Wilcoxon or McNemar test does, understates the
standard error badly and produces p-values that cannot be defended.

Three analyses are reported, from most to least conservative.

  1. TRIAL-LEVEL. One value per trial: the median excess span, and the
     within-trial verdict-change rates. n = 28. This is the primary
     analysis because the trial is the unit of sampling.
  2. CLUSTER BOOTSTRAP. Resample whole trials with replacement and
     recompute the pooled statistic, giving an interval that respects the
     dependence.
  3. NAIVE. The comparison-level test, reported only to show how far it
     overstates the evidence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from variable_classes import load_trials

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def load() -> pd.DataFrame:
    """All comparisons, carrying the pre-randomisation flag.

    Pre-randomisation measurements must not enter the primary analysis: a
    variable recorded before allocation cannot carry a treatment effect, so
    including it would answer a different question and dilute the endpoint
    set with observations that are null by construction.  They are kept and
    analysed separately as the negative control.
    """
    return load_trials()


def trial_level(tr: pd.DataFrame) -> pd.DataFrame:
    return tr.groupby("trial").agg(
        n_cmp=("excess_span", "size"),
        t_span=("t_span", "median"),
        mw_span=("mw_span", "median"),
        excess=("excess_span", "median"),
        t_flip=("t_flips", "mean"),
        mw_flip=("mw_flips", "mean"),
    ).reset_index()


def cluster_bootstrap(tr: pd.DataFrame, B: int = 5000, seed: int = 7) -> dict:
    """Resample whole trials, recompute pooled verdict-change rates."""
    rng = np.random.default_rng(seed)
    trials = tr.trial.unique()
    by = {t: g for t, g in tr.groupby("trial")}
    tf, mf, ex = np.empty(B), np.empty(B), np.empty(B)
    for b in range(B):
        pick = rng.choice(trials, size=len(trials), replace=True)
        d = pd.concat([by[t] for t in pick], ignore_index=True)
        tf[b] = d.t_flips.mean()
        mf[b] = d.mw_flips.mean()
        ex[b] = d.excess_span.median()
    q = lambda v: np.percentile(v, [2.5, 97.5])
    return {
        "t_flip": (tr.t_flips.mean(), *q(tf)),
        "mw_flip": (tr.mw_flips.mean(), *q(mf)),
        "diff": ((tr.t_flips.mean() - tr.mw_flips.mean()), *q(tf - mf)),
        "excess_median": (tr.excess_span.median(), *q(ex)),
    }


def main() -> None:
    everything = load()
    tr = everything[~everything.baseline].copy()
    nc = everything[everything.baseline].copy()
    tl = trial_level(tr)
    tl.to_csv(RES / "trial_level_summary.csv", index=False)

    print(f"=== 1. TRIAL-LEVEL, n = {len(tl)} trials (primary analysis) ===")
    w = stats.wilcoxon(tl.t_span, tl.mw_span, alternative="greater")
    print(f"  median trial t-span   {tl.t_span.median():.3f}")
    print(f"  median trial mw-span  {tl.mw_span.median():.4f}")
    print(f"  excess positive in    {(tl.excess > 0).mean():.1%} of trials "
          f"({int((tl.excess > 0).sum())}/{len(tl)})")
    print(f"  Wilcoxon signed-rank, paired by trial:  p = {w.pvalue:.3g}")
    ws = stats.wilcoxon(tl.t_flip, tl.mw_flip, alternative="greater")
    print(f"  mean trial-level flip rate: t {tl.t_flip.mean():.3%}  "
          f"mw {tl.mw_flip.mean():.3%}")
    print(f"  Wilcoxon on flip rates, paired by trial: p = {ws.pvalue:.3g}")
    print(f"  trials where t flipped at least once:  "
          f"{int((tl.t_flip > 0).sum())}/{len(tl)}")
    print(f"  trials where mw flipped at least once: "
          f"{int((tl.mw_flip > 0).sum())}/{len(tl)}")

    print(f"\n=== 2. CLUSTER BOOTSTRAP, 5000 resamples of whole trials ===")
    cb = cluster_bootstrap(tr)
    for k, (est, lo, hi) in cb.items():
        unit = "" if k == "excess_median" else "%"
        scale = 1 if k == "excess_median" else 100
        print(f"  {k:14s} {scale*est:8.3f}{unit}   95% CI "
              f"[{scale*lo:.3f}, {scale*hi:.3f}]{unit}")

    print(f"\n=== 3. NEGATIVE CONTROL: pre-randomisation measurements ===")
    if len(nc):
        ntl = trial_level(nc)
        print(f"  {len(nc)} comparisons from {nc.trial.nunique()} trials, "
              f"no treatment effect possible")
        print(f"  median t-span   {nc.t_span.median():.4f}   "
              f"median mw-span  {nc.mw_span.median():.4f}")
        print(f"  verdict-change rate: t {nc.t_flips.mean():.2%}  "
              f"mw {nc.mw_flips.mean():.2%}")
        print(f"  excess span positive in "
              f"{int((ntl.excess > 0).sum())}/{len(ntl)} trials")
        wn2 = stats.wilcoxon(ntl.t_span, ntl.mw_span, alternative="greater")
        print(f"  Wilcoxon paired by trial: p = {wn2.pvalue:.3g}")

    print(f"\n=== 4. NAIVE comparison-level, for contrast only ===")
    wn = stats.wilcoxon(tr.t_span, tr.mw_span, alternative="greater")
    print(f"  n = {len(tr)} comparisons treated as independent")
    print(f"  Wilcoxon p = {wn.pvalue:.3g}   <-- overstates the evidence")


if __name__ == "__main__":
    main()
