"""Downstream analysis of the manifest-based comparison set.

Reads results/comparisons_v2.csv and computes, for each menu,

  strict   recorded, log_shift, normal_scores      (Proposition 1 exact)
  nondel   strict + winsor1 + winsor5              (the five sample-preserving
                                                    specifications of the earlier draft)
  all      nondel + trim1 + trim5

the span of each estimator, whether its verdict at 0.05 changes, the excess
span, and then the trial-level inference the paper reports: one value per
trial, Wilcoxon signed-rank paired by trial, and a cluster bootstrap over
trials for the pooled verdict-change rates. The same is done for the
negative controls. The ROC quantities are reported alongside.

Prints a report and writes results/v2_summary.json and results/v2_trial_level.csv.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
MENUS = {
    "strict": ["recorded", "log_shift", "normal_scores"],
    "nondel": ["recorded", "log_shift", "normal_scores", "winsor1", "winsor5"],
    "all": ["recorded", "log_shift", "normal_scores", "winsor1", "winsor5", "trim1", "trim5"],
}
B_BOOT = 5000
SEED = 20261031


def span(df: pd.DataFrame, est: str, specs: list[str]) -> pd.Series:
    p = df[[f"p_{est}_{s}" for s in specs]]
    return np.log10(p.max(axis=1)) - np.log10(p.min(axis=1))


def flips(df: pd.DataFrame, est: str, specs: list[str]) -> pd.Series:
    p = df[[f"p_{est}_{s}" for s in specs]] < 0.05
    return p.any(axis=1) & ~p.all(axis=1)


def add_menu_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for m, specs in MENUS.items():
        for est in ("t", "mw"):
            df[f"span_{est}_{m}"] = span(df, est, specs)
            df[f"flip_{est}_{m}"] = flips(df, est, specs)
        df[f"excess_{m}"] = df[f"span_t_{m}"] - df[f"span_mw_{m}"]
    return df


def cluster_boot(df: pd.DataFrame, col: str, rng: np.random.Generator) -> tuple[float, float, float]:
    trials = df.trial.unique()
    groups = {t: df.loc[df.trial == t, col].to_numpy(float) for t in trials}
    est = float(df[col].mean())
    boots = []
    for _ in range(B_BOOT):
        pick = rng.choice(trials, size=len(trials), replace=True)
        v = np.concatenate([groups[t] for t in pick])
        boots.append(v.mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return est, float(lo), float(hi)


def block(df: pd.DataFrame, name: str, rng: np.random.Generator, out: dict) -> None:
    print(f"\n#### {name}: {len(df)} comparisons from {df.trial.nunique()} trials")
    top = df.trial.value_counts()
    print(f"     largest three trials hold {top.head(3).sum() / len(df):.1%} of comparisons: {top.head(3).to_dict()}")
    res = {"n": int(len(df)), "n_trials": int(df.trial.nunique())}
    for m in MENUS:
        tl = df.groupby("trial").agg(span_t=(f"span_t_{m}", "median"), span_mw=(f"span_mw_{m}", "median"),
                                     excess=(f"excess_{m}", "median"), flip_t=(f"flip_t_{m}", "mean"), flip_mw=(f"flip_mw_{m}", "mean"),
                                     n=("trial", "size"))
        n_pos = int((tl.excess > 0).sum())
        w = stats.wilcoxon(tl.span_t, tl.span_mw, alternative="greater") if len(tl) > 1 else None
        ft, ft_lo, ft_hi = cluster_boot(df, f"flip_t_{m}", rng)
        fm, fm_lo, fm_hi = cluster_boot(df, f"flip_mw_{m}", rng)
        diff = df.groupby("trial").apply(lambda g: g[f"flip_t_{m}"].mean() - g[f"flip_mw_{m}"].mean())
        r = dict(menu=m,
                 trials_mean_span_larger=f"{n_pos}/{len(tl)}", wilcoxon_p=float(w.pvalue) if w else np.nan,
                 trials_with_t_flip=int((tl.flip_t > 0).sum()), trials_with_mw_flip=int((tl.flip_mw > 0).sum()),
                 flips_t=int(df[f"flip_t_{m}"].sum()), flips_mw=int(df[f"flip_mw_{m}"].sum()),
                 rate_t=ft, rate_t_ci=[ft_lo, ft_hi], rate_mw=fm, rate_mw_ci=[fm_lo, fm_hi],
                 median_trial_span_t=float(tl.span_t.median()), median_trial_span_mw=float(tl.span_mw.median()),
                 max_mw_span=float(df[f"span_mw_{m}"].max()))
        res[m] = r
        print(f"  [{m:6s}] mean-based span larger in {n_pos}/{len(tl)} trials (Wilcoxon one-sided p={r['wilcoxon_p']:.2e}); "
              f"trial-median spans t={r['median_trial_span_t']:.3f} mw={r['median_trial_span_mw']:.4f}; max MW span {r['max_mw_span']:.4f}")
        print(f"           verdict changes: t {r['flips_t']}/{len(df)} = {ft:.2%} [{ft_lo:.2%}, {ft_hi:.2%}]  "
              f"mw {r['flips_mw']}/{len(df)} = {fm:.2%} [{fm_lo:.2%}, {fm_hi:.2%}];  trials with any: t {r['trials_with_t_flip']}, mw {r['trials_with_mw_flip']}")
    out[name] = res


def main() -> None:
    rng = np.random.default_rng(SEED)
    df = add_menu_columns(pd.read_csv(RES / "comparisons_v2.csv"))
    out: dict = {}
    print(f"{len(df)} rows; occasions: {df.occasion.value_counts().to_dict()}")
    print(f"sources: {df.groupby(['source', 'occasion']).size().to_dict()}")
    # Exact invariance check of the ROC quantities and the Mann-Whitney p under the strict menu
    strict_mw = df[[f"p_mw_{s}" for s in MENUS['strict']]]
    print(f"\nMann-Whitney p identical across the strict menu in {(strict_mw.nunique(axis=1) == 1).mean():.1%} of rows "
          f"(max |Δlog10 p| = {df.span_mw_strict.max():.2e}); under winsorizing max MW span = {df.span_mw_nondel.max():.3f}")
    post = df[df.occasion == "post_first"]
    last = df[df.occasion == "post_last"]
    neg = df[df.occasion == "negative_control"]
    block(post, "post_first (primary)", rng, out)
    block(last, "post_last (sensitivity)", rng, out)
    block(neg, "negative_control", rng, out)
    # ROC-based test agreement with the Mann-Whitney verdict, and CvM where AUC is near 1/2
    print("\n#### ROC quantities (post_first)")
    print(f"  CvM permutation p < 0.05 in {(post.p_cvm_perm < 0.05).mean():.1%}; MW recorded p < 0.05 in {(post.p_mw_recorded < 0.05).mean():.1%}; "
          f"agreement {((post.p_cvm_perm < 0.05) == (post.p_mw_recorded < 0.05)).mean():.1%}")
    near = post[(post.auc - 0.5).abs() < 0.03]
    print(f"  comparisons with |AUC-1/2| < 0.03: {len(near)}; among them CvM perm p<0.05: {(near.p_cvm_perm < 0.05).sum()}, "
          f"Welch recorded p<0.05: {(near.p_t_recorded < 0.05).sum()}, MW p<0.05: {(near.p_mw_recorded < 0.05).sum()}")
    # Mechanism: SD ratio and tails
    post = post.assign(lsd=np.abs(np.log(post.sd_ratio)))
    for col, lab in [("lsd", "|log SD ratio|"), ("kurt", "excess kurtosis"), ("skew", "skewness")]:
        rho = stats.spearmanr(post[col], post.excess_nondel, nan_policy="omit")
        print(f"  Spearman(excess span nondel, {lab}) = {rho.statistic:+.2f} (comparison-level, descriptive)")
    q = pd.qcut(post.lsd, 4, labels=False, duplicates="drop")
    print("  flip rate (t, nondel) by quartile of |log SD ratio|:", post.groupby(q).flip_t_nondel.mean().round(4).to_dict())
    # Per-trial table
    tl = post.groupby(["trial", "source"]).agg(n=("trial", "size"), span_t=("span_t_nondel", "median"), span_mw=("span_mw_nondel", "median"),
                                              flip_t=("flip_t_nondel", "sum"), flip_mw=("flip_mw_nondel", "sum"), n_neg=("trial", lambda s: int((neg.trial == s.iloc[0]).sum()))).reset_index()
    tl.to_csv(RES / "v2_trial_level.csv", index=False)
    print("\n" + tl.to_string(index=False))
    with open(RES / "v2_summary.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
