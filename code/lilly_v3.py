"""The duloxetine stress-urinary-incontinence dataset: provenance, scale sensitivity and
heterogeneity by baseline severity, with valid inference.

Replaces lilly_v2.py after the 2026-11-01 audit.

What changed and why
  * Provenance. The file pools ten protocol codes and the loader keeps them. A participant
    flow is written: all records -> completion flag -> observed endpoint and severity
    category, by arm and by protocol. The analysed set is called what it is: a
    complete-case subset of a pooled dataset, not a randomized population.
  * Numerical policy. Values are rounded to nine significant figures on the pooled scale
    before any specification (rule 15), as in build_comparisons.
  * Menus. Strictly increasing (recorded, shifted log, normal scores); coarsening
    (winsorizing 1%, 5%; the two log-ratio floors, which merge all ratios below the floor);
    deleting (trimming). The strict-menu rank statistics are asserted identical.
  * Heterogeneity. The severity-label permutation of placement values is removed: it is
    not calibrated (audit diagnostic: 35% rejection under an equal-curve null). In its
    place, each stratum's probabilistic index is estimated with its two-sample U-statistic
    variance (DeLong et al. 1988), and heterogeneity is tested by (i) an omnibus Wald
    statistic on adjacent differences, chi-square with K-1 degrees of freedom, the
    interaction test of Harrison and Bosch (2025), and (ii) a pre-specified linear trend
    contrast across the ordered strata. Both are asymptotic. Within-stratum sharp-null
    permutation tests of the CvM departure are retained as tests of any effect.
  * Mean-based comparator. One common transformation is applied to the whole cohort
    (pooled across strata and arms, blind to arm) before the stratum-by-arm linear model,
    and the interaction is tested with an HC3 heteroskedasticity-robust Wald statistic.
    Stratum-specific mean differences carry Welch standard errors.

Writes results/lilly_v3.json and prints the tables.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from build_comparisons import SPECS, placement, roc_quantities, round_pooled  # noqa: E402
from sensitivity import make_trim, make_winsor, t_log_shift, t_normal_scores  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "data" / "lilly_baylor_yan.sas7bdat"
B = 1999
rng = np.random.default_rng(20261101)
STRICT = ("recorded", "log_shift", "normal_scores")


def load() -> tuple[pd.DataFrame, dict]:
    raw = pd.read_sas(DATA, format="sas7bdat", encoding="latin-1")
    raw["arm"] = np.where(raw.THERAPY.astype(str).str.contains("DULOX"), "duloxetine", "placebo")
    flow = {"records": len(raw), "protocols": int(raw.NEWPROTOCOL.nunique()),
            "by_arm_all": raw.arm.value_counts().to_dict(),
            "complete_by_arm": raw[raw.COMPLETE == 1].arm.value_counts().to_dict(),
            "complete_rate_by_arm": (raw.groupby("arm").COMPLETE.mean()).round(4).to_dict(),
            "allocation_by_protocol": pd.crosstab(raw.NEWPROTOCOL.astype(int), raw.arm).to_dict(),
            # records with the endpoint and the severity category observed, regardless of the
            # completion flag; reported beside the participant counts of the earlier analyses
            "endpoint_and_category": int((raw.PIEF.notna() & raw.STRATA.notna()).sum()),
            "horm50_observed": int(raw.HORM50.notna().sum())}
    d = raw[(raw.COMPLETE == 1) & raw.PIEF.notna() & raw.STRATA.notna()].copy()
    d["trt"] = d.arm == "duloxetine"
    d["stratum"] = d.STRATA.astype(int)
    d["protocol"] = d.NEWPROTOCOL.astype(int)
    ratio = 1 + d.PIEF / 100.0  # post/pre episode-frequency ratio; complete responders at 0
    flow.update({"analysed": len(d), "analysed_by_arm": d.arm.value_counts().to_dict(),
                 "pief_max": float(d.PIEF.max()),
                 "complete_responders": int((ratio <= 0).sum()),
                 # women whose ratio lies strictly between 0 and the floor, merged with the complete
                 # responders by the log-ratio specification floored at 0.10 and at 0.01
                 "near_complete_below_0.10": int(((ratio > 0) & (ratio < 0.10)).sum()),
                 "near_complete_below_0.01": int(((ratio > 0) & (ratio < 0.01)).sum()),
                 "analysed_by_stratum": d.stratum.value_counts().sort_index().to_dict(),
                 "analysed_by_protocol": d.protocol.value_counts().sort_index().to_dict(),
                 "stratum_by_arm": pd.crosstab(d.stratum, d.arm).to_dict()})
    return d[["PIEF", "trt", "stratum", "protocol", "AGE", "BMI"]], flow


def logratio(floor: float):
    def f(c, t):
        tr = lambda y: np.log(np.maximum(1.0 + y / 100.0, floor))
        return tr(c), tr(t)
    return f


MENU = [(n, f, kind) for n, f, kind in SPECS] + [("logratio0.01", logratio(0.01), "coarsen"), ("logratio0.1", logratio(0.10), "coarsen")]


def both(c, t):
    return (float(stats.ttest_ind(t, c, equal_var=False).pvalue), float(stats.mannwhitneyu(t, c, alternative="two-sided").pvalue))


def pi_u(c: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    """theta = P(T > C) + P(T = C)/2 and its two-sample U-statistic variance (DeLong)."""
    sc, st = np.sort(c), np.sort(t)
    U = (np.searchsorted(sc, t, "left") + 0.5 * (np.searchsorted(sc, t, "right") - np.searchsorted(sc, t, "left"))) / len(c)
    V = (len(t) - np.searchsorted(st, c, "right") + 0.5 * (np.searchsorted(st, c, "right") - np.searchsorted(st, c, "left"))) / len(t)
    theta = float(U.mean())
    var = float(np.var(U, ddof=1) / len(t) + np.var(V, ddof=1) / len(c))
    return theta, var


def stratum_table(d: pd.DataFrame) -> list[dict]:
    rows = []
    for s in sorted(d.stratum.unique()):
        sub = d[d.stratum == s]
        c, t = round_pooled(sub.PIEF[~sub.trt].to_numpy(float), sub.PIEF[sub.trt].to_numpy(float))
        r = dict(stratum=int(s), n_placebo=len(c), n_treated=len(t), sd_ratio=float(t.std(ddof=1) / c.std(ddof=1)),
                 skew=float(stats.skew(np.r_[c, t])), kurt=float(stats.kurtosis(np.r_[c, t])),
                 mean_diff=float(t.mean() - c.mean()), n_complete_responders=int((np.r_[c, t] <= -100).sum()))
        for name, fn, kind in MENU:
            x, z = fn(c, t)
            r[f"t_{name}"], r[f"mw_{name}"] = both(x, z)
        assert len({round(r[f"mw_{k}"], 12) for k in STRICT}) == 1, f"strict-menu rank p differs in stratum {s}"
        # benefit = lower PIEF, so the index of benefit compares -PIEF
        r["theta"], r["theta_var"] = pi_u(-c, -t)
        r["theta_lo"], r["theta_hi"] = r["theta"] - 1.96 * r["theta_var"] ** 0.5, r["theta"] + 1.96 * r["theta_var"] ** 0.5
        pv = placement(-c, -t); r["auc_check"], r["ks"], r["cvm"] = roc_quantities(pv)
        pooled = np.r_[-c, -t]; obs = r["cvm"]; cnt = 0
        for _ in range(B):
            p = rng.permutation(pooled)
            cnt += roc_quantities(placement(p[:len(c)], p[len(c):]))[2] >= obs
        r["p_cvm_perm"] = (cnt + 1) / (B + 1)
        def span(keys, est):
            v = [r[f"{est}_{k}"] for k in keys]; return float(np.log10(max(v)) - np.log10(min(v)))
        strict = list(STRICT); nondel = strict + ["winsor1", "winsor5"]; coars = nondel + ["logratio0.01", "logratio0.1"]
        r["t_span_strict"], r["mw_span_strict"] = span(strict, "t"), span(strict, "mw")
        r["t_span_nondel"], r["mw_span_nondel"] = span(nondel, "t"), span(nondel, "mw")
        r["t_span_nondel_floors"], r["mw_span_nondel_floors"] = span(coars, "t"), span(coars, "mw")
        rows.append(r)
    return rows


def heterogeneity(rows: list[dict]) -> dict:
    th = np.array([r["theta"] for r in rows]); v = np.array([r["theta_var"] for r in rows]); K = len(th)
    C = np.zeros((K - 1, K))
    for i in range(K - 1):
        C[i, i], C[i, i + 1] = -1.0, 1.0
    Vm = np.diag(v)
    q = float((C @ th) @ np.linalg.solve(C @ Vm @ C.T, C @ th))
    scores = np.arange(1, K + 1) - (K + 1) / 2.0
    slope = float(np.sum(scores * th) / np.sum(scores ** 2))
    se = float(np.sqrt(np.sum(scores ** 2 * v)) / np.sum(scores ** 2))
    z = slope / se
    return dict(omnibus_wald=q, omnibus_df=K - 1, omnibus_p=float(stats.chi2.sf(q, K - 1)),
                trend_slope_per_stratum=slope, trend_se=se, trend_z=z,
                trend_p_two_sided=float(2 * stats.norm.sf(abs(z))), trend_p_increasing=float(stats.norm.sf(z)))


def cohort_transform(y: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Apply one specification to the WHOLE cohort (all strata and arms, blind to arm).

    Returns the transformed values and a row mask (all True except for trimming).
    Data-dependent constants (shift, percentiles, ranks) are computed once on the cohort.
    """
    mask = np.ones(len(y), bool)
    if name == "recorded":
        return y, mask
    if name == "log_shift":
        shift = 0.0
        if y.min() <= 0:
            shift = -y.min() + 0.01 * max(np.ptp(y), 1.0)
        return np.log(y + shift), mask
    if name == "normal_scores":
        return stats.norm.ppf(stats.rankdata(y) / (len(y) + 1)), mask
    if name.startswith("winsor"):
        pct = float(name[6:]); lo, hi = np.percentile(y, [pct, 100 - pct]); return np.clip(y, lo, hi), mask
    if name.startswith("trim"):
        pct = float(name[4:]); lo, hi = np.percentile(y, [pct, 100 - pct]); mask = (y >= lo) & (y <= hi); return y, mask
    if name.startswith("logratio"):
        floor = float(name[8:]); return np.log(np.maximum(1.0 + y / 100.0, floor)), mask
    raise ValueError(name)


def hc3_interaction(d: pd.DataFrame, name: str) -> dict:
    """Stratum x arm linear model on a COMMON transformation of the whole cohort; HC3 Wald test of interaction."""
    y_all = d.PIEF.to_numpy(float)
    y, keep = cohort_transform(y_all, name)
    keep &= np.isfinite(y)
    y = y[keep]; trt = d.trt.to_numpy(float)[keep]; st = d.stratum.to_numpy()[keep]
    strata = sorted(np.unique(st))
    S = np.column_stack([(st == s).astype(float) for s in strata])
    X = np.column_stack([S, S * trt[:, None]])            # stratum means and stratum-specific arm effects
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    meat = (X * ((e / (1 - h)) ** 2)[:, None]).T @ X
    V = XtX_inv @ meat @ XtX_inv
    K = len(strata)
    eff = beta[K:]; Veff = V[K:, K:]
    C = np.zeros((K - 1, K))
    for i in range(K - 1):
        C[i, i], C[i, i + 1] = -1.0, 1.0
    q = float((C @ eff) @ np.linalg.solve(C @ Veff @ C.T, C @ eff))
    return dict(spec=name, p_interaction_hc3=float(stats.chi2.sf(q, K - 1)), wald=q,
                effects={int(s): float(eff[i]) for i, s in enumerate(strata)},
                effect_se={int(s): float(np.sqrt(Veff[i, i])) for i, s in enumerate(strata)},
                n_used=int(keep.sum()))


def main() -> None:
    d, flow = load()
    print("FLOW:", json.dumps(flow, indent=1, default=str)[:1500])
    rows = stratum_table(d)
    tab = pd.DataFrame(rows).set_index("stratum")
    pd.set_option("display.width", 250)
    cols = ["n_placebo", "n_treated", "sd_ratio", "kurt", "theta", "theta_lo", "theta_hi", "p_cvm_perm", "t_recorded", "t_winsor1", "t_log_shift", "t_logratio0.01", "t_normal_scores", "mw_recorded", "mw_logratio0.1",
            "t_span_strict", "t_span_nondel", "t_span_nondel_floors", "mw_span_strict", "mw_span_nondel", "mw_span_nondel_floors"]
    print(tab[cols].to_string(float_format=lambda v: f"{v:.3g}"))
    het = heterogeneity(rows)
    print("\nHeterogeneity of the probabilistic index across strata (two-arm U-statistic variances):")
    print(f"  omnibus Wald = {het['omnibus_wald']:.2f} on {het['omnibus_df']} df, p = {het['omnibus_p']:.4f}")
    print(f"  linear trend: slope {het['trend_slope_per_stratum']:+.4f} per stratum (SE {het['trend_se']:.4f}), z = {het['trend_z']:.2f}, two-sided p = {het['trend_p_two_sided']:.4f}, one-sided (increasing) p = {het['trend_p_increasing']:.4f}")
    inter = [hc3_interaction(d, n) for n, fn, kind in MENU]
    print("\nMean-based stratum x arm interaction, COMMON transformation of the whole cohort, HC3 Wald:")
    for r in inter:
        eff = ", ".join(f"{s}:{e:+.3g} ({r['effect_se'][s]:.2g})" for s, e in r["effects"].items())
        print(f"  {r['spec']:14s} p = {r['p_interaction_hc3']:.3g}   effects (SE) {eff}")
    neg = {}
    for col in ("AGE", "BMI"):
        dd = d.dropna(subset=[col]); out = []
        for s in sorted(dd.stratum.unique()):
            q = dd[dd.stratum == s]; c, t = round_pooled(q[col][~q.trt].to_numpy(float), q[col][q.trt].to_numpy(float))
            out.append({n: both(*fn(c, t)) for n, fn, kind in MENU[:5]})
        neg[col] = out
        print(f"\nNegative control {col}: Welch p by stratum and spec:", [{k: round(v[0], 3) for k, v in r.items()} for r in out])
    with open(ROOT / "results" / "lilly_v3.json", "w") as fh:
        json.dump(dict(flow=flow, strata=rows, heterogeneity=het, interaction=inter, negative_controls=neg,
                       menu=[(n, kind) for n, _, kind in MENU], B=B), fh, indent=1, default=float)


if __name__ == "__main__":
    main()
