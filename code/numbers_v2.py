"""Write every number the merged manuscript quotes as a LaTeX macro (AOAS/numbers.tex).

Revised 2026-11-01 after the audit:
  * excess-span correlations use the excess (Welch span minus Mann-Whitney span);
  * the simulation null reference is the n = 200 cell, with the range over n and the
    equality-of-distribution cells of the other families as separate macros;
  * the Lilly macros come from lilly_v3.json (participant flow, two-arm U-statistic
    heterogeneity tests, common-scale HC3 interaction, menus split);
  * the registry macros come from the endpoint codebook (primary_measures_v3.csv) with
    measure-level and trial-level denominators named separately;
  * validation rates of the heterogeneity tests (validate_pi_wald.csv);
  * sign-change counts of the mean difference across the menus, where stored.

Macro names are letters only. Nothing in the manuscript is typed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT.parent / "AOAS" / "numbers.tex"
OUT.parent.mkdir(exist_ok=True)
macros: dict[str, str] = {}
STRICT = ["recorded", "log_shift", "normal_scores"]
NONDEL = STRICT + ["winsor1", "winsor5"]


def m(name: str, value, fmt: str = "{}") -> None:
    key = "".join(ch for ch in name if ch.isalpha())
    macros[key] = fmt.format(value)


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}"


def pval(p: float) -> str:
    if p < 1e-4:
        e = int(np.floor(np.log10(p)))
        return f"{p / 10 ** e:.1f} \\times 10^{{{e}}}"
    return f"{p:.3f}" if p >= 0.001 else f"{p:.4f}"


def comma(n) -> str:
    return f"{int(n):,}"


def span(df, est, specs):
    p = df[[f"p_{est}_{s}" for s in specs]]
    return np.log10(p.max(axis=1)) - np.log10(p.min(axis=1))


def flips(df, est, specs):
    s = df[[f"p_{est}_{s}" for s in specs]] < 0.05
    return s.any(axis=1) & ~s.all(axis=1)


# ---------------------------------------------------------------- trials
summ = json.load(open(RES / "v2_summary.json"))
cmp = pd.read_csv(RES / "comparisons_v2.csv")
trials = pd.read_csv(RES / "trials_v2.csv")
post = summ["post_first (primary)"]; neg = summ["negative_control"]; last = summ["post_last (sensitivity)"]
m("nTrialsPost", post["n_trials"]); m("nCmpPost", comma(post["n"]))
m("nTrialsNeg", neg["n_trials"]); m("nCmpNeg", comma(neg["n"]))
m("nCmpLast", comma(last["n"])); m("nTrialsLast", last["n_trials"])
m("nTrialsManifested", len(trials)); m("nTrialsExcluded", int((trials.status == "excluded").sum()))
m("nTrialsNegOnly", int((trials.status == "negative_control_only").sum()))
m("nTrialsPDS", int(trials[(trials.source == "PDS") & (trials.n_post > 0)].shape[0]))
m("nTrialsDryad", int(trials[(trials.source == "Dryad") & (trials.n_post > 0)].shape[0]))
pf = cmp[cmp.occasion == "post_first"].copy()
m("shareTopThree", pct(pf.trial.value_counts().head(3).sum() / len(pf), 0))
for menu, tag in (("strict", "Strict"), ("nondel", "Nondel"), ("all", "All")):
    for block, btag in ((post, "Post"), (neg, "Neg"), (last, "Last")):
        r = block[menu]
        m(f"larger{tag}{btag}", r["trials_mean_span_larger"].replace("/", " of "))
        m(f"wilcoxon{tag}{btag}", pval(r["wilcoxon_p"]))
        m(f"flipsT{tag}{btag}", r["flips_t"]); m(f"flipsMW{tag}{btag}", r["flips_mw"])
        m(f"rateT{tag}{btag}", pct(r["rate_t"], 1)); m(f"rateTlo{tag}{btag}", pct(r["rate_t_ci"][0], 1)); m(f"rateThi{tag}{btag}", pct(r["rate_t_ci"][1], 1))
        m(f"rateMW{tag}{btag}", pct(r["rate_mw"], 2)); m(f"rateMWlo{tag}{btag}", pct(r["rate_mw_ci"][0], 2)); m(f"rateMWhi{tag}{btag}", pct(r["rate_mw_ci"][1], 2))
        m(f"trialsFlipT{tag}{btag}", r["trials_with_t_flip"]); m(f"trialsFlipMW{tag}{btag}", r["trials_with_mw_flip"])
        m(f"medSpanT{tag}{btag}", f"{r['median_trial_span_t']:.2f}"); m(f"medSpanMW{tag}{btag}", f"{r['median_trial_span_mw']:.3f}")
        m(f"maxSpanMW{tag}{btag}", f"{r['max_mw_span']:.3f}")
roles = pf.role.value_counts()
for k, v in roles.items():
    m(f"nRole{k.replace('_', ' ').title().replace(' ', '')}", int(v))
for menu, specs in (("Strict", STRICT), ("Nondel", NONDEL)):
    f = flips(pf, "t", specs); tr = f.groupby(pf.trial).mean()
    m(f"rateTequalTrial{menu}Post", pct(tr.mean(), 1))
    loo = [f[pf.trial != t].mean() for t in pf.trial.unique()]
    m(f"rateTlooMin{menu}Post", pct(min(loo), 1)); m(f"rateTlooMax{menu}Post", pct(max(loo), 1))
pf["excess"] = span(pf, "t", NONDEL) - span(pf, "mw", NONDEL)
pf["lsd"] = np.abs(np.log(pf.sd_ratio))
for col, tag in (("lsd", "SD"), ("kurt", "Kurt"), ("skew", "Skew")):
    rho = stats.spearmanr(pf[col], pf.excess, nan_policy="omit"); m(f"rho{tag}", f"{rho.statistic:+.2f}")
q = pd.qcut(pf.lsd, 4, labels=False, duplicates="drop"); fq = flips(pf, "t", NONDEL).groupby(q).mean()
m("flipQone", pct(fq.iloc[0], 1)); m("flipQfour", pct(fq.iloc[-1], 1))
if "d_recorded" in pf.columns:
    for menu, specs in (("Strict", STRICT), ("Nondel", NONDEL)):
        d = pf[[f"d_{s}" for s in specs]]
        sc = (np.sign(d).nunique(axis=1) > 1)
        m(f"signChanges{menu}Post", int(sc.sum())); m(f"signChanges{menu}PostPct", pct(sc.mean(), 1))
        m(f"signChanges{menu}PostTrials", int(pf.trial[sc].nunique()))
    dn = cmp[cmp.occasion == "negative_control"]
    sc = (np.sign(dn[[f"d_{s}" for s in STRICT]]).nunique(axis=1) > 1)
    m("signChangesStrictNeg", int(sc.sum())); m("signChangesStrictNegPct", pct(sc.mean(), 1))
cost = pf[pf.trial == "Colorec_Allianc_1994_201"]
for _, r in cost.iterrows():
    sp = np.log10(max(r[f"p_t_{s}"] for s in NONDEL)) - np.log10(min(r[f"p_t_{s}"] for s in NONDEL))
    m(f"cost{r['variable']}SpanT", f"{sp:.1f}"); m(f"cost{r['variable']}Skew", f"{r['skew']:.1f}")
    m(f"cost{r['variable']}Theta", f"{r['auc']:.3f}")
m("costMaxSpanMW", f"{max(np.log10(max(r[f'p_mw_{s}'] for s in NONDEL)) - np.log10(min(r[f'p_mw_{s}'] for s in NONDEL)) for _, r in cost.iterrows()):.2f}")
flu = pf[pf.variable == "postvax.B.Brisbane"].iloc[0]
m("fluSpanTstrict", f"{np.log10(max(flu[f'p_t_{s}'] for s in STRICT)) - np.log10(min(flu[f'p_t_{s}'] for s in STRICT)):.0f}")
m("fluSpanMWnondel", f"{np.log10(max(flu[f'p_mw_{s}'] for s in NONDEL)) - np.log10(min(flu[f'p_mw_{s}'] for s in NONDEL)):.3f}")
# Kernel counterexample of the supplement: Gaussian kernel, Silverman's normal-reference bandwidth
# h = 1.06 s n^(-1/5), control outcomes {1,2,3,4,8}, treated outcome 3.5, recorded and log scales.
def _ksmooth(c, y):
    c = np.asarray(c, float); h = 1.06 * c.std(ddof=1) * len(c) ** -0.2
    return float(np.mean(1 - stats.norm.cdf((y - c) / h)))
m("kernelRaw", f"{_ksmooth([1, 2, 3, 4, 8], 3.5):.4f}"); m("kernelLog", f"{_ksmooth(np.log([1, 2, 3, 4, 8]), np.log(3.5)):.4f}")
near = pf[(pf.auc - 0.5).abs() < 0.03]
m("nNearHalf", len(near)); m("nNearHalfCvm", int((near.p_cvm_perm < 0.05).sum()))
m("agreeCvmMW", pct(((pf.p_cvm_perm < 0.05) == (pf.p_mw_recorded < 0.05)).mean(), 1))
m("rejCvmPost", pct((pf.p_cvm_perm < 0.05).mean(), 1)); m("rejMWPost", pct((pf.p_mw_recorded < 0.05).mean(), 1))
pl = cmp[cmp.occasion.isin(["post_first", "post_last"])]
key = ["trial", "endpoint", "contrast"]
w = pl.pivot_table(index=key, columns="occasion", values=["n_a", "n_b"], aggfunc="first").dropna()
m("nMatchedFirstLast", len(w))
m("nLastLarger", int(((w[("n_a", "post_last")] + w[("n_b", "post_last")]) > (w[("n_a", "post_first")] + w[("n_b", "post_first")])).sum()))

# ---------------------------------------------------------------- Lilly (v3)
L = json.load(open(RES / "lilly_v3.json"))
F = L["flow"]
m("lillyRecords", comma(F["records"])); m("lillyProtocols", F["protocols"])
m("lillyRecDulox", comma(F["by_arm_all"]["duloxetine"])); m("lillyRecPlacebo", comma(F["by_arm_all"]["placebo"]))
m("lillyCompleteDulox", comma(F["complete_by_arm"]["duloxetine"])); m("lillyCompletePlacebo", comma(F["complete_by_arm"]["placebo"]))
m("lillyCompleteRateDulox", pct(F["complete_rate_by_arm"]["duloxetine"], 1)); m("lillyCompleteRatePlacebo", pct(F["complete_rate_by_arm"]["placebo"], 1))
m("lillyAnalysed", comma(F["analysed"])); m("lillyNTreated", comma(F["analysed_by_arm"]["duloxetine"])); m("lillyNPlacebo", comma(F["analysed_by_arm"]["placebo"]))
m("lillyNTotal", comma(F["analysed"]))
m("lillyEndpointCategory", comma(F["endpoint_and_category"]))
m("lillyPIEFMax", comma(round(F["pief_max"]))); m("lillyNearCompleteTen", comma(F["near_complete_below_0.10"]))
# Non-collapsibility example of the discussion (exact, by numerical integration): two equal
# strata, control N(0,1) and N(6,1), treated N(0.5,1) and N(6.5,1); the within-stratum index is
# Phi(0.5/sqrt 2) in both strata, the marginal index averages over all four stratum pairings.
from scipy.stats import norm as _norm
_d, _gap = 0.5, 6.0
m("collapseWithin", f"{_norm.cdf(_d / np.sqrt(2)):.3f}")
m("collapseMarginal", f"{0.5 * _norm.cdf(_d / np.sqrt(2)) + 0.25 * (_norm.cdf((_d + _gap) / np.sqrt(2)) + _norm.cdf((_d - _gap) / np.sqrt(2))):.3f}")
# participant counts REPORTED by the earlier analyses of the same source (literature constants,
# not computed here): Zhang, Zhao and Tubbs (2011, Section 6) and Meng and Tubbs (2020, Application)
m("zhangReportedN", comma(4940)); m("mengTubbsReportedN", comma(2200))
for s in L["strata"]:
    k = s["stratum"]; tag = ["", "One", "Two", "Three", "Four"][k]
    m(f"lillyN{tag}", comma(s["n_placebo"] + s["n_treated"]))
    m(f"lillyPI{tag}", f"{s['theta']:.3f}"); m(f"lillyPIlo{tag}", f"{s['theta_lo']:.3f}"); m(f"lillyPIhi{tag}", f"{s['theta_hi']:.3f}")
    for spec, st in (("recorded", "rec"), ("winsor1", "w"), ("winsor5", "wFive"), ("log_shift", "logshift"), ("logratio0.01", "logratio"), ("logratio0.1", "logratioTen"), ("normal_scores", "ns"), ("trim5", "trimFive")):
        m(f"lillyT{st}{tag}", pval(s[f"t_{spec}"]))
    m(f"lillyMW{tag}", pval(s["mw_recorded"])); m(f"lillyMWlogratioTen{tag}", pval(s["mw_logratio0.1"]))
    m(f"lillyCvmP{tag}", pval(s["p_cvm_perm"]))
    m(f"lillySpanTstrict{tag}", f"{s['t_span_strict']:.1f}"); m(f"lillySpanTnondel{tag}", f"{s['t_span_nondel']:.1f}"); m(f"lillySpanTfloors{tag}", f"{s['t_span_nondel_floors']:.1f}")
    m(f"lillySpanMWstrict{tag}", f"{s['mw_span_strict']:.3f}"); m(f"lillySpanMWnondel{tag}", f"{s['mw_span_nondel']:.3f}"); m(f"lillySpanMWfloors{tag}", f"{s['mw_span_nondel_floors']:.3f}")
    m(f"lillySD{tag}", f"{s['sd_ratio']:.2f}"); m(f"lillyKurt{tag}", f"{s['kurt']:.0f}"); m(f"lillyMeanDiff{tag}", f"{s['mean_diff']:+.1f}")
m("lillyCompleteResponders", comma(sum(s["n_complete_responders"] for s in L["strata"])))
H = L["heterogeneity"]
m("lillyWald", f"{H['omnibus_wald']:.2f}"); m("lillyWaldDf", H["omnibus_df"]); m("lillyWaldP", pval(H["omnibus_p"]))
m("lillyTrendSlope", f"{H['trend_slope_per_stratum']:+.3f}"); m("lillyTrendSE", f"{H['trend_se']:.3f}"); m("lillyTrendZ", f"{H['trend_z']:.2f}")
m("lillyTrendP", pval(H["trend_p_two_sided"])); m("lillyTrendPinc", pval(H["trend_p_increasing"]))
I = {r["spec"]: r for r in L["interaction"]}
m("lillyInterRec", pval(I["recorded"]["p_interaction_hc3"])); m("lillyInterLog", pval(I["logratio0.01"]["p_interaction_hc3"])); m("lillyInterLogshift", pval(I["log_shift"]["p_interaction_hc3"]))
m("lillyInterMin", pval(min(r["p_interaction_hc3"] for r in L["interaction"]))); m("lillyInterMax", pval(max(r["p_interaction_hc3"] for r in L["interaction"])))
m("lillyEffOneRec", f"{I['recorded']['effects']['1']:+.1f}"); m("lillyEffOneRecSE", f"{I['recorded']['effect_se']['1']:.0f}")
m("lillyEffOneLog", f"{I['logratio0.01']['effects']['1']:+.2f}"); m("lillyEffOneLogSE", f"{I['logratio0.01']['effect_se']['1']:.2f}")
m("lillyEffOneWone", f"{I['winsor1']['effects']['1']:+.1f}")
m("lillyEffFourRec", f"{I['recorded']['effects']['4']:+.1f}"); m("lillyEffFourRecSE", f"{I['recorded']['effect_se']['4']:.1f}")
m("lillyB", comma(L["B"]))
V = pd.read_csv(RES / "validate_pi_wald.csv")
m("valReps", comma(V.reps.iloc[0])); m("valWaldMin", f"{V.wald_rate.min():.1f}"); m("valWaldMax", f"{V.wald_rate.max():.1f}")
m("valTrendMin", f"{V.trend_rate.min():.1f}"); m("valTrendMax", f"{V.trend_rate.max():.1f}"); m("valScenarios", len(V))

# ---------------------------------------------------------------- simulation
sim = pd.read_csv(RES / "simulation_v2.csv")
nn = sim[(sim.dist == "normal") & (sim["shift"] == 0) & (sim.sd_ratio == 1.0)]
c200 = nn[nn.n == 200].iloc[0]
m("simNullT", pct(c200.flip_t_nondel, 1)); m("simNullTstrict", pct(c200.flip_t_strict, 1)); m("simNullMW", pct(c200.flip_mw_nondel, 1))
m("simNullTmin", pct(nn.flip_t_nondel.min(), 1)); m("simNullTmax", pct(nn.flip_t_nondel.max(), 1))
m("simNullTstrictMin", pct(nn.flip_t_strict.min(), 1)); m("simNullTstrictMax", pct(nn.flip_t_strict.max(), 1))
eq = sim[(sim["shift"] == 0) & (sim.sd_ratio == 1.0) & (sim.n == 200)]
for d in ("normal", "lognormal", "gamma", "contaminated"):
    row = eq[eq.dist == d].iloc[0]
    m(f"simNullFam{d.capitalize()}", pct(row.flip_t_nondel, 1)); m(f"simNullFam{d.capitalize()}Strict", pct(row.flip_t_strict, 1))
allnull = sim[(sim["shift"] == 0) & (sim.sd_ratio == 1.0)]
m("simNullAllMin", pct(allnull.flip_t_nondel.min(), 1)); m("simNullAllMax", pct(allnull.flip_t_nondel.max(), 1))
m("simMaxMWstrict", pct(sim.flip_mw_strict.max(), 2)); m("simMaxMWnondel", pct(sim.flip_mw_nondel.max(), 1))
m("simCells", len(sim)); m("simReps", comma(sim.reps.iloc[0]))
m("simCellsSpanLarger", int((sim.span_t_nondel_median > sim.span_mw_nondel_median).sum()))
r200 = sim[(sim.n == 200) & (sim["shift"] == 0.2)]
for d in ("normal", "lognormal", "gamma", "contaminated"):
    for vr, vt in ((1.0, "One"), (1.5, "OneFive"), (2.5, "TwoFive")):
        row = r200[(r200.dist == d) & (r200.sd_ratio == vr)].iloc[0]
        m(f"sim{d.capitalize()}{vt}T", pct(row.flip_t_nondel, 1)); m(f"sim{d.capitalize()}{vt}MW", pct(row.flip_mw_nondel, 1))
        m(f"simSpan{d.capitalize()}{vt}", f"{row.span_t_nondel_median:.2f}")
perm = pd.read_csv(RES / "simulation_v2_perm.csv")
for d in ("normal", "lognormal", "gamma", "contaminated"):
    for vr, vt in ((1.0, "One"), (1.5, "OneFive"), (2.5, "TwoFive")):
        row = perm[(perm.dist == d) & (perm["shift"] == 0) & (perm.sd_ratio == vr)].iloc[0]
        m(f"perm{d.capitalize()}{vt}Cvm", pct(row.cvm, 1)); m(f"perm{d.capitalize()}{vt}MW", pct(row.mw, 1)); m(f"perm{d.capitalize()}{vt}T", pct(row.t, 1))
        a = sim[(sim.dist == d) & (sim["shift"] == 0) & (sim.sd_ratio == vr) & (sim.n == 200)].iloc[0]
        m(f"sim{d.capitalize()}{vt}AUCnull", f"{a.auc_mean:.3f}")
m("permReps", comma(perm.reps.iloc[0])); m("permB", int(perm.B.iloc[0]))

# ---------------------------------------------------------------- registry (codebook v3)
reg = pd.read_csv(ROOT / "data" / "ctgov" / "primary_measures_v3.csv", low_memory=False)
m("regTrials", comma(reg.nct.nunique())); m("regMeasures", comma(len(reg)))
m("regContMeas", pct(reg.continuous.mean(), 1))
q = reg[reg.quant_cont]
m("regQuantMeas", pct(len(q) / len(reg), 1)); m("regQuantMeasN", comma(len(q)))
tq = reg.groupby("nct").quant_cont.any(); m("regQuantTrials", pct(tq.mean(), 1)); m("regQuantTrialsN", comma(tq.sum()))
st = q[q.method_class3 != "not stated"]
m("regStated", pct(len(st) / len(q), 1)); m("regStatedN", comma(len(st)))
vc = st.method_class3.value_counts(normalize=True)
m("regMean", pct(vc.get("mean-based", 0), 1)); m("regRank", pct(vc.get("rank-based", 0), 1)); m("regOther", pct(vc.get("other", 0), 1)); m("regBoth", pct(vc.get("both", 0), 1))
tr = st.groupby("nct").method_class3.agg(set)
m("regTrialsStated", comma(len(tr)))
m("regTrialAnyMean", pct(tr.map(lambda S: bool(S & {"mean-based", "both"})).mean(), 1)); m("regTrialAnyRank", pct(tr.map(lambda S: bool(S & {"rank-based", "both"})).mean(), 1))
m("regGeo", pct(q.log_scale_summary.mean(), 1))
et = reg[reg.continuous].endpoint_type.value_counts()
m("regTTEmeas", comma(et.get("time_to_event", 0))); m("regCountMeas", comma(et.get("count_or_rate", 0))); m("regPropMeas", comma(et.get("proportion", 0)))
m("regLogRankOld", comma(((reg.continuous) & (reg.method_class == "rank-based") & (reg.method_class3 == "survival")).sum()))
jpath = ROOT / "data" / "ctgov" / "registry_validation_judgments.csv"
if jpath.exists():
    J = pd.read_csv(jpath)
    for t, tag in (("quantitative", "Quant"), ("time_to_event", "TTE"), ("count_or_rate", "Count"), ("proportion", "Prop")):
        sub = J[J.endpoint_type == t]
        if len(sub):
            m(f"valReg{tag}N", len(sub)); m(f"valReg{tag}Correct", int(sub.correct.sum()))

with open(OUT, "w") as fh:
    fh.write("% Generated by Paper2_Reanalysis/code/numbers_v2.py. Do not edit.\n")
    for k, v in sorted(macros.items()):
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"wrote {len(macros)} macros to {OUT}")
