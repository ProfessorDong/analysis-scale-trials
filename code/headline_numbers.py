"""Single source of truth for every trial-evidence number in the manuscript.

Numbers in the manuscript were transcribed by hand and twice went stale when
the analysis was rerun, once in a way that survived into a compiled draft.
This writes them as LaTeX macros instead, together with the data files the
figures read, so the text, the figures and the analysis cannot disagree.

Run after analyze_pds.py.  Writes NaturePortfolio/data/numbers.tex and the
.dat files for figures 3 to 5.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from variable_classes import load_trials

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT.parent / "NaturePortfolio" / "data"

BOOT = 5000
SEED = 7


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trial comparisons, split into post- and pre-randomisation."""
    tr = load_trials()
    return tr[~tr.baseline].copy(), tr[tr.baseline].copy()


def trial_level(d: pd.DataFrame) -> pd.DataFrame:
    return d.groupby("trial").agg(
        t_span=("t_span", "median"), mw_span=("mw_span", "median"),
        excess=("excess_span", "median"),
        t_flip=("t_flips", "mean"), mw_flip=("mw_flips", "mean"),
    ).reset_index()


def cluster_boot(d: pd.DataFrame, B: int = BOOT) -> dict:
    rng = np.random.default_rng(SEED)
    trials = d.trial.unique()
    by = {t: g for t, g in d.groupby("trial")}
    tf, mf, ex = np.empty(B), np.empty(B), np.empty(B)
    for b in range(B):
        pick = rng.choice(trials, size=len(trials), replace=True)
        s = pd.concat([by[t] for t in pick], ignore_index=True)
        tf[b], mf[b], ex[b] = s.t_flips.mean(), s.mw_flips.mean(), s.excess_span.median()
    q = lambda v: np.percentile(v, [2.5, 97.5])
    return {"t": (d.t_flips.mean(), *q(tf)), "mw": (d.mw_flips.mean(), *q(mf)),
            "diff": (d.t_flips.mean() - d.mw_flips.mean(), *q(tf - mf)),
            "excess": (d.excess_span.median(), *q(ex))}


def sci(p: float) -> str:
    """A p-value as LaTeX scientific notation.

    Emitting "7.5e-9" into math mode typesets as "7.5e minus 9": the hyphen
    becomes a binary minus with spaces around it.  This returns the
    \\times 10^{} form the manuscript should show.
    """
    s = f"{p:.1e}"
    mant, exp = s.split("e")
    return f"{mant} \\times 10^{{{int(exp)}}}"


def ecdf(v: np.ndarray) -> pd.DataFrame:
    v = np.sort(v[np.isfinite(v)])
    v = np.clip(v, 1e-4, None)
    return pd.DataFrame({"x": v, "F": np.arange(1, len(v) + 1) / len(v)})


def main() -> None:
    post, pre = load()
    pub = pd.read_csv(RES / "screen_public_filtered.csv")
    sim = pd.read_csv(RES / "simulation_spans.csv")
    tl = trial_level(post)
    cb = cluster_boot(post)
    ptl = trial_level(pre)
    w_pre = stats.wilcoxon(ptl.t_span, ptl.mw_span, alternative="greater")

    w_span = stats.wilcoxon(tl.t_span, tl.mw_span, alternative="greater")
    w_flip = stats.wilcoxon(tl.t_flip, tl.mw_flip, alternative="greater")

    vr = post[np.isfinite(post.var_ratio)]
    rho_vr = stats.spearmanr(vr.excess_span, vr.var_ratio)
    rho_auc = stats.spearmanr(post.excess_span, (post.auc - 0.5).abs())
    q = pd.qcut(vr.var_ratio, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    qt = vr.groupby(q, observed=True).agg(
        t=("t_flips", "mean"), mw=("mw_flips", "mean"), ex=("excess_span", "median"))

    null = sim[(sim.dist == "normal") & (sim["shift"] == 0)
               & (sim.var_ratio == 1.0)]
    m = {
        "NTrials": len(tl), "NCmp": f"{len(post):,}",
        "NPDS": post[post.source == "PDS"].trial.nunique(),
        "NDryad": post[post.source == "Dryad"].trial.nunique(),
        "NPublic": f"{len(pub):,}", "NTotal": f"{len(post) + len(pub):,}",
        "TopShare": f"{100 * post.trial.value_counts().head(3).sum() / len(post):.0f}",
        "MedT": f"{tl.t_span.median():.3f}", "MedMW": f"{tl.mw_span.median():.4f}",
        "NWin": int((tl.excess > 0).sum()),
        "PSpan": sci(w_span.pvalue),
        "PFlip": sci(w_flip.pvalue),
        "TFlipTrials": int((tl.t_flip > 0).sum()),
        "MWFlipTrials": int((tl.mw_flip > 0).sum()),
        "PoolT": f"{100 * cb['t'][0]:.2f}", "PoolTLo": f"{100 * cb['t'][1]:.2f}",
        "PoolTHi": f"{100 * cb['t'][2]:.2f}",
        "PoolMW": f"{100 * cb['mw'][0]:.2f}",
        "PoolMWLo": f"{100 * cb['mw'][1]:.2f}", "PoolMWHi": f"{100 * cb['mw'][2]:.2f}",
        "PoolDiff": f"{100 * cb['diff'][0]:.1f}",
        "PoolDiffLo": f"{100 * cb['diff'][1]:.1f}",
        "PoolDiffHi": f"{100 * cb['diff'][2]:.1f}",
        "ExcessMed": f"{cb['excess'][0]:.3f}", "ExcessLo": f"{cb['excess'][1]:.3f}",
        "ExcessHi": f"{cb['excess'][2]:.3f}",
        "FlipRatio": f"{cb['t'][0] / max(cb['mw'][0], 1e-12):.0f}",
        "NVR": f"{len(vr):,}", "RhoVR": f"{rho_vr.statistic:.2f}",
        "PRhoVR": sci(rho_vr.pvalue),
        "RhoAUC": f"{rho_auc.statistic:.2f}",
        "QoneT": f"{100 * qt.t.iloc[0]:.2f}", "QfourT": f"{100 * qt.t.iloc[3]:.2f}",
        "QoneEx": f"{qt.ex.iloc[0]:.3f}", "QfourEx": f"{qt.ex.iloc[3]:.3f}",
        "QmaxMW": f"{100 * qt.mw.max():.2f}",
        "NPre": len(pre), "NPreTrials": pre.trial.nunique(),
        "PPre": sci(w_pre.pvalue),
        "PreT": f"{100 * pre.t_flips.mean():.2f}",
        # One decimal, for the plain-language abstract.
        "PreTOne": f"{100 * pre.t_flips.mean():.1f}",
        "PreMW": f"{100 * pre.mw_flips.mean():.2f}",
        "PostT": f"{100 * post.t_flips.mean():.2f}",
        "PostMW": f"{100 * post.mw_flips.mean():.2f}",
        "NullT": f"{100 * null.t_flip_rate.mean():.1f}",
        "NullMW": f"{100 * null.mw_flip_rate.mean():.1f}",
        "SimMaxT": f"{100 * sim.t_flip_rate.max():.1f}",
        "SimMinT": f"{100 * sim.t_flip_rate.min():.1f}",
        "SimReps": f"{int(sim.reps.max()):,}".replace(",", "{,}"),
        "SimCells": int(len(sim)),
        # Largest binomial standard error anywhere in the grid, so the table
        # is not read to a precision the replicate count cannot support.
        "SimSE": f"{100 * sim.t_flip_se.max():.1f}",
        # The asymmetry itself, across the whole grid.
        "SimCellsAsym": f"{100 * (sim.t_span_median > sim.mw_span_median).mean():.0f}",
        "SimMeanTSpan": f"{sim.t_span_median.mean():.2f}",
        "SimMeanMWSpan": f"{sim.mw_span_median.mean():.3f}",
        "SimMaxMW": f"{100 * sim.mw_flip_rate.max():.1f}",
        "PubT": f"{100 * pub.t_flips.mean():.2f}",
        "PubMW": f"{100 * pub.mw_flips.mean():.2f}",
    }

    # Table 1: the comparison-level distribution, trials alone and pooled
    # with the public datasets.
    allc = pd.concat([post, pub], ignore_index=True)
    for tag, d in (("Tr", post), ("All", allc)):
        m[f"{tag}N"] = f"{len(d):,}".replace(",", "{,}")
        for est, col in (("T", "t_span"), ("MW", "mw_span")):
            m[f"{tag}Med{est}"] = f"{d[col].median():.3f}"
            m[f"{tag}Pninenine{est}"] = f"{np.percentile(d[col], 99):.2f}"
            m[f"{tag}Max{est}"] = f"{d[col].max():.1f}"
        m[f"{tag}FlipT"] = f"{100 * d.t_flips.mean():.2f}"
        m[f"{tag}FlipMW"] = f"{100 * d.mw_flips.mean():.2f}"
        m[f"{tag}ExPos"] = f"{100 * (d.excess_span > 0).mean():.1f}"
    # Table 2, cell by cell, at the condition the caption names.  Written as
    # macros for the same reason as everything else: the table was correct
    # only because it happened not to have been rerun since it was typed.
    tab = sim[(sim.n == 200) & (sim["shift"] == 0.2)]

    # Dispersion at fixed shape, against shape at fixed dispersion.  The
    # corrected design separates these for the first time; the earlier one
    # moved the mean along with the spread in two of the four families.
    def cell(dist, vr, col):
        r = tab[(tab.dist == dist) & (tab.var_ratio == vr)]
        return r[col].iloc[0]
    m["SimNormLoSpan"] = f"{cell('normal', 1.0, 't_span_median'):.2f}"
    m["SimNormHiSpan"] = f"{cell('normal', 2.5, 't_span_median'):.2f}"
    m["SimNormLoFlip"] = f"{100 * cell('normal', 1.0, 't_flip_rate'):.1f}"
    m["SimNormHiFlip"] = f"{100 * cell('normal', 2.5, 't_flip_rate'):.1f}"
    m["SimContLoSpan"] = f"{cell('contaminated', 1.0, 't_span_median'):.2f}"
    m["SimContLoFlip"] = f"{100 * cell('contaminated', 1.0, 't_flip_rate'):.1f}"
    m["SimLogLoSpan"] = f"{cell('lognormal', 1.0, 't_span_median'):.2f}"
    m["SimGamLoSpan"] = f"{cell('gamma', 1.0, 't_span_median'):.2f}"
    words = {1.0: "One", 1.5: "OneFive", 2.5: "TwoFive"}
    for dist in ["normal", "lognormal", "gamma", "contaminated"]:
        for ratio, w in words.items():
            row = tab[(tab.dist == dist) & (tab.var_ratio == ratio)]
            for est, col in (("T", "t_flip_rate"), ("MW", "mw_flip_rate")):
                key = f"Sim{dist.capitalize()}{w}{est}"
                m[key] = f"{100 * row[col].iloc[0]:.1f}" if len(row) else "--"

    # Robustness of the verdict-change rate to extreme observations.  The
    # largest spans in the trial data trace to a handful of implausible
    # values: one uric acid reading of 254,000 umol/L against a median of
    # 268 makes the raw t-test null by inflating one arm's variance 208-fold,
    # and winsorising it away then gives P = 1e-24.  Theory predicts heavier
    # tails should flip more often, so this is partly the mechanism working
    # as designed, but it is also partly uncorrected data error, and the two
    # cannot be separated by any rule we are willing to defend.  Both bounds
    # are therefore reported.
    kk = post["kurt"].fillna(0)
    heavy = post[kk > 50]
    for tag, sub in (("Fifty", post[kk <= 50]), ("Twenty", post[kk <= 20])):
        stl = sub.groupby("trial").excess_span.median()
        m[f"Sens{tag}FlipT"] = f"{100 * sub.t_flips.mean():.2f}"
        m[f"Sens{tag}FlipMW"] = f"{100 * sub.mw_flips.mean():.2f}"
        m[f"Sens{tag}N"] = f"{len(sub):,}".replace(",", "{,}")
        m[f"Sens{tag}Win"] = int((stl > 0).sum())
        m[f"Sens{tag}Trials"] = int(len(stl))
    pk = pre["kurt"].fillna(0)
    m["SensNegFifty"] = f"{100 * pre[pk <= 50].t_flips.mean():.2f}"
    m["SensNegTwenty"] = f"{100 * pre[pk <= 20].t_flips.mean():.2f}"
    m["HeavyN"] = int(len(heavy))
    m["HeavyPct"] = f"{100 * (kk > 50).mean():.0f}"
    m["HeavyFlip"] = f"{100 * heavy.t_flips.mean():.1f}"
    m["HeavyRest"] = f"{100 * post[kk <= 50].t_flips.mean():.1f}"

    # Spans on p-values that never leave the rejection region.  A span of 46
    # orders between 1e-24 and 1e-70 is arithmetically real and inferentially
    # empty, and the maximum-span row of Table 1 is made of these.
    deep = (post.t_p_min < 1e-10) & (~post.t_flips.astype(bool))
    m["DeepN"] = int(deep.sum())
    m["DeepPct"] = f"{100 * deep.mean():.1f}"
    deep_share = 100 * post.loc[deep, "t_span"].sum() / post.t_span.sum()
    m["DeepShare"] = f"{deep_share:.0f}"

    # Scope of the phenomenon: how much of an oncology trial's recorded data
    # is a continuous post-randomisation endpoint at all.  A referee will ask
    # whether the claim generalises, and the honest answer is bounded.
    yq = pd.read_csv(RES / "pds_yield_audit.csv")
    ycls = yq.reason.str.replace(r" \(.*", "", regex=True)
    frac = 100 * ycls.value_counts() / len(yq)
    yper = yq.assign(e=yq.reason.str.startswith("ELIGIBLE")).groupby("study").e.sum()
    m["ScopeCols"] = f"{len(yq):,}".replace(",", "{,}")
    m["ScopeTrials"] = int(yq.study.nunique())
    m["ScopeElig"] = f"{frac.get('ELIGIBLE', 0):.0f}"
    m["ScopeTTE"] = f"{frac.get('censored time-to-event', 0):.0f}"
    m["ScopeExp"] = f"{frac.get('exposure or dose', 0) + frac.get('duration, censored or exposure', 0):.0f}"
    m["ScopeZero"] = int((yper == 0).sum())

    OUT.mkdir(parents=True, exist_ok=True)

    # Panel 3a has logarithmic axes, on which a span of exactly zero has no
    # position.  Those comparisons are written out separately and counted
    # rather than dropped in silence: every one of them has a positive
    # mean-based span, so leaving them off understates the asymmetry.
    pos = post[post.mw_span > 0]
    pos[["t_span", "mw_span"]].rename(columns={"t_span": "t", "mw_span": "mw"}) \
        .to_csv(OUT / "spans_trials.dat", sep=" ", index=False)
    zero = post[post.mw_span <= 0]
    m["FigZeroRank"] = int(len(zero))
    m["FigPlotted"] = f"{len(pos):,}".replace(",", "{,}")
    m["FigZeroPosT"] = int((zero.t_span > 0).sum())
    m["PublicSets"] = int(pd.read_csv(RES / "screen_public_filtered.csv")
                          .groupby(["package", "item"]).ngroups)
    # The ECDF panel clips at 1e-4 so that non-positive excess spans have a
    # position on the log axis.  Report how many are affected.
    m["EcdfClipTr"] = int((post.excess_span <= 1e-4).sum())
    m["EcdfClipPub"] = int((pub.excess_span <= 1e-4).sum())
    m["FigBelowLine"] = int((post.excess_span <= 0).sum())
    ecdf(post.excess_span.to_numpy()).to_csv(OUT / "ecdf_trials.dat", sep=" ", index=False)
    ecdf(pub.excess_span.to_numpy()).to_csv(OUT / "ecdf_public.dat", sep=" ", index=False)
    # Figure 4 compares verdict-change rates across sources and between pre-
    # and post-randomisation measurements.  Both are clustered, so bare point
    # estimates invite orderings the data cannot support; each bar carries a
    # cluster bootstrap interval, resampling whole trials or whole datasets.
    def boot_rate(d, col, B=4000, seed=11):
        rng = np.random.default_rng(seed)
        cl = d.trial.unique()
        by = {t: g for t, g in d.groupby("trial")}
        v = np.empty(B)
        for b in range(B):
            pick = rng.choice(cl, size=len(cl), replace=True)
            v[b] = pd.concat([by[t] for t in pick], ignore_index=True)[col].mean()
        lo, hi = np.percentile(100 * v, [2.5, 97.5])
        return 100 * d[col].mean(), lo, hi

    pubc = pub.assign(trial=pub.package.astype(str) + "/" + pub.item.astype(str))

    def bars(rows, dest):
        recs = []
        for label, d in rows:
            r = {"src": label}
            for tag, col in (("t", "t_flips"), ("mw", "mw_flips")):
                est, lo, hi = boot_rate(d, col)
                r[tag] = est
                r[f"{tag}em"] = max(est - lo, 0.0)
                r[f"{tag}ep"] = max(hi - est, 0.0)
            recs.append(r)
        out = pd.DataFrame(recs)
        out.to_csv(OUT / dest, sep=" ", index=False, float_format="%.4f")
        return out

    fl = bars([("PDS", post[post.source == "PDS"]),
               ("Dryad", post[post.source == "Dryad"]),
               ("Public", pubc)], "flips.dat")
    nc = bars([("Before", pre), ("After", post)], "negcontrol.dat")

    # Long form for the dot-and-interval version of Figure 5.  A bar encodes
    # its value by length, so a rate near 0.1% draws a bar too short to see
    # beside one near 10%.  A marker encodes value by position and stays the
    # same size whatever the value, which is the only way to show two rates
    # differing seventyfold on one linear axis.  Event counts travel with the
    # rows so the figure can print them.
    def dots(rows, dest):
        recs = []
        for y, (label, d) in enumerate(reversed(rows)):
            et, lot, hit = boot_rate(d, "t_flips")
            em, lom, him = boot_rate(d, "mw_flips")
            recs.append({"y": y, "t": et, "tem": max(et - lot, 0.0),
                         "tep": max(hit - et, 0.0), "mw": em,
                         "mwem": max(em - lom, 0.0), "mwep": max(him - em, 0.0),
                         "km": int(d.t_flips.sum()), "kr": int(d.mw_flips.sum()),
                         "nt": len(d)})
        pd.DataFrame(recs).to_csv(OUT / dest, sep=" ", index=False,
                                  float_format="%.4f")

    dots([("PDS", post[post.source == "PDS"]),
          ("Dryad", post[post.source == "Dryad"]),
          ("Public", pubc)], "flips_dots.dat")
    dots([("Before", pre), ("After", post)], "negcontrol_dots.dat")

    # Supplementary figures.  The 108-condition grid, laid out as 12 rows
    # (distribution x dispersion ratio) by 9 columns (sample size x effect),
    # so the whole simulation is visible rather than the single slice Table 2
    # reports.  Written long, with integer row and column indices, for
    # pgfplots matrix plot.
    dists = ["normal", "lognormal", "gamma", "contaminated"]
    ratios, sizes, effects = [1.0, 1.5, 2.5], [50, 200, 800], [0.0, 0.2, 0.5]
    rows = []
    for r, (d, vr_) in enumerate((d, v) for d in dists for v in ratios):
        for c, (nn, dd) in enumerate((n_, e) for n_ in sizes for e in effects):
            cell = sim[(sim.dist == d) & (sim.var_ratio == vr_)
                       & (sim.n == nn) & (sim["shift"] == dd)]
            rows.append({"col": c, "row": r,
                         "t": 100 * cell.t_flip_rate.iloc[0],
                         "mw": 100 * cell.mw_flip_rate.iloc[0]})
    pd.DataFrame(rows).to_csv(OUT / "simgrid.dat", sep=" ", index=False,
                              float_format="%.3f")

    # Screening funnel: every numeric variable in the Project Data Sphere
    # trials, and the reason each was kept or dropped.
    yq2 = pd.read_csv(RES / "pds_yield_audit.csv")
    cls = yq2.reason.str.replace(r" \(.*", "", regex=True)
    order = ["identifier, date or flag", "categorical or too coarse",
             "censored time-to-event", "exposure or dose",
             "duration, censored or exposure"]
    labels = {"identifier, date or flag": "identifier, date or flag",
              "categorical or too coarse": "categorical or too coarse",
              "censored time-to-event": "censored time-to-event",
              "exposure or dose": "dose or exposure",
              "duration, censored or exposure": "duration variable"}
    remaining = len(yq2)
    fun = [{"step": "screened", "n": remaining, "drop": 0}]
    for k in order:
        d_ = int((cls == k).sum())
        remaining -= d_
        fun.append({"step": labels[k], "n": remaining, "drop": d_})
    pds_all = pd.read_csv(RES / "pds_results.csv")
    fun.append({"step": "comparisons formed", "n": len(pds_all), "drop": 0})
    fun.append({"step": "post-randomisation", "n": int((~pds_all.baseline).sum()),
                "drop": int(pds_all.baseline.sum())})
    fd = pd.DataFrame(fun)
    # Indexed rather than keyed by label: pgfplots symbolic coordinates do not
    # take quoted strings with spaces, and the labels are fixed by the code
    # rather than by the data, so they live in the figure.
    fd.insert(0, "idx", range(len(fd)))
    fd[["idx", "n", "drop"]].to_csv(OUT / "funnel.dat", sep=" ", index=False)
    m["FunnelComparisons"] = f"{len(pds_all):,}".replace(",", "{,}")
    m["FunnelPost"] = f"{int((~pds_all.baseline).sum()):,}".replace(",", "{,}")
    m["FunnelPre"] = int(pds_all.baseline.sum())
    m["FunnelStart"] = f"{len(yq2):,}".replace(",", "{,}")
    m["FunnelEligible"] = int((cls == "ELIGIBLE").sum())

    # Trial-level paired spans.  Every figure so far shows comparisons, but
    # the primary test is a Wilcoxon signed-rank on the trial-level values, so
    # the unit of inference was invisible.  Marker area encodes the number of
    # comparisons, which lets a reader check that the result is not carried by
    # the three trials holding 56% of them.
    per_t = post.groupby("trial").agg(
        k=("excess_span", "size"), t=("t_span", "median"),
        mw=("mw_span", "median")).reset_index()
    per_t[["t", "mw", "k"]].to_csv(OUT / "spans_by_trial.dat", sep=" ",
                                   index=False, float_format="%.6f")
    m["TrialsAbove"] = int((per_t.t > per_t.mw).sum())
    big3 = per_t.nlargest(3, "k")
    m["BigThreeMaxExcess"] = f"{(big3.t - big3.mw).max():.2f}"

    # Figure 4 panel a: the simulation curve for every generating family, at
    # the condition Table 2 reports.  Earlier versions plotted only normal and
    # lognormal, the two that rise monotonically, which made the panel look
    # like a cleaner confirmation than the grid actually gives.
    cur = tab.pivot_table(index="var_ratio", columns="dist", values="t_flip_rate") * 100
    cur["mwmax"] = (tab.pivot_table(index="var_ratio", columns="dist",
                                    values="mw_flip_rate") * 100).max(axis=1)
    cur.reset_index().rename(columns={"var_ratio": "vr"}).to_csv(
        OUT / "simcurve.dat", sep=" ", index=False, float_format="%.3f")

    # Figure 4 panel b: verdict-change rate by quartile of the SD ratio, with
    # a cluster bootstrap interval inside each quartile.
    qrecs = []
    for lab, g in zip(["Q1", "Q2", "Q3", "Q4"],
                      [vr[q == k] for k in ["Q1", "Q2", "Q3", "Q4"]]):
        r = {"src": lab}
        for tag, col in (("t", "t_flips"), ("mw", "mw_flips")):
            est, lo, hi = boot_rate(g, col)
            r[tag] = est
            r[f"{tag}em"] = max(est - lo, 0.0)
            r[f"{tag}ep"] = max(hi - est, 0.0)
        qrecs.append(r)
    pd.DataFrame(qrecs).to_csv(OUT / "quartiles.dat", sep=" ", index=False,
                               float_format="%.4f")

    # Panel 4b rests only on comparisons with an estimable dispersion ratio.
    # That statistic is computed in the Project Data Sphere pipeline and not in
    # the Dryad screen, so the panel covers 13 trials, not all 27, and saying
    # otherwise overstates its breadth.
    m["FigBTrials"] = int(vr.trial.nunique())
    m["FigBN"] = f"{len(vr):,}".replace(",", "{,}")
    m["SimCondN"] = int(tab.n.iloc[0])
    m["SimCondDelta"] = f"{tab['shift'].iloc[0]:.1f}"

    # Raw event counts.  The verdict-change ratio is the reciprocal of a very
    # small number: two rank-based changes in the trial set, so one event
    # either way moves it between 139-fold and 46-fold.  The counts are the
    # honest way to state it, and they are more striking than the ratio.
    m["FlipsMeanTr"] = int(post.t_flips.sum())
    m["FlipsRankTr"] = int(post.mw_flips.sum())
    m["FlipsMeanAll"] = int(allc.t_flips.sum())
    m["FlipsRankAll"] = int(allc.mw_flips.sum())
    m["FlipsMeanPre"] = int(pre.t_flips.sum())
    m["FlipsRankPre"] = int(pre.mw_flips.sum())

    m["FigFourAllN"] = f"{len(allc):,}".replace(",", "{,}")
    m["FigFourAllT"] = f"{100 * allc.t_flips.mean():.2f}"
    m["FigFourAllMW"] = f"{100 * allc.mw_flips.mean():.2f}"
    m["FigFourRatioAll"] = f"{allc.t_flips.mean() / max(allc.mw_flips.mean(), 1e-12):.0f}"
    m["PreOneIn"] = f"{1 / pre.t_flips.mean():.0f}"

    # Supplementary tables that enumerate the evidence.  Generated for the
    # same reason as the macros: a hand-maintained per-trial table silently
    # kept a trial that the pipeline had stopped producing.
    si = OUT.parent / "si"
    per = post.groupby(["source", "trial"]).agg(
        n=("n", "max"), k=("excess_span", "size"),
        t=("t_span", "median"), mw=("mw_span", "median"),
        tf=("t_flips", "mean"), mf=("mw_flips", "mean")).reset_index()
    per = per.sort_values(["source", "trial"])
    with open(si / "sm_table_trials.tex", "w") as fh:
        fh.write("% Generated by headline_numbers.py; do not edit.\n")
        for r in per.itertuples():
            name = str(r.trial).replace("_", r"\_")
            fh.write(f"{r.source} & {name} & {r.n} & {r.k} & {r.t:.3f} & "
                     f"{r.mw:.4f} & {100*r.tf:.2f} & {100*r.mf:.2f}\\\\\n")

    qrows = vr.groupby(q, observed=True).agg(
        n=("excess_span", "size"), vrm=("var_ratio", "median"),
        ex=("excess_span", "median"), tf=("t_flips", "mean"),
        mf=("mw_flips", "mean")).reset_index(drop=True)
    labels = ["Q1 most equal", "Q2", "Q3", "Q4 most unequal"]
    with open(si / "sm_table_quartiles.tex", "w") as fh:
        fh.write("% Generated by headline_numbers.py; do not edit.\n")
        for lab, r in zip(labels, qrows.itertuples()):
            fh.write(f"{lab} & {r.n} & {r.vrm:.3f} & {r.ex:.3f} & "
                     f"{100*r.tf:.2f} & {100*r.mf:.2f}\\\\\n")
        # booktabs needs the closing rule in the same token stream as the
        # rows; an \input that ends on \\ leaves \bottomrule stranded.
        fh.write("\\bottomrule\n")
    print(f"wrote {si/'sm_table_trials.tex'} ({len(per)} trials) and "
          f"{si/'sm_table_quartiles.tex'}")

    with open(si / "sm_table_sim.tex", "w") as fh:
        fh.write("% Generated by headline_numbers.py; do not edit.\n")
        for dist in ["normal", "lognormal", "gamma", "contaminated"]:
            for vr in (1.0, 1.5, 2.5):
                r = sim[(sim.dist == dist) & (sim.n == 200) & (sim.var_ratio == vr)]
                r = r.set_index("shift")
                t = " & ".join(f"{100 * r.t_flip_rate[d]:.1f}" for d in (0.0, 0.2, 0.5))
                w = " & ".join(f"{100 * r.mw_flip_rate[d]:.1f}" for d in (0.0, 0.2, 0.5))
                fh.write(f"{dist} & {vr:.1f} & {t} & {w}\\\\\n")
        fh.write("\\bottomrule\n")

    # Written last, so that macros added while building the figure data are
    # included.  Emitting it earlier silently dropped six of them.
    with open(OUT / "numbers.tex", "w") as fh:
        fh.write("% Generated by Paper2_Reanalysis/code/headline_numbers.py.\n"
                 "% Do not edit by hand; rerun the script instead.\n")
        for k, v in m.items():
            fh.write(f"\\newcommand{{\\num{k}}}{{{v}}}\n")

    print(f"{len(m)} macros -> {OUT/'numbers.tex'}")
    for k, v in m.items():
        print(f"  {k:14s} {v}")
    print("\nvariance-ratio quartiles (flip %, median excess span):")
    print((qt.assign(t=100 * qt.t, mw=100 * qt.mw)).round(3).to_string())
    print("\nfigure 4 panel a coordinates:")
    print("  " + " ".join(f"({r.src},{r.t:.2f})" for r in fl.itertuples()))
    print("  " + " ".join(f"({r.src},{r.mw:.2f})" for r in fl.itertuples()))
    print("figure 4 panel b: before %s/%s  after %s/%s"
          % (m["PreT"], m["PreMW"], m["PostT"], m["PostMW"]))
    print("figure 5 panel b: " + " ".join(
        f"({l},{100*t:.2f})" for l, t in zip(["Q1", "Q2", "Q3", "Q4"], qt.t)))
    print("                  " + " ".join(
        f"({l},{100*v:.2f})" for l, v in zip(["Q1", "Q2", "Q3", "Q4"], qt.mw)))


if __name__ == "__main__":
    main()
