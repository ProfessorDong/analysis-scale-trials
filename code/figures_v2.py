"""Data files for the merged paper's figures, generated from the v2 results.

Writes to AOAS/figures/data/. Every value drawn in a figure comes from here or
from numbers.tex; nothing is typed into a figure source.

  spans_cmp.dat, spans_trial.dat   Figure: paired spans (sample-preserving menu)
  rates.dat                        Figure: verdict-change rates with cluster bootstrap intervals
  simcurve.dat, quartiles.dat      Figure: mechanism (simulation, trials)
  lilly_spec.dat, lilly_roc_*.dat  Figure: the severity-stratified trial
  roc_illustration.dat             Figure: invariance illustration (lognormal arms)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from build_comparisons import placement  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT.parent / "AOAS" / "figures" / "data"
OUT.mkdir(parents=True, exist_ok=True)
NONDEL = ["recorded", "log_shift", "normal_scores", "winsor1", "winsor5"]
STRICT = NONDEL[:3]
rng = np.random.default_rng(20261031)
FLOOR = 1e-4   # plotting floor for a span of exactly zero on a log axis


def span(df, est, specs):
    p = df[[f"p_{est}_{s}" for s in specs]]
    return np.log10(p.max(axis=1)) - np.log10(p.min(axis=1))


def flips(df, est, specs):
    s = df[[f"p_{est}_{s}" for s in specs]] < 0.05
    return s.any(axis=1) & ~s.all(axis=1)


def boot_rate(df, col, B=5000):
    trials = df.trial.unique(); g = {t: df.loc[df.trial == t, col].to_numpy(float) for t in trials}
    est = df[col].mean(); bs = []
    for _ in range(B):
        pick = rng.choice(trials, len(trials), replace=True); bs.append(np.concatenate([g[t] for t in pick]).mean())
    lo, hi = np.percentile(bs, [2.5, 97.5]); return 100 * est, 100 * (est - lo), 100 * (hi - est)


def write(name, df):
    df.to_csv(OUT / name, sep=" ", index=False, float_format="%.6g")


cmp = pd.read_csv(RES / "comparisons_v2.csv")
for m, specs in (("nondel", NONDEL), ("strict", STRICT)):
    for est in ("t", "mw"):
        cmp[f"span_{est}_{m}"] = span(cmp, est, specs); cmp[f"flip_{est}_{m}"] = flips(cmp, est, specs)
post = cmp[cmp.occasion == "post_first"]; neg = cmp[cmp.occasion == "negative_control"]; last = cmp[cmp.occasion == "post_last"]

# ---- paired spans
write("spans_cmp.dat", pd.DataFrame({"mw": np.maximum(post.span_mw_nondel, FLOOR), "t": np.maximum(post.span_t_nondel, FLOOR)}))
tl = post.groupby("trial").agg(mw=("span_mw_nondel", "median"), t=("span_t_nondel", "median"), k=("trial", "size")).reset_index()
tl["mw"] = np.maximum(tl.mw, FLOOR); tl["t"] = np.maximum(tl.t, FLOOR)
write("spans_trial.dat", tl[["mw", "t", "k"]])

# ---- verdict-change rates (rows from the bottom up)
rows = []
for y, (df, m, lab) in enumerate([(neg, "nondel", "before randomization, sample-preserving"), (neg, "strict", "before randomization, strictly increasing"),
                                  (last, "nondel", "last visit, sample-preserving"), (post, "nondel", "first visit, sample-preserving"),
                                  (post, "strict", "first visit, strictly increasing")]):
    t, tem, tep = boot_rate(df, f"flip_t_{m}"); mw, mwem, mwep = boot_rate(df, f"flip_mw_{m}")
    rows.append(dict(y=y, t=t, tem=tem, tep=tep, mw=mw, mwem=mwem, mwep=mwep, km=int(df[f"flip_t_{m}"].sum()), kr=int(df[f"flip_mw_{m}"].sum()), nt=len(df), label=lab))
write("rates.dat", pd.DataFrame(rows))

# ---- mechanism: trial quartiles of |log SD ratio| (bootstrap by trial)
pf = post.assign(lsd=np.abs(np.log(post.sd_ratio))).dropna(subset=["lsd"])
pf["q"] = pd.qcut(pf.lsd, 4, labels=["Q1", "Q2", "Q3", "Q4"])
qrows = []
for q, g in pf.groupby("q", observed=True):
    t, tem, tep = boot_rate(g, "flip_t_nondel"); mw, mwem, mwep = boot_rate(g, "flip_mw_nondel")
    qrows.append(dict(src=q, t=t, tem=tem, tep=tep, mw=mw, mwem=mwem, mwep=mwep, n=len(g), ntr=g.trial.nunique()))
write("quartiles.dat", pd.DataFrame(qrows))
if (RES / "simulation_v2.csv").exists():
    sim = pd.read_csv(RES / "simulation_v2.csv"); s = sim[(sim.n == 200) & (sim["shift"] == 0.2)]
    sc = s.pivot(index="sd_ratio", columns="dist", values="flip_t_nondel") * 100
    sc["mwmax"] = s.groupby("sd_ratio").flip_mw_nondel.max() * 100
    write("simcurve.dat", sc.reset_index().rename(columns={"sd_ratio": "vr"}))

# ---- Lilly: specification curves per stratum and ROC curves
L = json.load(open(RES / "lilly_v3.json"))
specs = [("recorded", "recorded", "strict"), ("log_shift", "shifted log", "strict"), ("normal_scores", "normal scores", "strict"),
         ("winsor1", "winsorized 1%", "coarsen"), ("winsor5", "winsorized 5%", "coarsen"), ("logratio0.01", "log-ratio, floor 0.01", "coarsen"), ("logratio0.1", "log-ratio, floor 0.1", "coarsen"),
         ("trim1", "trimmed 1%", "delete"), ("trim5", "trimmed 5%", "delete")]
srows = []
for s_ in L["strata"]:
    for i, (k, lab, cls) in enumerate(specs):
        srows.append(dict(stratum=s_["stratum"], x=i, t=s_[f"t_{k}"], mw=s_[f"mw_{k}"], cls=cls, spec=lab.replace(" ", "~")))
write("lilly_spec.dat", pd.DataFrame(srows))
d = pd.read_sas(ROOT.parent / "data" / "lilly_baylor_yan.sas7bdat", format="sas7bdat", encoding="latin-1")
d = d[(d.COMPLETE == 1) & d.PIEF.notna() & d.STRATA.notna()]
d = d.assign(trt=d.THERAPY.astype(str).str.contains("DULOX"), stratum=d.STRATA.astype(int))
grid = np.linspace(0, 1, 201)
for st in sorted(d.stratum.unique()):
    sub = d[d.stratum == st]; c = -sub.PIEF[~sub.trt].to_numpy(float); t = -sub.PIEF[sub.trt].to_numpy(float)
    pv = placement(c, t); roc = np.searchsorted(np.sort(pv), grid, side="right") / len(pv)
    write(f"lilly_roc_{st}.dat", pd.DataFrame({"t": grid, "roc": roc}))

# ---- invariance illustration: lognormal arms as in the earlier Figure 1, with the ROC curve
muC, sigC, muT, sigT = 0.0, 0.5, -0.2, 1.0
yc = rng.lognormal(muC, sigC, 20000); yt = rng.lognormal(muT, sigT, 20000)
pv = placement(yc, yt); roc = np.searchsorted(np.sort(pv), grid, side="right") / len(pv)
write("roc_illustration.dat", pd.DataFrame({"t": grid, "roc": roc}))
xs = np.linspace(0.01, 6, 300)
from scipy import stats  # noqa: E402
write("dens_raw.dat", pd.DataFrame({"x": xs, "c": stats.lognorm.pdf(xs, sigC, scale=np.exp(muC)), "t": stats.lognorm.pdf(xs, sigT, scale=np.exp(muT))}))
xl = np.linspace(-3, 3, 300)
write("dens_log.dat", pd.DataFrame({"x": xl, "c": stats.norm.pdf(xl, muC, sigC), "t": stats.norm.pdf(xl, muT, sigT)}))
ill = dict(rawDiff=np.exp(muT + sigT**2 / 2) - np.exp(muC + sigC**2 / 2), logDiff=muT - muC,
           theta=stats.norm.cdf((muT - muC) / np.sqrt(sigT**2 + sigC**2)), muC=muC, sigC=sigC, muT=muT, sigT=sigT)
with open(OUT / "illustration_numbers.tex", "w") as fh:
    for k, v in ill.items():
        fh.write(f"\\newcommand{{\\ill{k}}}{{{v:+.2f}}}\n" if k in ("rawDiff", "logDiff") else f"\\newcommand{{\\ill{k}}}{{{v:.3f}}}\n" if k == "theta" else f"\\newcommand{{\\ill{k}}}{{{v:g}}}\n")
print("figure data written to", OUT)
