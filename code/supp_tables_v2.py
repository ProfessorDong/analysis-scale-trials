"""Generate the supplement's tables as LaTeX fragments in AOAS/supp_tables/.

  trials_table.tex      one row per manifested dataset: source, design, status,
                        comparisons, negative controls, median spans, verdict changes
  ledger_counts.tex     dispositions in the exclusion ledger
  sim_grid.tex          the full simulation grid (verdict-change rates by cell)
  sim_perm.tex          permutation rejection rates in the crossing sub-grid
  registry.tex          the registry census
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT.parent / "AOAS" / "supp_tables"
OUT.mkdir(exist_ok=True)
NONDEL = ["recorded", "log_shift", "normal_scores", "winsor1", "winsor5"]

NAMES = {
    "Breast_Allianc_2002_200": "CALGB 40101 (adjuvant breast)", "Breast_Allianc_2006_216": "ACOSOG Z1031 (neoadjuvant breast)",
    "Breast_Allianc_2008_158": "CALGB 40502 (metastatic breast)", "Breast_Allianc_2009_162": "CALGB 40603 (neoadjuvant breast, $2\\times2$)",
    "Colorec_Allianc_1994_201": "COST (laparoscopic vs open colectomy)", "Colorec_Allianc_2004_161": "N0147 (adjuvant colon)",
    "Colorec_Amgen_2004_310": "Amgen 20020408 (panitumumab, mCRC)", "Colorec_Amgen_2005_262": "PACCE 20040249 (mCRC)",
    "Colorec_Amgen_2006_263": "Amgen 20050181 (mCRC, 2nd line)", "Colorec_Amgen_2006_309": "PRIME 20050203 (mCRC, 1st line)",
    "HeadNe_Amgen_2007_265": "SPECTRUM 20050251 (head and neck)", "LungSm_Allianc_1998_261": "CALGB 9732 (small-cell lung)",
    "LungSm_Allianc_2007_218": "CALGB 30504 (small-cell lung)", "LungSm_Amgen_2002_266": "Amgen 20010145 (darbepoetin, SCLC)",
    "LungSm_EliLill_2011_287": "Lilly LY2510924 (SCLC, phase 2)", "Multiple_Allianc_2002_202": "N99C4 (smoking cessation)",
    "Multiple_Allianc_2002_213": "CALGB 80101 (adjuvant gastric)", "Multiple_Brigham_454": "VITAL knee-pain cohort ($2\\times2$)",
    "Pancrea_ClovisO_2010_186": "LEAP CO-101-001 (pancreatic)",
    "dryad_18931zdb7": "CHESS (hypertension, cluster)", "dryad_275d4": "Synbiotic sepsis prevention (India)",
    "dryad_59zw3r27n": "TERECO (COVID-19 telerehabilitation)", "dryad_8cz8w9gsn": "Caloric restriction, insulin sensitivity",
    "dryad_9cnp5hqtv": "Snoezelen (cerebral palsy)", "dryad_cv37539": "Influenza vaccine, children (HAI titer)",
    "dryad_k3j9kd5qd": "IV vs oral iron in pregnancy", "dryad_m0cfxpp37": "CLIMB (chlorhexidine, COPD)",
    "dryad_n8pk0p31r": "Maternal education (Uganda, cluster)", "dryad_pc866t1t6": "SMS breastfeeding support (Kenya)",
    "dryad_qnk98sfm2": "CHW home visits (South Africa, cluster)", "dryad_rg57q6m": "NTrainer (preterm oral stimulation)",
    "dryad_sj3tx96k5": "Adolescent weight loss (Egypt)", "dryad_sqv9s4ncs": "KAT-AKI (kidney action team)",
}


def esc(s: str) -> str:
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def span(df, est):
    p = df[[f"p_{est}_{s}" for s in NONDEL]]
    return np.log10(p.max(axis=1)) - np.log10(p.min(axis=1))


def flips(df, est):
    s = df[[f"p_{est}_{s}" for s in NONDEL]] < 0.05
    return s.any(axis=1) & ~s.all(axis=1)


def trials_table():
    tr = pd.read_csv(RES / "trials_v2.csv"); cmp = pd.read_csv(RES / "comparisons_v2.csv")
    cmp["st"], cmp["smw"], cmp["ft"], cmp["fmw"] = span(cmp, "t"), span(cmp, "mw"), flips(cmp, "t"), flips(cmp, "mw")
    post = cmp[cmp.occasion == "post_first"]
    g = post.groupby("trial").agg(st=("st", "median"), smw=("smw", "median"), ft=("ft", "sum"), fmw=("fmw", "sum"))
    rows = []
    for _, r in tr.iterrows():
        name = NAMES.get(r.trial, r.trial)
        if r.status == "excluded":
            rows.append(f"{name} & {r.source} & excluded & -- & -- & -- & -- & {esc(r.why.split(':')[0])} \\\\")
            continue
        s = g.loc[r.trial] if r.trial in g.index else None
        rows.append(f"{name} & {r.source} & {r.status.replace('_', ' ')} & {int(r.n_post)} & {int(r.n_negctl)} & "
                    + (f"{s.st:.2f} / {s.smw:.3f} & {int(s.ft)} / {int(s.fmw)}" if s is not None else "-- & --") + " & \\\\")
    body = "\n".join(rows)
    (OUT / "trials_table.tex").write_text(
        "\\begin{tabular}{llcccccl}\n\\toprule\nTrial & Source & Status & Post comp. & Neg. controls & Median span mean / rank & Verdict changes mean / rank & Reason \\\\\n\\midrule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n")


def ledger_counts():
    l = pd.read_csv(RES / "exclusion_ledger.csv")
    c = l.disposition.value_counts()
    rows = "\n".join(f"{esc(k)} & {v} \\\\" for k, v in c.items())
    (OUT / "ledger_counts.tex").write_text("\\begin{tabular}{lr}\n\\toprule\nDisposition & Entries \\\\\n\\midrule\n" + rows + "\n\\bottomrule\n\\end{tabular}\n")


def sim_tables():
    if not (RES / "simulation_v2_perm.csv").exists():
        return
    sim = pd.read_csv(RES / "simulation_v2.csv")
    rows = []
    for (d, n, sh), g in sim.groupby(["dist", "n", "shift"]):
        cells = " & ".join(f"{100*r.flip_t_strict:.1f} / {100*r.flip_t_nondel:.1f} / {100*r.flip_mw_nondel:.1f}" for _, r in g.sort_values("sd_ratio").iterrows())
        rows.append(f"{d} & {n} & {sh:.1f} & {cells} \\\\")
    (OUT / "sim_grid.tex").write_text(
        "\\begin{tabular}{llc|ccc}\n\\toprule\nFamily & $n$ per arm & Effect & SD ratio 1.0 & 1.5 & 2.5 \\\\\n\\midrule\n" + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n")
    perm = pd.read_csv(RES / "simulation_v2_perm.csv")
    rows = [f"{r.dist} & {r["shift"]:.1f} & {r.sd_ratio:.1f} & {100*r.cvm:.1f} & {100*r.ks:.1f} & {100*r.mw:.1f} & {100*r.t:.1f} \\\\" for _, r in perm.iterrows()]
    (OUT / "sim_perm.tex").write_text(
        "\\begin{tabular}{llcrrrr}\n\\toprule\nFamily & Effect & SD ratio & CvM perm. & KS perm. & Mann--Whitney & Welch \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def registry():
    reg = pd.read_csv(ROOT / "data" / "ctgov" / "primary_measures_v3.csv", low_memory=False)
    q = reg[reg.quant_cont]; stated = q[q.method_class3 != "not stated"]
    tr = stated.groupby("nct").method_class3.agg(set)
    rows = [
        ("Trials in frame (interventional, randomized, completed, results posted)", f"{reg.nct.nunique():,}"),
        ("Primary outcome measures", f"{len(reg):,}"),
        ("\\quad with a mean, median, geometric or least-squares mean parameter", f"{reg.continuous.sum():,} ({100*reg.continuous.mean():.1f}\\%)"),
        ("\\quad\\quad classed quantitative (uncensored, not a count or participant proportion)", f"{len(q):,} ({100*len(q)/len(reg):.1f}\\%)"),
        ("Trials with at least one quantitative primary measure", f"{reg.groupby('nct').quant_cont.any().sum():,} ({100*reg.groupby('nct').quant_cont.any().mean():.1f}\\%)"),
        ("Quantitative measures with a stated statistical method", f"{len(stated):,} ({100*len(stated)/len(q):.1f}\\%)"),
        ("\\quad mean-based / rank-based / other / both", f"{100*(stated.method_class3=='mean-based').mean():.1f} / {100*(stated.method_class3=='rank-based').mean():.1f} / {100*(stated.method_class3=='other').mean():.1f} / {100*(stated.method_class3=='both').mean():.1f}\\%"),
        (f"Trials with a quantitative primary and a stated method ({len(tr):,}): any mean-based / any rank-based", f"{100*tr.map(lambda S: bool(S & {'mean-based','both'})).mean():.1f}\\% / {100*tr.map(lambda S: bool(S & {'rank-based','both'})).mean():.1f}\\%"),
        ("Quantitative measures summarized by a geometric mean", f"{100*q.log_scale_summary.mean():.1f}\\%"),
    ]
    (OUT / "registry.tex").write_text("\\begin{tabular}{lr}\n\\toprule\nQuantity & Value \\\\\n\\midrule\n" + "\n".join(f"{a} & {b} \\\\" for a, b in rows) + "\n\\bottomrule\n\\end{tabular}\n")


def lilly_tables():
    import json
    L = json.load(open(RES / "lilly_v3.json")); F = L["flow"]
    rows = [("Records in the supplied file", f"{F['records']:,}", f"{F['by_arm_all']['duloxetine']:,} / {F['by_arm_all']['placebo']:,}"),
            ("Protocol codes", str(F["protocols"]), ""),
            ("Completion flag set", f"{F['complete_by_arm']['duloxetine'] + F['complete_by_arm']['placebo']:,}", f"{F['complete_by_arm']['duloxetine']:,} / {F['complete_by_arm']['placebo']:,}"),
            ("Completion rate", "", f"{100*F['complete_rate_by_arm']['duloxetine']:.1f}\\% / {100*F['complete_rate_by_arm']['placebo']:.1f}\\%"),
            ("Completed, endpoint and severity category observed (analysed)", f"{F['analysed']:,}", f"{F['analysed_by_arm']['duloxetine']:,} / {F['analysed_by_arm']['placebo']:,}")]
    (OUT / "lilly_flow.tex").write_text("\\begin{tabular}{lrr}\n\\toprule\nStage & Total & Duloxetine / placebo \\\\\n\\midrule\n" + "\n".join(f"{a} & {b} & {c} \\\\" for a, b, c in rows) + "\n\\bottomrule\n\\end{tabular}\n")
    ap = F["allocation_by_protocol"]
    codes = sorted(set(ap["duloxetine"]) | set(ap["placebo"]), key=lambda k: int(k))
    prot = "\n".join(f"{int(k)} & {ap['duloxetine'].get(k, 0):,} & {ap['placebo'].get(k, 0):,} \\\\" for k in codes)
    (OUT / "lilly_protocols.tex").write_text("\\begin{tabular}{lrr}\n\\toprule\nProtocol code & Duloxetine & Placebo \\\\\n\\midrule\n" + prot + "\n\\bottomrule\n\\end{tabular}\n")
    H = L["heterogeneity"]
    rows = [f"{s_['stratum']} & {s_['n_placebo']} & {s_['n_treated']} & {s_['theta']:.3f} & [{s_['theta_lo']:.3f}, {s_['theta_hi']:.3f}] & {s_['mean_diff']:+.1f} & {s_['p_cvm_perm']:.4f} \\\\" for s_ in L["strata"]]
    (OUT / "lilly_het.tex").write_text("\\begin{tabular}{lrrccrc}\n\\toprule\nCategory & Placebo & Duloxetine & $\\hat\\theta$ & 95\\% CI & Mean difference (recorded) & CvM permutation $P$ \\\\\n\\midrule\n" + "\n".join(rows)
        + f"\n\\midrule\n\\multicolumn{{7}}{{l}}{{Omnibus Wald on adjacent differences: $Q = {H['omnibus_wald']:.2f}$, {H['omnibus_df']} df, $P = {H['omnibus_p']:.4f}$; linear trend $z = {H['trend_z']:.2f}$, two-sided $P = {H['trend_p_two_sided']:.4f}$}} \\\\\n\\bottomrule\n\\end{{tabular}}\n")
    rows = [f"{r['spec'].replace('_', ' ')} & {r['p_interaction_hc3']:.3f} & " + " & ".join(f"{r['effects'][k]:+.3g} ({r['effect_se'][k]:.2g})" for k in ('1', '2', '3', '4')) + " \\\\" for r in L["interaction"]]
    (OUT / "lilly_interaction.tex").write_text("\\begin{tabular}{lccccc}\n\\toprule\nSpecification (cohort-wide) & HC3 Wald $P$ & Effect (SE), cat.\\ 1 & 2 & 3 & 4 \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def validation_table():
    V = pd.read_csv(RES / "validate_pi_wald.csv")
    rows = [f"{r.scenario} & {r.theta} & {r.wald_rate:.1f} [{r.wald_lo:.1f}, {r.wald_hi:.1f}] & {r.trend_rate:.1f} [{r.trend_lo:.1f}, {r.trend_hi:.1f}] \\\\" for r in V.itertuples()]
    (OUT / "validation.tex").write_text("\\begin{tabular}{lccc}\n\\toprule\nScenario (equal indices across four strata) & $\\theta$ & Omnibus Wald (\\%) & Trend $z$ (\\%) \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def registry_codebook():
    J = pd.read_csv(ROOT / "data" / "ctgov" / "registry_validation_judgments.csv")
    g = J.groupby("endpoint_type").correct.agg(["sum", "count"])
    reg = pd.read_csv(ROOT / "data" / "ctgov" / "primary_measures_v3.csv", low_memory=False)
    et = reg[reg.continuous].endpoint_type.value_counts()
    rows = [f"{t.replace('_', ' ')} & {et.get(t, 0):,} & {int(g.loc[t, 'sum'])} / {int(g.loc[t, 'count'])} \\\\" for t in ["quantitative", "time_to_event", "count_or_rate", "proportion"]]
    (OUT / "registry_codebook.tex").write_text("\\begin{tabular}{lrr}\n\\toprule\nEndpoint class (continuous parameter type) & Measures & Reading sample correct / read \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def null_by_family():
    sim = pd.read_csv(RES / "simulation_v2.csv")
    e = sim[(sim["shift"] == 0) & (sim.sd_ratio == 1.0)].sort_values(["dist", "n"])
    rows = [f"{r.dist} & {r.n} & {100*r.flip_t_strict:.1f} & {100*r.flip_t_nondel:.1f} & {100*r.flip_mw_nondel:.1f} \\\\" for r in e.itertuples()]
    (OUT / "null_by_family.tex").write_text("\\begin{tabular}{llrrr}\n\\toprule\nFamily & $n$ per arm & Welch, strict (\\%) & Welch, sample-preserving (\\%) & Mann--Whitney, sample-preserving (\\%) \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    trials_table(); ledger_counts(); sim_tables(); registry(); lilly_tables(); validation_table(); registry_codebook(); null_by_family()
    print("supplement tables written to", OUT)
