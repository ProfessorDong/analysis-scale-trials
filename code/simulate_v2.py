"""Simulation for the merged paper: the specification menus and the ROC quantities.

Grid as in simulate_span.py (four generating families, three sample sizes,
three standardized effects, three ratios of arm standard deviations, treated
arms moment-matched so the factors vary independently), REPS replicates per
cell. For every replicate the seven specifications of build_comparisons.SPECS
are applied on the generated scale and both tests recorded, together with the
empirical ROC quantities (AUC, KS, CvM) of the generated data.

A second, smaller grid at n = 200 runs the permutation test of the
Cramér-von Mises departure (B = 199) so that its rejection rate can be
compared with the Mann-Whitney and Welch tests, in particular where the arms
have equal means and unequal spread and the ROC curve crosses the diagonal.

Outputs results/simulation_v2.csv (per-cell summaries) and
results/simulation_v2_perm.csv.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from build_comparisons import SPECS, placement, roc_quantities  # noqa: E402
from simulate_span import draw  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPS = int(os.environ.get("SIM_REPS", "2000"))
PERM_REPS = int(os.environ.get("PERM_SIM_REPS", "500"))
PERM_B = 199
MENUS = {"strict": ["recorded", "log_shift", "normal_scores"],
         "nondel": ["recorded", "log_shift", "normal_scores", "winsor1", "winsor5"]}
DISTS = ["normal", "lognormal", "gamma", "contaminated"]


def one(a, b):
    out = {}
    for name, fn, _ in SPECS:
        x, z = fn(a, b)
        out[f"t_{name}"] = stats.ttest_ind(z, x, equal_var=False).pvalue
        out[f"mw_{name}"] = stats.mannwhitneyu(z, x, alternative="two-sided").pvalue
    out["auc"], out["ks"], out["cvm"] = roc_quantities(placement(a, b))
    return out


def summarize(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    s = {}
    for m, specs in MENUS.items():
        for est in ("t", "mw"):
            p = df[[f"{est}_{k}" for k in specs]]
            sp = np.log10(p.max(axis=1)) - np.log10(p.min(axis=1))
            sig = p < 0.05
            s[f"span_{est}_{m}_median"] = float(sp.median())
            s[f"flip_{est}_{m}"] = float((sig.any(axis=1) & ~sig.all(axis=1)).mean())
            s[f"rej_{est}_{m}_recorded"] = float(sig.iloc[:, 0].mean())
    s["auc_mean"] = float(df.auc.mean())
    s["cvm_mean"] = float(df.cvm.mean())
    s["ks_mean"] = float(df.ks.mean())
    s["mw_span_strict_max"] = float((np.log10(df[[f"mw_{k}" for k in MENUS['strict']]].max(axis=1)) - np.log10(df[[f"mw_{k}" for k in MENUS['strict']]].min(axis=1))).max())
    return s


def main() -> None:
    rng = np.random.default_rng(20261031)
    cells = []
    for dist in DISTS:
        for n in (50, 200, 800):
            for shift in (0.0, 0.2, 0.5):
                for vr in (1.0, 1.5, 2.5):
                    rows = [one(*draw(dist, n, shift, vr, rng)) for _ in range(REPS)]
                    s = summarize(rows)
                    s.update(dist=dist, n=n, shift=shift, sd_ratio=vr, reps=REPS)
                    cells.append(s)
                    print(f"{dist:12s} n={n:3d} d={shift:.1f} vr={vr:.1f}  flip t strict {s['flip_t_strict']:.3f} nondel {s['flip_t_nondel']:.3f} | "
                          f"mw strict {s['flip_mw_strict']:.3f} nondel {s['flip_mw_nondel']:.3f} | AUC {s['auc_mean']:.3f} CvM {s['cvm_mean']:.4f}", flush=True)
    pd.DataFrame(cells).to_csv(ROOT / "results" / "simulation_v2.csv", index=False)

    # Permutation sub-grid: n = 200, shift 0 and 0.2, all families and SD ratios.
    perm = []
    for dist in DISTS:
        for shift in (0.0, 0.2):
            for vr in (1.0, 1.5, 2.5):
                rej = {"cvm": 0, "ks": 0, "mw": 0, "t": 0}
                for _ in range(PERM_REPS):
                    a, b = draw(dist, 200, shift, vr, rng)
                    pooled = np.concatenate([a, b])
                    obs_auc, obs_ks, obs_cvm = roc_quantities(placement(a, b))
                    c_cvm = c_ks = 0
                    for _ in range(PERM_B):
                        p = rng.permutation(pooled)
                        _, pk, pc = roc_quantities(placement(p[:200], p[200:]))
                        c_cvm += pc >= obs_cvm
                        c_ks += pk >= obs_ks
                    rej["cvm"] += (c_cvm + 1) / (PERM_B + 1) < 0.05
                    rej["ks"] += (c_ks + 1) / (PERM_B + 1) < 0.05
                    rej["mw"] += stats.mannwhitneyu(b, a, alternative="two-sided").pvalue < 0.05
                    rej["t"] += stats.ttest_ind(b, a, equal_var=False).pvalue < 0.05
                r = {k: v / PERM_REPS for k, v in rej.items()}
                r.update(dist=dist, n=200, shift=shift, sd_ratio=vr, reps=PERM_REPS, B=PERM_B)
                perm.append(r)
                print(f"PERM {dist:12s} d={shift:.1f} vr={vr:.1f}  reject: CvM {r['cvm']:.3f} KS {r['ks']:.3f} MW {r['mw']:.3f} Welch {r['t']:.3f}", flush=True)
    pd.DataFrame(perm).to_csv(ROOT / "results" / "simulation_v2_perm.csv", index=False)


if __name__ == "__main__":
    main()
