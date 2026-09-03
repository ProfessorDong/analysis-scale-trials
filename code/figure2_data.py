"""Data and labels for Figure 2, the two specification curves.

Like Figure 1, this panel pair was built from .dat files with no script
behind them and with its annotations typed into the TikZ source by hand.
Both panels are recomputed here from their sources so the figure, its
labels and the caption cannot drift apart.

Panel a, Beat the Blues, is fully public: the CSV is fetched from the
Rdatasets mirror, exactly as multistat-roc/reproduce/reproduce_worked_example.py
does, so anyone can rebuild it. If the download is unavailable the existing
.dat file is left alone and the script says so rather than writing something
it cannot verify.

Panel b, the incontinence trial, is under a data use agreement and cannot be
redistributed, so it is read from the stored per-specification results in
results/lilly_sensitivity.csv rather than from the patient records.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "NaturePortfolio" / "data"

# The five sample-preserving specifications, in the order the figure plots
# them.  Trimming is excluded: it deletes observations and so changes the
# population attribute of the estimand.
MENU = ["raw", "winsor 1%", "winsor 5%", "log-shift", "normal scores"]
LABEL = ["as recorded", "winsorize 1%", "winsorize 5%", "log scale", "normal scores"]


def lilly(stratum: int = 2) -> pd.DataFrame:
    d = pd.read_csv(ROOT / "results" / "lilly_sensitivity.csv")
    d = d[d.stratum == stratum].set_index("transform")
    missing = [m for m in MENU if m not in d.index]
    if missing:
        raise SystemExit(f"lilly_sensitivity.csv lacks {missing} for stratum {stratum}")
    d = d.loc[MENU]
    return pd.DataFrame({"spec": range(1, 6), "t": d.t_p.to_numpy(),
                         "mw": d.mw_p.to_numpy()})


def btheb() -> pd.DataFrame | None:
    """Rebuild panel a from the public CSV, reusing the reproduce script.

    Importing its MENU rather than restating the transforms keeps the figure
    and the released reproduction script provably on the same definitions.
    """
    try:
        sys.path.insert(0, str(ROOT.parent / "multistat-roc" / "reproduce"))
        import reproduce_worked_example as rwe
        from scipy import stats

        sub = pd.read_csv(rwe.URL)[["treatment", "bdi.4m"]].dropna()
        # Lower BDI is better, so the endpoint is negated to make larger
        # values better, matching the orientation rule used throughout.
        control = -sub.loc[sub.treatment == "TAU", "bdi.4m"].to_numpy(float)
        treated = -sub.loc[sub.treatment != "TAU", "bdi.4m"].to_numpy(float)
        rows = []
        for i, (name, fn) in enumerate(rwe.MENU, 1):
            x, z = fn(control, treated)
            rows.append({"spec": i,
                         "t": stats.ttest_ind(z, x, equal_var=False).pvalue,
                         "mw": stats.mannwhitneyu(z, x,
                                                  alternative="two-sided").pvalue})
        if [n for n, _ in rwe.MENU] != LABEL:
            raise SystemExit(f"menu order differs: {[n for n, _ in rwe.MENU]} vs {LABEL}")
        return pd.DataFrame(rows)
    except SystemExit:
        raise
    except Exception as e:  # network, layout or API change
        print(f"  Beat the Blues not rebuilt ({type(e).__name__}: {e});"
              " existing .dat left untouched")
        return None


def write(name: str, d: pd.DataFrame) -> None:
    out = pd.DataFrame({"spec": d.spec,
                        "tl": np.log10(d.t), "mwl": np.log10(d.mw)})
    out.to_csv(OUT / f"speccurve_{name}.dat", sep=" ", index=False)


def span(p: np.ndarray) -> float:
    return float(np.log10(p.max()) - np.log10(p.min()))


def main() -> None:
    li = lilly()
    write("lilly", li)
    bt = btheb()
    if bt is not None:
        write("btheb", bt)
    else:
        cur = pd.read_csv(OUT / "speccurve_btheb.dat", sep=r"\s+")
        bt = pd.DataFrame({"spec": cur.spec, "t": 10 ** cur.tl, "mw": 10 ** cur.mwl})

    rec = pd.concat([
        bt.assign(case="BtheB", label=LABEL),
        li.assign(case="Lilly", label=LABEL)], ignore_index=True)
    rec = rec.assign(tl=np.log10(rec.t), mwl=np.log10(rec.mw))
    rec[["case", "spec", "label", "t", "mw", "tl", "mwl"]].to_csv(
        OUT / "speccurve.dat", sep=" ", index=False)

    m = {
        "figBmin": f"{bt.t.min():.3f}", "figBmax": f"{bt.t.max():.3f}",
        "figBspan": f"{span(bt.t.to_numpy()):.2f}",
        "figBrank": f"{span(bt.mw.to_numpy()):.3f}",
        "figLspan": f"{span(li.t.to_numpy()):.1f}",
        "figLrank": f"{span(li.mw.to_numpy()):.3f}",
        # Rounded up, so the caption's "to within" claim is true of both.
        "figRankBound": f"{np.ceil(100 * max(span(bt.mw.to_numpy()), span(li.mw.to_numpy()))) / 100:.2f}",
        "figLraw": f"{li.t.iloc[0]:.4f}",
        "figLmax": f"{li.t.max():.3f}",
        # Typeset form for the smallest mean-based P in panel b.
        "figLmin": (lambda v: f"{v.split('e')[0]} \\times 10^{{{int(v.split('e')[1])}}}")(
            f"{li.t.min():.1e}"),
        # The span with the log shift removed, which is the robustness check
        # the caption points at: the transform is pathological on this
        # endpoint but the conclusion does not depend on it.
        "figLspanNoLog": f"{span(li.t.drop(index=3).to_numpy()):.1f}",
    }
    with open(OUT / "fig2_numbers.tex", "w") as fh:
        fh.write("% Generated by Paper2_Reanalysis/code/figure2_data.py.\n"
                 "% Do not edit by hand; rerun the script instead.\n")
        for k, v in m.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    for k, v in m.items():
        print(f"  {k:15s} {v}")
    print(f"\nwrote speccurve.dat, speccurve_btheb.dat, speccurve_lilly.dat,"
          f" fig2_numbers.tex to {OUT}")


if __name__ == "__main__":
    main()
