"""Cross-trial sweep of the analyst-choice instability claim.

For every eligible trial and endpoint this measures how far the two
p-values move as the preprocessing choice varies.  The headline quantity
is the *span*: the number of orders of magnitude a p-value covers across
the transform menu.  The claim under test is that the mean-based span is
large while the rank-based span is small.

Spans are computed over the five transforms that keep every observation
(raw, both winsorisings, log-shift, normal scores).  Trimming is reported
separately because deleting observations changes the estimand, so a span
that included it would conflate two different objections.

Endpoints measured before randomisation are marked `baseline=True`.  They
carry no treatment effect by construction and serve as negative controls.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from sensitivity import TRANSFORMS, evaluate  # noqa: E402

# Transforms that preserve the full sample.  Spans are computed over these.
NON_DELETING = {"raw", "winsor 1%", "winsor 5%", "log-shift", "normal scores"}


@dataclass
class Endpoint:
    col: str
    higher_is_better: bool
    baseline: bool = False


@dataclass
class Trial:
    name: str
    package: str
    item: str
    arm: str
    control: object
    endpoints: List[Endpoint]
    note: str = ""


TRIALS: List[Trial] = [
    Trial(
        "Beat the Blues", "HSAUR", "BtheB", "treatment", "TAU",
        [
            Endpoint("bdi.pre", higher_is_better=False, baseline=True),
            Endpoint("bdi.2m", higher_is_better=False),
            Endpoint("bdi.4m", higher_is_better=False),
            Endpoint("bdi.6m", higher_is_better=False),
            Endpoint("bdi.8m", higher_is_better=False),
        ],
        "RCT of computerised CBT vs treatment as usual; lower BDI is better",
    ),
    Trial(
        "Licorice gargle", "medicaldata", "licorice_gargle", "treat", 0,
        [
            Endpoint("preOp_calcBMI", higher_is_better=True, baseline=True),
            Endpoint("pacu30min_throatPain", higher_is_better=False),
            Endpoint("pacu30min_swallowPain", higher_is_better=False),
            Endpoint("pacu90min_throatPain", higher_is_better=False),
            Endpoint("postOp4hour_throatPain", higher_is_better=False),
            Endpoint("pod1am_throatPain", higher_is_better=False),
        ],
        "RCT of licorice vs sugar-water gargle before intubation",
    ),
    Trial(
        "Supraclavicular block", "medicaldata", "supraclavicular", "group", 1,
        [
            Endpoint("bmi", higher_is_better=True, baseline=True),
            Endpoint("onset_sensory", higher_is_better=False),
            Endpoint("onset_motor", higher_is_better=False),
            Endpoint("med_duration", higher_is_better=True),
            Endpoint("vps_rest", higher_is_better=False),
            Endpoint("vps_movement", higher_is_better=False),
            Endpoint("opioid_total", higher_is_better=False),
        ],
        "RCT of two supraclavicular anesthesia mixtures",
    ),
    Trial(
        "OPT periodontal", "medicaldata", "opt", "Group", "C",
        [
            Endpoint("GA.at.outcome", higher_is_better=True),
            Endpoint("Birthweight", higher_is_better=True),
            Endpoint("Apgar5", higher_is_better=True),
        ],
        "RCT of periodontal therapy in pregnancy; known null",
    ),
    Trial(
        "rhDNase", "survival", "rhDNase", "trt", 0,
        [Endpoint("fev", higher_is_better=True, baseline=True)],
        "RCT of dornase alfa; FEV recorded at entry so baseline only",
    ),
    Trial(
        "UDCA", "survival", "udca", "trt", 0,
        [
            Endpoint("bili", higher_is_better=False, baseline=True),
            Endpoint("riskscore", higher_is_better=False, baseline=True),
        ],
        "RCT of ursodeoxycholic acid; labs recorded at entry",
    ),
    Trial(
        "PBC", "survival", "pbc", "trt", 1.0,
        [
            Endpoint("bili", higher_is_better=False, baseline=True),
            Endpoint("chol", higher_is_better=False, baseline=True),
            Endpoint("alk.phos", higher_is_better=False, baseline=True),
        ],
        "RCT of D-penicillamine; labs recorded at registration",
    ),
]


def _fetch(trial: Trial) -> Optional[pd.DataFrame]:
    import statsmodels.api as sm

    try:
        return sm.datasets.get_rdataset(trial.item, trial.package).data
    except Exception as exc:  # noqa: BLE001
        print(f"  fetch failed for {trial.name}: {str(exc)[:90]}")
        return None


def sweep_endpoint(
    df: pd.DataFrame, trial: Trial, ep: Endpoint
) -> Optional[Dict[str, object]]:
    """Run the transform menu on one endpoint and summarise the spans."""
    if ep.col not in df.columns:
        return None
    sub = df[[ep.col, trial.arm]].dropna()
    if len(sub) < 30:
        return None

    sign = 1.0 if ep.higher_is_better else -1.0
    y = sign * sub[ep.col].to_numpy(dtype=float)
    is_trt = (sub[trial.arm] != trial.control).to_numpy()
    y_c, y_t = y[~is_trt], y[is_trt]
    if len(y_c) < 15 or len(y_t) < 15:
        return None

    rows = {}
    for name, fn in TRANSFORMS:
        try:
            a, b = fn(y_c, y_t)
            rows[name] = evaluate(a, b)
        except Exception:  # noqa: BLE001
            continue
    if "raw" not in rows:
        return None

    keep = {k: v for k, v in rows.items() if k in NON_DELETING}
    t_span = _span([v["t_p"] for v in keep.values()])
    mw_span = _span([v["mw_p"] for v in keep.values()])
    base = rows["raw"]

    return {
        "trial": trial.name,
        "endpoint": ep.col,
        "baseline": ep.baseline,
        "n": int(base["n"]),
        "auc": base["auc"],
        "skew": base["skew"],
        "kurt": base["kurt"],
        "t_p_raw": base["t_p"],
        "mw_p_raw": base["mw_p"],
        "t_p_min": min(v["t_p"] for v in keep.values()),
        "t_p_max": max(v["t_p"] for v in keep.values()),
        "mw_p_min": min(v["mw_p"] for v in keep.values()),
        "mw_p_max": max(v["mw_p"] for v in keep.values()),
        "t_span": t_span,
        "mw_span": mw_span,
        "excess_span": t_span - mw_span,
        "flips_significance": _flips(keep, 0.05),
    }


def _span(ps: Sequence[float]) -> float:
    """Orders of magnitude covered by a set of p-values."""
    vals = [p for p in ps if np.isfinite(p) and p > 0]
    if len(vals) < 2:
        return float("nan")
    return float(np.log10(max(vals)) - np.log10(min(vals)))


def _flips(rows: Dict[str, Dict[str, float]], alpha: float) -> str:
    """Whether the significance verdict changes with the transform."""
    t_sig = {v["t_p"] < alpha for v in rows.values() if np.isfinite(v["t_p"])}
    mw_sig = {v["mw_p"] < alpha for v in rows.values() if np.isfinite(v["mw_p"])}
    parts = []
    if len(t_sig) > 1:
        parts.append("t-test")
    if len(mw_sig) > 1:
        parts.append("Mann-Whitney")
    return "+".join(parts) if parts else "neither"


def run() -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for trial in TRIALS:
        print(f"[{trial.name}] {trial.package}::{trial.item}")
        df = _fetch(trial)
        if df is None:
            continue
        for ep in trial.endpoints:
            rec = sweep_endpoint(df, trial, ep)
            if rec is None:
                print(f"  skip {ep.col}: absent or too small")
                continue
            records.append(rec)
            tag = "baseline" if ep.baseline else "outcome"
            print(
                f"  {ep.col:24s} [{tag}] n={rec['n']:4d} AUC={rec['auc']:.3f} "
                f"t-span={rec['t_span']:5.2f} mw-span={rec['mw_span']:5.2f}"
            )
    return pd.DataFrame(records)


if __name__ == "__main__":
    out = run()
    dest = Path(__file__).resolve().parents[1] / "results" / "sweep_tierA.csv"
    out.to_csv(dest, index=False)
    print(f"\nwrote {dest}  ({len(out)} endpoint rows)")
