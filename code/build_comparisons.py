"""Build the trial comparison set from the per-trial manifests.

Reads manifest/trials/*.yaml (written by manifest/write_manifests.py under the
rules in manifest/RULES.md) and produces

  results/comparisons_v2.csv     one row per arm contrast per endpoint per
                                 occasion (negative control, first post-dose
                                 visit, last post-dose visit), with both tests
                                 under all seven specifications and the ROC
                                 quantities
  results/exclusion_ledger.csv   every candidate variable in every manifested
                                 table, and every unmanifested deposit, with
                                 its disposition
  results/trials_v2.csv          one row per trial

Nothing in here guesses. If a manifest does not name a column, the column does
not enter, and the ledger says so.
"""
from __future__ import annotations

import fnmatch
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from sensitivity import make_trim, make_winsor, t_log_shift, t_normal_scores, t_raw  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "manifest" / "trials"
PDS = ROOT / "data" / "ProjectDataSphere"
DRYAD = ROOT / "data" / "raw" / "dryad"
RESULTS = ROOT / "results"

MIN_PER_ARM = 20
MIN_DISTINCT = 10
PERM_B = 999
SEED = 20261031

SPECS = [
    ("recorded", t_raw, "strict"),
    ("log_shift", t_log_shift, "strict"),
    ("normal_scores", t_normal_scores, "strict"),
    ("winsor1", make_winsor(1.0), "coarsen"),
    ("winsor5", make_winsor(5.0), "coarsen"),
    ("trim1", make_trim(1.0), "delete"),
    ("trim5", make_trim(5.0), "delete"),
]

EXCLUDED_VISIT = re.compile(
    r"unsched|retest|discontinu|end of (treatment|study|trial)|safety follow|follow.?up|afup|resp|average|progression|^$",
    re.I,
)


# --------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------
def locate(trial: dict, name: str) -> Path:
    if trial["source"] == "PDS":
        for d in (PDS / trial["id"] / "extracted", PDS / trial["id"]):
            if (d / name).exists():
                return d / name
    else:
        if (DRYAD / name).exists():
            return DRYAD / name
    raise FileNotFoundError(f"{trial['id']}: {name}")


_cache: dict[Path, pd.DataFrame] = {}


def read(path: Path, read_kw: dict | None = None) -> pd.DataFrame:
    if path in _cache:
        return _cache[path]
    if path.suffix.lower() == ".sas7bdat":
        df = pd.read_sas(path, format="sas7bdat", encoding="latin-1")
    else:
        kw = {"low_memory": False}
        kw.update(read_kw or {})
        try:
            df = pd.read_csv(path, **kw)
        except Exception:
            kw.pop("low_memory", None)
            df = pd.read_csv(path, sep=None, engine="python", **kw)
    _cache[path] = df
    return df


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def placement(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mid-rank placement values of b in a: P_a(Y > b_i) + 0.5 P_a(Y = b_i)."""
    sa = np.sort(a)
    gt = len(sa) - np.searchsorted(sa, b, side="right")
    eq = np.searchsorted(sa, b, side="right") - np.searchsorted(sa, b, side="left")
    return (gt + 0.5 * eq) / len(sa)


def roc_quantities(pv: np.ndarray) -> tuple[float, float, float]:
    """Area, KS and Cramer-von Mises departure of the midpoint-placement EDF from the diagonal.

    Exact for the EDF: with sorted placements u_(1) <= ... <= u_(n),
    D = max_i max{i/n - u_(i), u_(i) - (i-1)/n} and
    W^2 = int (EDF(t) - t)^2 dt = 1/(12 n^2) + (1/n) sum_i (u_(i) - (2i-1)/(2n))^2.
    The area is 1 - mean(pv) = P(T > C) + P(T = C)/2 with midpoint placements.
    """
    u = np.sort(pv); n = len(u); i = np.arange(1, n + 1)
    d = float(max(np.max(i / n - u), np.max(u - (i - 1) / n)))
    w2 = float(1.0 / (12.0 * n * n) + np.mean((u - (2 * i - 1) / (2.0 * n)) ** 2))
    return float(1.0 - pv.mean()), d, w2


def cvm_perm_p(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> float:
    pooled = np.concatenate([a, b])
    na = len(a)
    obs = roc_quantities(placement(a, b))[2]
    count = 0
    for _ in range(PERM_B):
        p = rng.permutation(pooled)
        if roc_quantities(placement(p[:na], p[na:]))[2] >= obs:
            count += 1
    return (count + 1) / (PERM_B + 1)


def round_pooled(a: np.ndarray, b: np.ndarray, digits: int = 9) -> tuple[np.ndarray, np.ndarray]:
    """Round both arms to `digits` significant figures on the pooled scale.

    Deposited files carry floating-point noise (-0.331 next to
    -0.33099999999999996) that a rank treats as two values and a shifted
    logarithm collapses into one. Nine significant figures exceed any
    clinical measurement's precision and remove the noise, so that the
    rank-based quantities are exactly invariant as Proposition 1 states.
    The grid is common to both arms.
    """
    pooled = np.concatenate([a, b])
    scale = np.max(np.abs(pooled)) if len(pooled) else 0.0
    if scale == 0:
        return a, b
    q = 10.0 ** (digits - 1 - np.floor(np.log10(scale)))
    return np.round(a * q) / q, np.round(b * q) / q


def compare(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict | None:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    a, b = round_pooled(a, b)
    if len(a) < MIN_PER_ARM or len(b) < MIN_PER_ARM:
        return None
    pooled = np.concatenate([a, b])
    if len(np.unique(pooled)) < MIN_DISTINCT or pooled.std() == 0:
        return None
    out = {"n_a": len(a), "n_b": len(b), "n_distinct": int(len(np.unique(pooled)))}
    for name, fn, _ in SPECS:
        try:
            x, z = fn(a, b)
            out[f"p_t_{name}"] = stats.ttest_ind(z, x, equal_var=False).pvalue
            out[f"p_mw_{name}"] = stats.mannwhitneyu(z, x, alternative="two-sided").pvalue
            out[f"d_{name}"] = float(z.mean() - x.mean())          # mean difference, second arm minus first, on that scale
            out[f"auc_{name}"] = float(1.0 - placement(x, z).mean())  # area on that scale (identical under strict members)
        except Exception:
            out[f"p_t_{name}"] = np.nan
            out[f"p_mw_{name}"] = np.nan
            out[f"d_{name}"] = np.nan
            out[f"auc_{name}"] = np.nan
    pv = placement(a, b)
    out["auc"], out["ks"], out["cvm"] = roc_quantities(pv)
    out["p_cvm_perm"] = cvm_perm_p(a, b, rng)
    out["skew"] = float(stats.skew(pooled))
    out["kurt"] = float(stats.kurtosis(pooled))
    out["sd_ratio"] = float(b.std(ddof=1) / a.std(ddof=1)) if a.std(ddof=1) > 0 else np.nan
    out["mean_a"], out["mean_b"] = float(a.mean()), float(b.mean())
    out["median_a"], out["median_b"] = float(np.median(a)), float(np.median(b))
    return out


# --------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------
def load_arms(trial: dict) -> tuple[pd.DataFrame, list[dict]]:
    """Return (frame with subject + one or more arm columns, contrasts).

    Each contrast is {"column": c, "label": ..., "levels": [l1, l2]}.
    """
    spec = trial["arms"]
    df = read(locate(trial, spec["file"]), spec.get("read_kw"))
    subj = spec.get("subject")
    if subj is None:
        df = df.reset_index().rename(columns={"index": "_row"})
        subj = "_row"
    for k, v in (spec.get("filter") or {}).items():
        df = df[df[k].astype(str).str.strip() == str(v)]
    df = df.assign(_subject=subject_key(df, subj))
    contrasts = []
    if spec.get("type") == "factorial":
        cols = list(spec["factors"].keys())
        frame = df[["_subject"] + cols].dropna()
        for c in cols:
            frame[c] = frame[c].astype(str).str.strip()
        if (frame.groupby("_subject")[cols].nunique() > 1).any().any():
            raise ValueError(f"{trial['id']}: a participant key carries more than one allocation")
        frame = frame.drop_duplicates(subset=["_subject"]).copy()
        for c in cols:
            lv = sorted(frame[c].unique())
            contrasts.append({"column": c, "label": f"{c}: {lv[1]} vs {lv[0]} ({spec['factors'][c]})", "levels": [lv[0], lv[1]]})
    else:
        c = spec["column"]
        frame = df[["_subject", c]].dropna().copy()
        frame[c] = frame[c].astype(str).str.strip()
        if (frame.groupby("_subject")[c].nunique() > 1).any():
            raise ValueError(f"{trial['id']}: a participant key carries more than one allocation")
        frame = frame.drop_duplicates(subset=["_subject"])
        for lv in spec.get("exclude_levels") or []:
            frame = frame[frame[c] != lv]
        counts = frame[c].value_counts()
        levels = [l for l in counts.index if counts[l] >= MIN_PER_ARM]
        levels = sorted(levels)
        for i in range(len(levels)):
            for j in range(i + 1, len(levels)):
                contrasts.append({"column": c, "label": f"{levels[j]} vs {levels[i]}", "levels": [levels[i], levels[j]]})
    return frame, contrasts


# --------------------------------------------------------------------------
# Measurement series
# --------------------------------------------------------------------------
class Series:
    """One measured quantity at one occasion, keyed by subject."""

    def __init__(self, endpoint, variable, visit, timing, order, role, label, table, values: pd.Series):
        self.endpoint, self.variable, self.visit = endpoint, variable, visit
        self.timing, self.order, self.role, self.label, self.table = timing, order, role, label, table
        self.values = values  # index subject (str), float values


def visit_class(label, visits: dict) -> tuple[str | None, tuple]:
    """Classify a visit label: ('pre'|'pre_dose'|'post'|None, order key)."""
    s = str(label).strip()
    if s in visits.get("pre", []):
        return "pre", (0,)
    if s in visits.get("pre_dose", []):
        return "pre_dose", (0,)
    post = visits.get("post")
    if isinstance(post, list):
        if s in post:
            return "post", (post.index(s) + 1,)
        return None, ()
    m = re.match(post, s)
    if m:
        return "post", tuple(int(g) for g in m.groups())
    return None, ()


def numeric_visit_class(v, visits: dict) -> tuple[str | None, tuple]:
    if pd.isna(v):
        return None, ()
    v = float(v)
    if v in [float(x) for x in visits.get("pre_dose", [])]:
        return "pre_dose", (0,)
    if "pre_max" in visits and v <= visits["pre_max"]:
        return "pre", (0,)
    if "post_min" in visits and v >= visits["post_min"]:
        return "post", (v,)
    return None, ()


def subject_key(df: pd.DataFrame, subj) -> pd.Series:
    """Participant key as a string; a list of columns gives a composite key (site|id)."""
    if isinstance(subj, (list, tuple)):
        return df[list(subj)].astype(str).apply(lambda r: "|".join(v.strip() for v in r), axis=1)
    return df[subj].astype(str).str.strip()


def first_per_subject(df: pd.DataFrame, subj, val: str) -> pd.Series:
    d = pd.DataFrame({"_k": subject_key(df, subj), "_v": pd.to_numeric(df[val], errors="coerce")})
    d = d.dropna().drop_duplicates(subset=["_k"], keep="first")
    return d.set_index("_k")["_v"]


def table_series(trial: dict, tbl: dict, ledger: list) -> list[Series]:
    path = locate(trial, tbl["file"])
    df = read(path, tbl.get("read_kw"))
    for k, v in (tbl.get("filter") or {}).items():
        df = df[pd.to_numeric(df[k], errors="coerce") == v] if df[k].dtype.kind in "if" else df[df[k].astype(str) == str(v)]
    subj = tbl.get("subject")
    if subj is None:
        df = df.reset_index().rename(columns={"index": "_row"})
        subj = "_row"
    out: list[Series] = []
    kind = tbl["kind"]
    excl = tbl.get("exclude") or {}

    def excl_reason(name: str) -> str | None:
        for pat, why in excl.items():
            if fnmatch.fnmatch(name, pat) or name == pat:
                return why
        return None

    if kind == "wide":
        variables = tbl.get("variables") or {}
        for c in df.columns:
            if c == subj:
                continue
            if c in variables:
                v = variables[c]
                s = first_per_subject(df, subj, c)
                out.append(Series(v["endpoint"] or c, c, "", v["timing"], v["order"], v["role"], v["label"], tbl["file"], s))
                ledger.append(dict(trial=trial["id"], table=tbl["file"], variable=c, disposition=v["timing"], role=v["role"], note=v["label"]))
            else:
                why = excl_reason(str(c))
                numeric = df[c].dtype.kind in "if" and df[c].dropna().nunique() >= MIN_DISTINCT
                ledger.append(dict(trial=trial["id"], table=tbl["file"], variable=c, disposition="excluded",
                                   role="", note=why or ("not manifested: " + ("numeric, no eligible role (rules 5-8)" if numeric else "non-numeric or fewer than 10 distinct values (rule 9)"))))
    elif kind == "long":
        par, val, vis = tbl["param"], tbl["value"], tbl["visit"]
        exclude_params = tbl.get("exclude_params") or {}
        roles = tbl.get("param_roles") or {}
        for p in pd.unique(df[par].dropna()):
            pname = str(p).strip()
            if pname in exclude_params and not exclude_params[pname].startswith("tumour marker retained"):
                ledger.append(dict(trial=trial["id"], table=tbl["file"], variable=pname, disposition="excluded", role="", note=exclude_params[pname]))
                continue
            blk = df[df[par] == p]
            role = roles.get(pname, tbl["role"])
            seen = []
            for v in pd.unique(blk[vis].dropna()):
                cls, key = visit_class(v, tbl["visits"])
                if cls is None:
                    continue
                s = first_per_subject(blk[blk[vis] == v], subj, val)
                out.append(Series(pname, pname, str(v).strip(), cls, key, role, f"{pname} at {v}", tbl["file"], s))
                seen.append((cls, str(v).strip()))
            ledger.append(dict(trial=trial["id"], table=tbl["file"], variable=pname, disposition="candidate", role=role,
                               note="visits: " + "; ".join(f"{c}:{l}" for c, l in seen)))
    elif kind == "wide_visit":
        vis = tbl["visit"]
        for c, v in (tbl.get("variables") or {}).items():
            numeric_visits = isinstance(tbl["visits"], dict) and "post_min" in tbl["visits"]
            for lab in pd.unique(df[vis].dropna()):
                cls, key = numeric_visit_class(lab, tbl["visits"]) if numeric_visits else visit_class(lab, tbl["visits"])
                if cls is None:
                    continue
                s = first_per_subject(df[df[vis] == lab], subj, c)
                out.append(Series(v["endpoint"] or c, c, str(lab), cls, key, v["role"], f"{v['label']} at {lab}", tbl["file"], s))
            ledger.append(dict(trial=trial["id"], table=tbl["file"], variable=c, disposition="candidate", role=v["role"], note=v["label"]))
        for c in df.columns:
            if c not in (tbl.get("variables") or {}) and c not in (subj, vis):
                ledger.append(dict(trial=trial["id"], table=tbl["file"], variable=c, disposition="excluded", role="", note=excl_reason(str(c)) or "not manifested"))
    elif kind == "wide_windows":
        t = tbl["time"]
        tt = pd.to_numeric(df[t], errors="coerce")
        w = tbl["windows"]
        for c, v in (tbl.get("variables") or {}).items():
            lo, hi = w["pre"]
            pre = df[(tt <= hi) & ((lo is None) | (tt > (lo if lo is not None else -np.inf)))].sort_values(t)
            out.append(Series(v["endpoint"] or c, c, "baseline", "pre", (0,), v["role"], f"{v['label']} at baseline", tbl["file"], first_per_subject(pre, subj, c)))
            for i, (lab, lo, hi) in enumerate(w["post"]):
                blk = df[(tt >= lo) & (tt < hi)].sort_values(t)
                out.append(Series(v["endpoint"] or c, c, lab, "post", (i + 1,), v["role"], f"{v['label']} at {lab}", tbl["file"], first_per_subject(blk, subj, c)))
            ledger.append(dict(trial=trial["id"], table=tbl["file"], variable=c, disposition="candidate", role=v["role"], note=v["label"]))
        for c in df.columns:
            if c not in (tbl.get("variables") or {}) and c not in (subj, t):
                ledger.append(dict(trial=trial["id"], table=tbl["file"], variable=c, disposition="excluded", role="", note=excl_reason(str(c)) or "not manifested"))
    else:
        raise ValueError(kind)
    return out


# --------------------------------------------------------------------------
# Comparison assembly
# --------------------------------------------------------------------------
def arm_values(s: Series, arms: pd.DataFrame, column: str, level: str) -> np.ndarray:
    ids = arms.loc[arms[column] == level, "_subject"]
    return s.values.reindex(ids).dropna().to_numpy(float)


def eligible(s: Series, arms: pd.DataFrame, con: dict) -> bool:
    a = arm_values(s, arms, con["column"], con["levels"][0])
    b = arm_values(s, arms, con["column"], con["levels"][1])
    if len(a) < MIN_PER_ARM or len(b) < MIN_PER_ARM:
        return False
    return len(np.unique(np.concatenate([a, b]))) >= MIN_DISTINCT


def arm_counts(s: Series, arms: pd.DataFrame, con: dict) -> tuple[int, int, int]:
    a = arm_values(s, arms, con["column"], con["levels"][0])
    b = arm_values(s, arms, con["column"], con["levels"][1])
    return len(a), len(b), int(len(np.unique(np.concatenate([a, b])))) if len(a) + len(b) else 0


SELECTION: list[dict] = []


def process_trial(trial: dict, ledger: list, rng: np.random.Generator) -> list[dict]:
    arms, contrasts = load_arms(trial)
    series: list[Series] = []
    for tbl in trial.get("tables") or []:
        series.extend(table_series(trial, tbl, ledger))
    recs = []
    # A measurement series is identified by its endpoint name within the trial, across
    # tables, so that an endpoint stored in one file per visit is still one series.
    # Cross-table merges are printed for review.
    endpoints = sorted({s.endpoint for s in series})
    for ep in endpoints:
        ss = [s for s in series if s.endpoint == ep]
        tables = sorted({s.table for s in ss})
        if len(tables) > 1:
            print(f"    series '{ep}' spans tables {tables}")
        for con in contrasts:
            pre_all = [s for s in ss if s.timing == "pre"]
            post_all = sorted([s for s in ss if s.timing == "post"], key=lambda s: s.order)
            pre = [s for s in pre_all if eligible(s, arms, con)]
            post = [s for s in post_all if eligible(s, arms, con)]
            chosen: list[tuple[str, Series]] = [("negative_control", s) for s in pre[:1]]
            if post:
                chosen.append(("post_first", post[0]))
                if len(post) > 1:
                    chosen.append(("post_last", post[-1]))
            chosen_ids = {id(s): occ for occ, s in chosen}
            for s in ss:
                na, nb, nd = arm_counts(s, arms, con)
                if id(s) in chosen_ids:
                    disp = chosen_ids[id(s)]
                elif s.timing == "pre_dose":
                    disp = "unused: pre-dose occasion (rule 7)"
                elif na < MIN_PER_ARM or nb < MIN_PER_ARM:
                    disp = "ineligible: fewer than 20 observed per arm (rule 13)"
                elif nd < MIN_DISTINCT:
                    disp = "ineligible: fewer than 10 distinct values (rule 9)"
                elif s.timing == "pre":
                    disp = "not selected: a later-listed baseline occasion of the same series"
                else:
                    disp = "not selected: intermediate post-dose visit (rule 10)"
                SELECTION.append(dict(trial=trial["id"], endpoint=ep, table=s.table, variable=s.variable, visit=s.visit,
                                      timing=s.timing, order=str(s.order), contrast=con["label"], n_a=na, n_b=nb,
                                      n_distinct=nd, disposition=disp))
            for occasion, s in chosen:
                a = arm_values(s, arms, con["column"], con["levels"][0])
                b = arm_values(s, arms, con["column"], con["levels"][1])
                r = compare(a, b, rng)
                if r is None:
                    continue
                r.update(trial=trial["id"], source=trial["source"], design=trial.get("design", ""),
                         table=s.table, endpoint=s.endpoint, variable=s.variable, visit=s.visit,
                         occasion=occasion, role=s.role, label=s.label, contrast=con["label"],
                         n_post_visits_eligible=len(post))
                recs.append(r)
    return recs


def main() -> None:
    rng = np.random.default_rng(SEED)
    ledger: list[dict] = []
    rows: list[dict] = []
    trials_out: list[dict] = []
    for path in sorted(MANIFESTS.glob("*.yaml")):
        trial = yaml.safe_load(path.read_text())
        status = trial.get("status")
        print(f"=== {trial['id']} [{status}]", flush=True)
        if status == "excluded":
            ledger.append(dict(trial=trial["id"], table="", variable="", disposition="trial excluded", role="", note=trial.get("why", "")))
            trials_out.append(dict(trial=trial["id"], source=trial["source"], status=status, n_post=0, n_post_last=0, n_negctl=0, why=trial.get("why", "")))
            continue
        recs = process_trial(trial, ledger, rng)
        n_post = sum(r["occasion"] == "post_first" for r in recs)
        n_last = sum(r["occasion"] == "post_last" for r in recs)
        n_neg = sum(r["occasion"] == "negative_control" for r in recs)
        print(f"    post_first {n_post}  post_last {n_last}  negative_control {n_neg}", flush=True)
        trials_out.append(dict(trial=trial["id"], source=trial["source"], status=status if n_post else "negative_control_only",
                               n_post=n_post, n_post_last=n_last, n_negctl=n_neg, why=trial.get("why", "")))
        rows.extend(recs)
    RESULTS.mkdir(exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "comparisons_v2.csv", index=False)
    pd.DataFrame(ledger).to_csv(RESULTS / "exclusion_ledger.csv", index=False)
    pd.DataFrame(SELECTION).to_csv(RESULTS / "selection_ledger.csv", index=False)
    pd.DataFrame(trials_out).to_csv(RESULTS / "trials_v2.csv", index=False)
    print(f"\nwrote {len(out)} comparisons; "
          f"{(out.occasion == 'post_first').sum()} post_first from {out[out.occasion == 'post_first'].trial.nunique()} trials; "
          f"{(out.occasion == 'negative_control').sum()} negative controls from {out[out.occasion == 'negative_control'].trial.nunique()} trials")


if __name__ == "__main__":
    main()
