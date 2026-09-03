"""Analyse Project Data Sphere trials under the pre-registered transform menu.

The ten downloaded studies use three conventions:

  A. Modern ADaM: a long file keyed by PARAMCD/PARAM with the value in
     AVAL, plus an ADSL carrying the randomised arm.
  B. Amgen legacy: long files where each row is a visit and the laboratory
     values sit in named numeric columns, with the arm embedded in the
     same file.
  C. Wide: one row per participant, endpoints as named columns.

All three are reduced to the same thing: one row per participant, for one
parameter, at one visit.  Enforcing that is a correctness requirement
rather than tidiness.  Stacking visits would make rows non-independent and
invalidate both the t-test and the Mann-Whitney test being compared, which
is the trap that removed 14 tables from the Dryad screen.

Baseline-flagged and screening visits are analysed too, and marked, since
a randomised trial should show no arm difference before treatment starts.
They are the built-in negative controls.
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from sensitivity import TRANSFORMS, evaluate  # noqa: E402
from sweep import NON_DELETING, _span  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PDS = ROOT / "data" / "ProjectDataSphere"

MIN_PER_ARM = 20
MIN_UNIQUE = 10
MAX_PARAMS = 30
MAX_VISITS = 8

SUBJ = ["USUBJID", "SUBJID", "SUBJECT_ID", "PATID", "PTID", "SUBJECT", "ID"]
# Allocation column names seen across the downloaded trials.  Factorial
# designs rarely use a column called "arm": VITAL encodes its 2x2 as
# vitdactive/fishoilactive, and CALGB 40603 as factora/factorb.
ARM = ["TRTP", "TRT", "ARM", "TRTA", "TX", "TXGROUP", "RANDF", "TRTGRP", "ATRT",
       "VITDACTIVE", "FISHOILACTIVE", "TREATMENT", "RANDGRP",
       "FACTORA", "FACTORB", "INDRX",
       # Alliance (CALGB) legacy exports.
       "TREAT_ASSIGNED", "TRT_ARM", "TRT_ARM_LABEL", "TRTARM", "ARM_ASSIGNED",
       "RANDARM", "TX_ARM", "TREATMENT_ARM"]

# Allocation columns that record what a participant ACTUALLY received rather
# than what they were randomised to.  Using one breaks randomisation, because
# the groups it forms are defined partly by post-randomisation behaviour.
NOT_ASSIGNED = re.compile(r"received|actual|_act$|^arm_rec", re.I)

# Levels that are not randomised arms.  A screen failure or an untreated
# withdrawal never received allocation, so including them would compare
# groups that randomisation never balanced.
NOT_RANDOMISED = re.compile(
    r"screen\s*fail|not\s*treat|untreated|not\s*random|excluded|^n/?a$|unknown|missing",
    re.I,
)
VALUE = ["AVAL", "LBSTRESN", "VSSTRESN", "QSSTRESN", "LSSLD", "RESULT", "AVALC"]
PARAM = ["PARAMCD", "PARAM", "LBTESTCD", "LBTEST", "VSTESTCD", "QSTESTCD", "LSCAT"]
VISIT = ["AVISIT", "VISIT", "AVISITN", "VISITNUM", "VISITDY"]

BASELINE_PAT = re.compile(r"screen|baseline|^day\s*-?[01]|week\s*0\b|^bl$", re.I)

# Variables measured before randomisation, identified by name rather than by
# a visit label.  Wide one-row-per-participant files carry no visit column, so
# a baseline covariate is otherwise indistinguishable from an outcome and gets
# counted as one.  That is the wrong way round: under randomisation these have
# no treatment effect by construction, which makes them negative controls, the
# most informative stratum in the design.  The B_ prefix is the Amgen baseline
# convention; DIAGMONS and METMONS are months since diagnosis and since
# metastasis, both fixed before allocation.
PRE_RANDOMISATION = re.compile(
    r"^b_|^baseline|^base_|_base$|^scr_|_scr$|^pre_|_pre$"
    r"|diagmons|metmons|^cadiag|^age|educat|cigpd"
    r"|height|^bsa$|^b?bmi\d?$|^weight$|^wgt$"
    # Pathology recorded at the qualifying surgery, which precedes
    # randomisation in the adjuvant trials (CALGB 80101, Z1031).
    r"|lymph_node|nodeaxil|nodesent|^tsize",
    re.I,
)

# Datasets and variables that record the intervention itself rather than a
# response to it.  Drug dose is fixed by allocation, so comparing arms on
# it separates them almost perfectly by construction and produces enormous
# spurious spans.  These are not endpoints and must never enter the screen.
SKIP_FILE = re.compile(
    r"^ex\b|exposure|adex|dos(e|ing)|chemotx|radiotx|othcantx|conmed|^cm\b|adcm"
    r"|excomp|exvial|disposit|^ds\b|eligcrit|(^|_)ae($|_)|adae|c_ae"
    r"|medhist|^mh\b|^fmt$|_fmt$|format",
    re.I,
)
SKIP_PARAM = re.compile(
    # Dose appears far more often as a suffix than a prefix.  Anchoring on
    # ^dose let BADOSE, IRAWDOSE, FBCDOSE, bev_ndoses, DXMNFDOS and DOSREFWK
    # through, every one of them a delivered drug amount fixed largely by
    # allocation.  This is the same error that produced 301-order spans on the
    # first pass, reappearing under different names, so match dose anywhere.
    r"dose|dosag|dos$|dosref|mnfdos|^fst..ds$"
    # Counts of infusions or administrations are exposure by another name.
    r"|infus|(^|_)num(al|ch|dl|b|fb|fi|ir|ox)?inf|numchall|ninfus"
    r"|^trtdur|adminis|drug"
    # Number of treatment cycles delivered is exposure.  In a maintenance
    # design it is partly a tolerability outcome, but it is close enough to
    # allocated dose that including it repeats the error that produced
    # 301-order spans on the first pass.  Excluded.
    r"|cycle|course|ncours|^visits?$",
    re.I,
)

# Censored time-to-event variables.  A duration that is subject to censoring
# is not a continuous outcome in the sense this analysis requires: both the
# t-test and the Mann-Whitney test ignore the censoring and neither is a
# valid analysis of it.  Survival endpoints belong to a different estimand
# and a different method, so they are excluded outright.
TIME_TO_EVENT = re.compile(
    r"surv|^os|_os$|^os_|pfs|dfs|ttf|ttp|^tte|efs|rfs|dor"
    r"|(^|_)(time|dur|duration|yrs|years?|mos|months?|days?|dy|wk|weeks?)$"
    r"|time_to|_to_(death|event|prog)|death|dth|relapse|recur|progress"
    r"|followup|follow_up|fu_"
    # Follow-up and event-time variables that carry a numeric suffix, such as
    # futime8 or pgtime5, and cancer-incidence year variables such as brcayrs
    # and randyrs.  Deliberately worded so that a procedure duration like
    # surgtime, which is not censored, is retained.
    r"|fu_?time|pg_?time|dfs_?time|os_?time|surv_?time|prog_?time"
    r"|ca_?yrs?\d*$|[a-z]{2,}cayrs|randyrs"
    # Duration variables reached the screen under names the tokens above do
    # not catch: failtime and failtime2 (time to treatment failure), fumos
    # (follow-up months), RDURLR and SDDURCR (duration of response and of
    # stable disease), RSPDURWK, TFAILMT.  Every one is censored.
    r"|fail(time|cens|mt)|^tfail|fumos|(^|_)mos$|rspdur|(^|_)sddur|(^|_)rdurw?(lr|cr)"
    # Day-of-event variables in the Amgen exports: PDDYLR is the day of
    # progressive disease by local review, RDYCR the day of response by
    # central review.  Both are censored event times recorded in days.
    r"|^pddy|^rdy(cr|lr)|^pdmt|dy(cr|lr)\d*$|^eotpday|^antumdy",
    re.I,
)

# Any remaining variable naming a duration.  The Amgen and Clovis exports
# carry dozens: BDURWK and OXDURWK are weeks of drug delivered, so exposure;
# IEDUR and SKSEVDUR are adverse-event durations, censored at discontinuation;
# DIAGDUR and METDDUR are pre-randomisation history, not endpoints.  Rather
# than enumerate a rule per prefix, exclude the class and allow back only
# procedure durations, which are measured start to finish on every
# participant and are genuine continuous outcomes.
DURATION = re.compile(r"dur", re.I)
DURATION_OK = re.compile(r"^dursurg$|surg_?time|proc_?dur|anes|oper", re.I)
SKIP_NUM = re.compile(
    r"seq$|^age$|dy$|dyi$|days?$|years?$|^visitdy$|_dt$|date|studyid|siteid"
    r"|^trtpn$|^trtan$|n$|cd$|fl$|yn$|code"
    # Participant and record identifiers are never endpoints.  A file whose
    # own key differs from the arm file's key would otherwise contribute its
    # identifier as an outcome.
    r"|(^|_)(id|mask_?id|subjid|usubjid|ptid|patid|record)($|_)|_id$|^id$",
    re.I,
)


def _col(df: pd.DataFrame, names: list[str]) -> str | None:
    up = {str(c).upper(): c for c in df.columns}
    for n in names:
        if n in up:
            return up[n]
    return None


def _subject_col(df: pd.DataFrame) -> str | None:
    """Subject identifier, falling back to any unique id-like column.

    Deposited trial files do not always use a CDISC name.  The ACOSOG
    Z1031 export keys on `mask_id`, and requiring a standard name silently
    dropped that trial, which carries a primary efficacy endpoint.  The
    fallback accepts any id-like column unique across rows, which is the
    defining property of a participant key in a one-row-per-patient file.
    """
    c = _col(df, SUBJ)
    if c is not None:
        return c
    for col in df.columns:
        if re.search(r"id$|^id|_id|mask|record", str(col), re.I):
            s = df[col].dropna()
            if len(s) == len(df) and s.nunique() == len(df):
                return col
    return None


def read(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix.lower() == ".sas7bdat":
            return pd.read_sas(path, format="sas7bdat", encoding="latin-1")
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return None


def run_menu(a: np.ndarray, b: np.ndarray) -> dict | None:
    """Both tests across the sample-preserving transform menu."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < MIN_PER_ARM or len(b) < MIN_PER_ARM:
        return None
    if np.std(np.concatenate([a, b])) == 0:
        return None
    rows = {}
    for name, fn in TRANSFORMS:
        try:
            x, z = fn(a, b)
            r = evaluate(x, z)
            if np.isfinite(r["t_p"]) and np.isfinite(r["mw_p"]):
                rows[name] = r
        except Exception:
            continue
    keep = {k: v for k, v in rows.items() if k in NON_DELETING}
    if "raw" not in rows or len(keep) < 3:
        return None
    base = rows["raw"]
    if not (0.02 < base["auc"] < 0.98):
        return None
    t_span = _span([v["t_p"] for v in keep.values()])
    mw_span = _span([v["mw_p"] for v in keep.values()])
    return {
        "n": int(base["n"]), "n_a": len(a), "n_b": len(b),
        "auc": base["auc"], "skew": base["skew"], "kurt": base["kurt"],
        "var_ratio": base.get("var_ratio", float("nan")),
        "t_p_raw": base["t_p"], "mw_p_raw": base["mw_p"],
        "t_p_min": min(v["t_p"] for v in keep.values()),
        "t_p_max": max(v["t_p"] for v in keep.values()),
        "mw_p_min": min(v["mw_p"] for v in keep.values()),
        "mw_p_max": max(v["mw_p"] for v in keep.values()),
        "t_span": t_span, "mw_span": mw_span, "excess_span": t_span - mw_span,
        "t_flips": len({v["t_p"] < 0.05 for v in keep.values()}) > 1,
        "mw_flips": len({v["mw_p"] < 0.05 for v in keep.values()}) > 1,
    }


def one_per_subject(df: pd.DataFrame, subj: str, val: str) -> pd.DataFrame:
    """Collapse to one row per participant, keeping the first record."""
    if subj == val:
        return pd.DataFrame(columns=[subj, val])
    d = df[[subj, val]].dropna()
    d[val] = pd.to_numeric(d[val], errors="coerce")
    d = d.dropna()
    return d.drop_duplicates(subset=[subj], keep="first")


def compare(vals: pd.DataFrame, arms: pd.DataFrame, subj: str, val: str) -> list[dict]:
    """Every pairwise arm contrast for one parameter/visit table."""
    m = vals.merge(arms, on=subj, how="inner")
    levels = [l for l, c in m["_arm"].value_counts().items() if c >= MIN_PER_ARM]
    out = []
    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            a = m.loc[m["_arm"] == levels[i], val].to_numpy(float)
            b = m.loc[m["_arm"] == levels[j], val].to_numpy(float)
            r = run_menu(a, b)
            if r:
                r["contrast"] = f"{str(levels[i])[:18]} vs {str(levels[j])[:18]}"
                out.append(r)
    return out


def arm_table(df: pd.DataFrame, subj: str, arm: str) -> pd.DataFrame:
    t = df[[subj, arm]].dropna().drop_duplicates(subset=[subj])
    t.columns = [subj, "_arm"]
    t["_arm"] = t["_arm"].astype(str).str.strip()
    return t[~t["_arm"].str.contains(NOT_RANDOMISED, na=False)]


def process(study: Path) -> list[dict]:
    # Deposits are not consistent about where the data sits.  Some unpack into
    # an `extracted/` subdirectory, others leave the files loose in the study
    # root.  Requiring `extracted/` silently dropped CALGB 40603 once and would
    # have dropped three of the four trials in the second Alliance batch, each
    # time with no error and a plausible-looking zero.  Look in both places.
    def _data(d: Path) -> list[Path]:
        if not d.exists():
            return []
        return [p for p in sorted(d.iterdir())
                if p.is_file() and p.suffix.lower() in {".sas7bdat", ".csv"}]

    files = _data(study / "extracted") or _data(study)
    if not files:
        print("  ZERO: no .sas7bdat or .csv anywhere in the study directory")
        return []

    # Locate the arm assignment covering the MOST participants.  Picking the
    # first or smallest match silently substitutes a subset file for the
    # trial: the Clovis study carries an arm column on a 66-subject duration
    # dataset as well as on its 366-subject ADSL, and taking the former would
    # discard four fifths of the trial.
    arms, subj_key, arm_src = None, None, None
    for p in files:
        df = read(p)
        if df is None or df.empty:
            continue
        s = _subject_col(df)
        if s is None:
            continue
        for a in [c for c in df.columns
                  if str(c).upper() in set(ARM) and not NOT_ASSIGNED.search(str(c))]:
            if df[a].dropna().nunique() < 2:
                continue
            cand = arm_table(df, s, a)
            if cand["_arm"].nunique() < 2 or len(cand) < 2 * MIN_PER_ARM:
                continue
            if arms is None or len(cand) > len(arms):
                arms, subj_key, arm_src = cand, s, f"{p.name}:{a}"
    if arms is None:
        print("  no arm assignment found")
        return []
    print(f"  arms from {arm_src} -> "
          f"{dict(list(arms['_arm'].value_counts().items())[:4])}")

    recs = []
    for p in files:
        if SKIP_FILE.search(p.stem):
            continue
        df = read(p)
        if df is None or df.empty:
            continue
        s = _subject_col(df)
        if s is None:
            continue
        if s != subj_key:
            df = df.rename(columns={s: subj_key})
            s = subj_key
        val, par, vis = _col(df, VALUE), _col(df, PARAM), _col(df, VISIT)

        tables: list[tuple[str, str, pd.DataFrame, str]] = []
        if val is not None:
            # Convention A: long, keyed by parameter.
            params = ([None] if par is None
                      else list(pd.unique(df[par].dropna()))[:MAX_PARAMS])
            for pc in params:
                blk = df if pc is None else df[df[par] == pc]
                visits = ([None] if vis is None
                          else list(pd.unique(blk[vis].dropna()))[:MAX_VISITS])
                for v in visits:
                    sub = blk if v is None else blk[blk[vis] == v]
                    tables.append((str(pc or val), str(v or ""),
                                   one_per_subject(sub, s, val), val))
        else:
            # Conventions B and C: endpoints in named numeric columns.
            # The subject key and any arm column must never be treated as
            # endpoints.  Selecting df[[s, c]] with c == s yields duplicate
            # columns and a DataFrame where a Series is expected.
            reserved = {str(s).upper()} | set(ARM)
            nums = [c for c in df.columns
                    if df[c].dtype.kind in "if"
                    and str(c).upper() not in reserved
                    and not SKIP_NUM.search(str(c))
                    and df[c].dropna().nunique() >= MIN_UNIQUE]
            for c in nums[:MAX_PARAMS]:
                visits = ([None] if vis is None
                          else list(pd.unique(df[vis].dropna()))[:MAX_VISITS])
                for v in visits:
                    sub = df if v is None else df[df[vis] == v]
                    tables.append((str(c), str(v or ""),
                                   one_per_subject(sub, s, c), c))

        for pname, vname, tbl, vcol in tables:
            if SKIP_PARAM.search(pname) or SKIP_PARAM.search(str(vcol)):
                continue
            if TIME_TO_EVENT.search(pname) or TIME_TO_EVENT.search(str(vcol)):
                continue
            if any(DURATION.search(x) and not DURATION_OK.search(x)
                   for x in (pname, str(vcol))):
                continue
            if len(tbl) < 2 * MIN_PER_ARM or tbl[vcol].nunique() < MIN_UNIQUE:
                continue
            for r in compare(tbl, arms, s, vcol):
                r.update({
                    "study": study.name, "file": p.name,
                    "param": pname, "visit": vname,
                    # A visit label is the authority when the file has one.
                    # The name-based rule applies only to wide files, where
                    # there is no visit column to consult; using it everywhere
                    # would move genuine post-treatment weights and pain
                    # scores into the negative-control stratum and weaken it.
                    "baseline": bool(
                        BASELINE_PAT.search(vname) if vname.strip()
                        else PRE_RANDOMISATION.search(pname)),
                })
                recs.append(r)
    return recs


def main() -> None:
    all_recs = []
    for study in sorted(p for p in PDS.iterdir() if p.is_dir()):
        print(f"\n=== {study.name} ===", flush=True)
        r = process(study)
        print(f"  {len(r)} comparisons")
        all_recs.extend(r)
    out = pd.DataFrame(all_recs)
    dest = ROOT / "results" / "pds_results.csv"
    out.to_csv(dest, index=False)
    print(f"\nwrote {dest}: {len(out)} comparisons "
          f"from {out.study.nunique() if len(out) else 0} studies")


if __name__ == "__main__":
    main()
