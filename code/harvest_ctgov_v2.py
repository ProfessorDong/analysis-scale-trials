"""Registry census for the merged paper, with the denominators the review asked for.

Differences from harvest_ctgov.py:
  * the query requires interventional, randomized studies with posted results
    and completed status, so the frame is 'completed randomized trials with
    results', not 'anything with results';
  * every page is taken, so this is a census of that frame, not the first
    12,000 records;
  * allocation and study type are kept on every row so the frame can be
    verified locally;
  * one row per PRIMARY outcome measure, with its analyses' statistical methods
    aggregated, so that trial-level, measure-level and analysis-level
    proportions are all computable with the right denominator;
  * the classification of parameter types and statistical methods is in this
    file, with the regular expressions visible.

Outputs data/ctgov/primary_measures_v2.csv and prints the headline
proportions with their denominators.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "ctgov"
API = "https://clinicaltrials.gov/api/v2/studies"
PAGE = 1000
PAUSE = 0.4
QUERY = "AREA[HasResults]true AND AREA[StudyType]INTERVENTIONAL AND AREA[DesignAllocation]RANDOMIZED"
FIELDS = ["NCTId", "OverallStatus", "Phase", "StudyType", "DesignAllocation", "DesignInterventionModel",
          "EnrollmentCount", "StartDate", "ResultsFirstPostDate", "OutcomeMeasuresModule", "LargeDocumentModule"]

CONTINUOUS_PARAM = {"MEAN", "MEDIAN", "GEOMETRIC_MEAN", "LEAST_SQUARES_MEAN", "GEOMETRIC_LEAST_SQUARES_MEAN"}
LOG_PARAM = {"GEOMETRIC_MEAN", "GEOMETRIC_LEAST_SQUARES_MEAN"}
MEAN_BASED = re.compile(r"t.?test|ancova|anova|mixed|mmrm|least.?squares|linear (model|regression)|regression, linear|repeated measures|gee|generalized estimating", re.I)
RANK_BASED = re.compile(r"wilcoxon|mann.?whitney|kruskal|jonckheere|rank|van elteren|hodges", re.I)


def fetch_page(token):
    params = {"query.term": QUERY, "filter.overallStatus": "COMPLETED", "pageSize": PAGE, "fields": ",".join(FIELDS)}
    if token:
        params["pageToken"] = token
    for attempt in range(6):
        try:
            r = requests.get(API, params=params, timeout=180)
            if r.ok:
                return r.json()
            print(f"  HTTP {r.status_code}: {r.text[:200]}", flush=True)
        except Exception as e:
            print(f"  error {e}", flush=True)
        time.sleep(2 ** attempt)
    return {}


def method_class(methods: list[str]) -> str:
    text = " | ".join(m for m in methods if m)
    if not text.strip():
        return "not stated"
    if RANK_BASED.search(text) and not MEAN_BASED.search(text):
        return "rank-based"
    if MEAN_BASED.search(text) and not RANK_BASED.search(text):
        return "mean-based"
    if MEAN_BASED.search(text) and RANK_BASED.search(text):
        return "both"
    return "other"


def rows_from_study(s: dict) -> list[dict]:
    ps = s.get("protocolSection", {})
    nct = ps.get("identificationModule", {}).get("nctId", "")
    design = ps.get("designModule", {})
    info = design.get("designInfo", {}) or {}
    docs = (s.get("documentSection", {}).get("largeDocumentModule", {}).get("largeDocs", []) or [])
    has_sap = any("SAP" in (d.get("typeAbbrev") or "") for d in docs)
    oms = (s.get("resultsSection", {}).get("outcomeMeasuresModule", {}).get("outcomeMeasures", []) or [])
    out = []
    for i, om in enumerate(oms):
        if om.get("type") != "PRIMARY":
            continue
        analyses = om.get("analyses") or []
        methods = [(a.get("statisticalMethod") or "").strip() for a in analyses]
        pt = om.get("paramType", "") or ""
        out.append({
            "nct": nct, "status": ps.get("statusModule", {}).get("overallStatus", ""),
            "study_type": design.get("studyType", ""), "allocation": info.get("allocation", ""),
            "model": info.get("interventionModel", ""), "phase": ",".join(design.get("phases", []) or []),
            "enrollment": (design.get("enrollmentInfo", {}) or {}).get("count"),
            "start": ps.get("statusModule", {}).get("startDateStruct", {}).get("date", ""),
            "results_posted": ps.get("statusModule", {}).get("resultsFirstPostDateStruct", {}).get("date", ""),
            "has_sap": has_sap, "measure_index": i, "param_type": pt, "dispersion": om.get("dispersionType", ""),
            "unit": (om.get("unitOfMeasure") or "")[:60], "title": (om.get("title") or "")[:120],
            "n_analyses": len(analyses), "methods": " | ".join(m for m in methods if m)[:200],
            "continuous": pt in CONTINUOUS_PARAM, "log_scale_summary": pt in LOG_PARAM,
            "method_class": method_class(methods),
        })
    return out


def main() -> None:
    token, rows, pages, n_studies = None, [], 0, 0
    while True:
        j = fetch_page(token)
        studies = j.get("studies", [])
        if not studies:
            break
        n_studies += len(studies)
        for s in studies:
            rows.extend(rows_from_study(s))
        pages += 1
        token = j.get("nextPageToken")
        print(f"  page {pages}: {n_studies} studies, {len(rows)} primary measures", flush=True)
        if not token:
            break
        time.sleep(PAUSE)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "primary_measures_v2.csv", index=False)
    print(f"\nwrote {len(df)} primary measures from {df.nct.nunique()} trials ({n_studies} studies fetched)")
    frame_ok = (df.study_type == "INTERVENTIONAL") & (df.allocation == "RANDOMIZED")
    print(f"frame check: {frame_ok.mean():.4f} of rows are interventional+randomized")
    trials = df.groupby("nct").agg(any_cont=("continuous", "any"))
    print(f"trials with at least one continuous primary measure: {trials.any_cont.sum()} / {len(trials)} = {trials.any_cont.mean():.1%}")
    print(f"primary measures continuous: {df.continuous.sum()} / {len(df)} = {df.continuous.mean():.1%}")
    cont = df[df.continuous]
    stated = cont[cont.method_class != "not stated"]
    print(f"continuous measures with a stated method: {len(stated)} / {len(cont)} = {len(stated) / len(cont):.1%}")
    print("method class among those:", (stated.method_class.value_counts(normalize=True) * 100).round(1).to_dict())
    print(f"continuous measures with geometric (log-scale) summary: {cont.log_scale_summary.mean():.1%}")
    tr_cont = cont.groupby("nct").agg(mean=("method_class", lambda s: (s == "mean-based").any()), rank=("method_class", lambda s: (s == "rank-based").any()), stated=("method_class", lambda s: (s != "not stated").any()))
    ts = tr_cont[tr_cont.stated]
    print(f"trial-level, among {len(ts)} trials with a continuous primary and a stated method: any mean-based {ts['mean'].mean():.1%}, any rank-based {ts['rank'].mean():.1%}")


if __name__ == "__main__":
    main()
