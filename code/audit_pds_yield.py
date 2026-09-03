"""Why each Project Data Sphere trial does or does not contribute comparisons.

A trial that yields nothing is not necessarily a failure of the pipeline.  An
oncology trial whose only post-randomisation measurements are censored
survival times has no continuous endpoint of the kind this study concerns,
and that is a fact about the trial rather than a bug.  Distinguishing the two
matters: a silent zero has three times turned out to be a defect, and a
defect that looks like a substantive finding is the worst kind.

Writes one row per numeric column per trial with the reason it was kept or
dropped, so the scope claim in the manuscript rests on counted evidence.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import analyze_pds as A  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def reason(col: str, s: pd.Series) -> str:
    c = str(col)
    if A.SKIP_PARAM.search(c):
        return "exposure or dose"
    if A.TIME_TO_EVENT.search(c):
        return "censored time-to-event"
    if A.DURATION.search(c) and not A.DURATION_OK.search(c):
        return "duration, censored or exposure"
    if A.SKIP_NUM.search(c):
        return "identifier, date or flag"
    n = s.dropna().nunique()
    if n < A.MIN_UNIQUE:
        return f"categorical or too coarse ({n} distinct)"
    if A.PRE_RANDOMISATION.search(c):
        return "ELIGIBLE (pre-randomisation, negative control)"
    return "ELIGIBLE"


def main() -> None:
    rows = []
    for study in sorted(p for p in A.PDS.iterdir() if p.is_dir()):
        ex = study / "extracted"
        files = ([p for p in sorted(ex.iterdir())
                  if p.is_file() and p.suffix.lower() in {".sas7bdat", ".csv"}]
                 if ex.exists() else [])
        if not files:
            files = [p for p in sorted(study.iterdir())
                     if p.is_file() and p.suffix.lower() in {".sas7bdat", ".csv"}]
        for p in files:
            if A.SKIP_FILE.search(p.stem):
                continue
            df = A.read(p)
            if df is None or df.empty:
                continue
            for c in df.columns:
                if df[c].dtype.kind not in "if":
                    continue
                if str(c).upper() in set(A.ARM):
                    continue
                rows.append({"study": study.name, "file": p.name, "column": str(c),
                             "n_distinct": int(df[c].dropna().nunique()),
                             "reason": reason(c, df[c])})
    out = pd.DataFrame(rows)
    dest = ROOT / "results" / "pds_yield_audit.csv"
    out.to_csv(dest, index=False)

    per = (out.assign(elig=out.reason.str.startswith("ELIGIBLE"))
              .groupby("study").elig.agg(["sum", "size"])
              .rename(columns={"sum": "eligible_cols", "size": "numeric_cols"}))
    print(per.to_string())
    print("\nreasons across all trials:")
    print(out.reason.str.replace(r" \(.*", "", regex=True).value_counts().to_string())
    print(f"\ntrials with no eligible column: "
          f"{int((per.eligible_cols == 0).sum())} of {len(per)}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
