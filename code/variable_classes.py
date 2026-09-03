"""Shared classification of trial variables, and the loader that applies it.

Two questions have to be answered the same way for every data source, or the
pooled analysis is incoherent:

  1. Is this variable an outcome at all?  Identifiers, design labels and
     allocation records are not, and each of them has at some point entered
     the screen and produced a spurious result.
  2. Was it measured before randomisation?  If so it cannot carry a treatment
     effect, which disqualifies it from the primary analysis and qualifies it
     as a negative control.

Project Data Sphere trials are classified inside analyze_pds.py, which sees
the visit labels.  Dryad tables carry no visit column, so they are classified
here by name.  Both routes end in load_trials(), so the manuscript, the
figures and the clustered inference all draw on one definition.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results"

# Not outcomes at all: participant, family and site identifiers, the cluster
# label in a cluster-randomised trial, the assessor, and counts of the
# intervention itself.
NOT_AN_OUTCOME = re.compile(
    r"^(?:cluster|family|dz|el_assessor|interventionnumber|total_recs|date_diff)$"
    r"|(?:^|_)id$|^id|_id$|assessor|^record",
    re.I,
)

# Measured before randomisation.  Age, baseline laboratory values, smoking
# history, enrolment timing and birth characteristics in trials whose
# intervention begins after birth.
PRE_RANDOMISATION_NAME = re.compile(
    r"^age|_age$|agemons|agesyr|momage|mompar|parity"
    r"|pack_?years|years_?smoke"
    r"|^bl_|^baseline|_baseline$|baseline_|_enroll$|enrollment_days|^el_month"
    r"|admit$|^screening|^scr_"
    r"|^bw$|^ga$|povertscore|wealth",
    re.I,
)


def load_trials() -> pd.DataFrame:
    """All trial comparisons with `baseline` and `source` set consistently.

    Returns every comparison; callers split on `baseline` themselves so that
    the negative-control stratum is never silently dropped.
    """
    pds = pd.read_csv(RES / "pds_results.csv")
    pds = pds.assign(trial=pds.study, source="PDS",
                     variable=pds.param.astype(str))

    dry = pd.read_csv(RES / "screen_dryad.csv")
    name = dry.outcome_var.astype(str)
    dry = dry.assign(trial=dry.doi, source="Dryad", variable=name,
                     baseline=name.str.contains(PRE_RANDOMISATION_NAME))
    dry = dry[~name.str.contains(NOT_AN_OUTCOME)]

    tr = pd.concat([pds, dry], ignore_index=True)
    tr["baseline"] = tr.baseline.fillna(False).astype(bool)
    return tr[np.isfinite(tr.excess_span)].copy()
