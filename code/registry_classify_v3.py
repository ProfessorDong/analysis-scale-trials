"""Endpoint-type and method codebook for the registry census (audit items E05, E06).

Reads data/ctgov/primary_measures_v2.csv (the full census of interventional, randomized,
completed trials with posted results) and adds

  endpoint_type   quantitative | time_to_event | count_or_rate | proportion | unclear
                  from the measure title, unit and parameter type, by the rules below
  method_class3   mean-based | rank-based | survival | other | both | not stated
                  where log-rank, Cox and hazard-ratio methods form their own class

and reports the quantities the paper may cite, with their denominators, at the measure
and the trial level. It also draws a validation sample of measures for reading.

Rules (regular expressions on title and unit, case-insensitive):
  time_to_event: survival | time to | time-to | progression.free | disease.free | event.free |
                 relapse.free | recurrence.free | duration of response | time until |
                 the words months/weeks/days/years as the UNIT together with a MEDIAN or
                 a survival method
  count_or_rate: unit begins with "number of" or contains count | rate | episodes |
                 exacerbations | events, and the parameter is a mean of counts
  proportion:    unit mentions participants, percentage of participants, proportion
  quantitative:  everything else with a mean/median/LS-mean/geometric-mean parameter
Writes data/ctgov/primary_measures_v3.csv, data/ctgov/registry_validation_sample.csv and
prints the summary.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CT = ROOT / "data" / "ctgov"

TTE_WORDS = re.compile(r"surviv(al|e)\b|progression.?free|disease.?free|event.?free|relapse.?free|recurrence.?free|duration of response|time to (death|progression|relapse|recurrence|event|failure|first)|\b(PFS|DFS|EFS|RFS|TTP)\b|\bOS\b", re.I)
TIME_TO = re.compile(r"time to |time-to|time until|time from .* to|hours until|days until", re.I)
TIME_UNIT = re.compile(r"^(months?|weeks?|days?|years?|hours?)\b", re.I)
COUNT_UNIT = re.compile(r"^number of|count|\brate\b|episodes|exacerbations|events|occurrences|per (100|1000|person|patient)", re.I)
PROP_UNIT = re.compile(r"(percentage|proportion|number|percent) of (participants|patients|subjects|women|men|children|infants|eyes)|^participants$|^subjects$", re.I)
MEAN_BASED = re.compile(r"t.?test|ancova|anova|mmrm|least.?squares|linear (model|regression)|regression, linear|repeated measures|mixed(?! effects logistic)|gee|generalized estimating", re.I)
RANK_BASED = re.compile(r"wilcoxon|mann.?whitney|kruskal|jonckheere|van elteren|rank.?sum|signed.?rank|hodges", re.I)
SURVIVAL = re.compile(r"log.?rank|\bcox\b|proportional.?hazard|hazard ratio|kaplan", re.I)


def endpoint_type(row) -> str:
    title, unit, pt = str(row.title), str(row.unit), str(row.param_type)
    methods = str(row.methods)
    survival_method = bool(SURVIVAL.search(methods))
    if TTE_WORDS.search(title) or TTE_WORDS.search(unit) or survival_method:
        return "time_to_event"
    if TIME_TO.search(title) and (pt == "MEDIAN" or TIME_UNIT.search(unit)) and pt != "MEAN":
        return "time_to_event"
    if PROP_UNIT.search(unit):
        return "proportion"
    if COUNT_UNIT.search(unit):
        return "count_or_rate"
    return "quantitative"


def method_class3(methods: str) -> str:
    text = str(methods)
    if not text.strip() or text == "nan":
        return "not stated"
    surv, mean, rank = bool(SURVIVAL.search(text)), bool(MEAN_BASED.search(text)), bool(RANK_BASED.search(text))
    if surv and not (mean or rank):
        return "survival"
    if mean and rank:
        return "both"
    if mean:
        return "mean-based"
    if rank:
        return "rank-based"
    return "other"


def main() -> None:
    reg = pd.read_csv(CT / "primary_measures_v2.csv", low_memory=False)
    reg["endpoint_type"] = reg.apply(endpoint_type, axis=1)
    reg["method_class3"] = reg.methods.map(method_class3)
    reg["quant_cont"] = reg.continuous & (reg.endpoint_type == "quantitative")
    reg.to_csv(CT / "primary_measures_v3.csv", index=False)
    n_meas, n_tr = len(reg), reg.nct.nunique()
    print(f"frame: {n_tr:,} trials, {n_meas:,} primary measures")
    print("endpoint_type among measures with a continuous parameter type:", reg[reg.continuous].endpoint_type.value_counts().to_dict())
    q = reg[reg.quant_cont]
    print(f"\nquantitative (uncensored, non-count, non-proportion) continuous measures: {len(q):,} / {n_meas:,} = {100*len(q)/n_meas:.1f}%")
    tq = reg.groupby("nct").quant_cont.any()
    print(f"trials with at least one such measure: {tq.sum():,} / {n_tr:,} = {100*tq.mean():.1f}%")
    stated = q[q.method_class3 != "not stated"]
    print(f"\nwith a stated method: {len(stated):,} / {len(q):,} = {100*len(stated)/len(q):.1f}%")
    print("method_class3 among those (measure level):", (100 * stated.method_class3.value_counts(normalize=True)).round(1).to_dict())
    tr = stated.groupby("nct").method_class3.agg(lambda s: set(s))
    any_mean = tr.map(lambda S: bool(S & {"mean-based", "both"})).mean(); any_rank = tr.map(lambda S: bool(S & {"rank-based", "both"})).mean()
    print(f"trial level, {len(tr):,} trials with a quantitative primary and a stated method: any mean-based {100*any_mean:.1f}%, any rank-based {100*any_rank:.1f}%")
    print(f"geometric-mean summary among quantitative measures: {100*q.log_scale_summary.mean():.1f}%")
    print("\nold rank-based class now classed:", reg[(reg.continuous) & (reg.method_class == "rank-based")].method_class3.value_counts().to_dict())
    print("MEDIAN-parameter measures by endpoint_type:", reg[reg.param_type == "MEDIAN"].endpoint_type.value_counts().to_dict())
    # validation sample: 40 measures stratified over the four types, for reading
    rng = np.random.default_rng(20261101)
    parts = [reg[reg.continuous & (reg.endpoint_type == t)].sample(n=min(10, (reg.continuous & (reg.endpoint_type == t)).sum()), random_state=1) for t in ["quantitative", "time_to_event", "count_or_rate", "proportion"]]
    val = pd.concat(parts)[["nct", "param_type", "unit", "title", "methods", "endpoint_type", "method_class3"]]
    val.to_csv(CT / "registry_validation_sample.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
    print("\nVALIDATION SAMPLE:\n", val.to_string(index=False))


if __name__ == "__main__":
    main()
