# analysis-scale-trials

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22281795.svg)](https://doi.org/10.5281/zenodo.22281795)

Code, manifests, ledgers and derived results for **"The analysis scale and the
treatment-effect summary: scale-invariant ROC summaries and a comparative
reanalysis of randomized trials"** (Meng, Tubbs and Dong; submitted to *The
Annals of Applied Statistics*).

A randomized trial with a continuous endpoint must choose a population-level
summary of the treatment effect. A difference in means is not fully specified
until the scale on which the mean is taken is fixed; the receiver operating
characteristic curve of treated against control outcomes, conditional on
covariates, is exactly invariant to every strictly increasing transformation,
and so are its area (the probabilistic index) and its departure from the
diagonal. This repository holds everything needed to rebuild the paper's
multi-trial reanalysis (312 post-randomization comparisons from 20 randomized
trials, 406 pre-randomization negative controls from 24), its simulation, its
duloxetine case study and its registry census, and to regenerate every number,
table and figure in the manuscript from the aggregate results.

Release v1.0.0 of this repository accompanied an earlier, unsubmitted draft
with a different comparison set and is superseded by v2.0.0.

## What is here

| Path | Contents |
|---|---|
| `manifest/RULES.md` | The seventeen written rules under which the comparison set was built |
| `manifest/trials/*.yaml` | One manifest per dataset (33): allocation variable, participant key, tables, variable roles and timing, evidence notes |
| `manifest/write_manifests.py` | Writes the manifests from the adjudicated specifications |
| `code/` | The pipeline: `build_comparisons.py`, `analyze_v2.py`, `simulate_v2.py`, `lilly_v3.py`, `validate_pi_wald.py`, `harvest_ctgov_v2.py`, `registry_classify_v3.py`, and the generators `numbers_v2.py`, `figures_v2.py`, `supp_tables_v2.py` |
| `results/` | One row per comparison (`comparisons_v2.csv`), the trial table, the selection and exclusion ledgers, trial-level inference, simulation results, size validation, duloxetine aggregates, build logs |
| `registry/` | The classified ClinicalTrials.gov census (92,849 primary measures from 41,245 trials) and the hundred-measure reading sample with judgments |
| `manuscript_generators/` | `numbers.tex`, the macros the manuscript reads, and `flatten.py` |
| `RUN_ORDER.md` | The order to run things in, with and without the restricted data |
| `CHANGELOG.md` | The amendment record of the comparison set |

## Start here, no credentials needed

    pip install -r requirements.txt
    cd code
    python numbers_v2.py        # every macro the manuscript quotes, from results/
    python supp_tables_v2.py    # the supplement tables
    python simulate_v2.py       # optional: the simulation grid (SIM_REPS=2000, about an hour)
    python validate_pi_wald.py  # optional: size of the heterogeneity tests

`results/comparisons_v2.csv` holds, for every arm contrast, endpoint and
occasion, the sample sizes, the Welch and Mann-Whitney p-values, the mean
difference and the area under each of the seven specifications, the empirical
ROC quantities and the permutation p-value of the Cramer-von Mises departure.
`results/selection_ledger.csv` gives every candidate comparison with its
disposition and `results/exclusion_ledger.csv` every candidate variable.

## What is not here, and why

No participant-level data. Two of the three sources are held under data use
agreements that permit the publication of aggregate results but not the
redistribution of records: nineteen Project Data Sphere trials, and a pooled
duloxetine dataset in stress urinary incontinence. The fourteen Dryad deposits
are public under CC0 and are identified by DOI in the manifests. Rebuilding the
comparison set from participant-level data therefore requires the agreements;
`RUN_ORDER.md` describes both routes.

## Requirements

Python 3.12 or later with numpy, scipy, pandas and pyyaml (`requirements.txt`,
`environment.yml`). No GPU, no compiled extensions. The environment used was
Python 3.12.13, NumPy 2.2.6, SciPy 1.17.1, pandas 3.0.2 on Linux.

## Citation

Archived on Zenodo under the concept DOI https://doi.org/10.5281/zenodo.22281795,
which resolves to the latest version. See `CITATION.cff`.

## License

MIT, see `LICENSE`. The license covers the code, manifests and derived results in
this repository. It does not extend to the underlying trial data, which remain
governed by their respective agreements.
