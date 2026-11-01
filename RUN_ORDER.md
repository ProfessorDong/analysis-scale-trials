# Reproducing the article

Two routes. The first needs no credentials and rebuilds every number, table and figure of
the article from the aggregate results in `results/`. The second rebuilds the aggregate
results themselves from participant-level trial data, which cannot be redistributed.

## Route 1: from published aggregates (no credentials)

```bash
pip install -r requirements.txt
cd code
python numbers_v2.py          # -> AOAS/numbers.tex, every macro the manuscript quotes
python figures_v2.py          # -> figure data (.dat) for the TikZ sources; the Lilly ROC
                              #    panel needs the restricted file and is skipped without it
python supp_tables_v2.py      # -> supplement table fragments
python simulate_v2.py         # optional: regenerates results/simulation_v2*.csv (SIM_REPS=2000, ~1 h)
python validate_pi_wald.py    # optional: regenerates results/validate_pi_wald.csv (VAL_REPS=2000)
python registry_classify_v3.py  # re-applies the endpoint codebook to registry/primary_measures_*.csv
```

`results/comparisons_v2.csv` holds one row per arm contrast per endpoint per occasion with
both tests under all seven specifications, the mean difference and area under each, the ROC
quantities, and the permutation p-value; `selection_ledger.csv` gives every candidate
comparison and its disposition; `exclusion_ledger.csv` every candidate variable.

## Route 2: from participant-level data (requires agreements)

Project Data Sphere studies (data use agreement) go under `data/ProjectDataSphere/<study>/`
with their deposited files; Dryad deposits (CC0) under `data/raw/dryad/` with the file
naming `doi_<doi>__<file>`; the duloxetine dataset (data use agreement) at
`../data/lilly_baylor_yan.sas7bdat`. Then:

```bash
cd manifest && python write_manifests.py      # trials/*.yaml from the adjudicated specifications
cd ../code
python build_comparisons.py                   # -> results/comparisons_v2.csv and the ledgers (seed 20261031, 999 permutations)
python analyze_v2.py                          # -> results/v2_summary.json (trial-level inference, cluster bootstrap)
python lilly_v3.py                            # -> results/lilly_v3.json
python harvest_ctgov_v2.py                    # re-harvests the registry census (network; the frame will have grown)
```

Rules are in `manifest/RULES.md`; every manifest carries its evidence notes. Seeds are fixed
in each script. The environment used was Python 3.12.13, NumPy 2.2.6, SciPy 1.17.1, pandas
3.0.2 on Linux.
