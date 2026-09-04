# analysis-scale-trials

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22281796.svg)](https://doi.org/10.5281/zenodo.22281796)

Code and derived results for **"An unrecorded analysis choice can change what a
randomized trial concludes."**

Before a trial is analyzed someone must decide how to summarize the treatment
effect: whether to compare averages, and whether to rescale or trim unusual
values first. Protocols routinely record what will be measured and leave that
choice unstated. This repository contains everything needed to check what the
choice costs.

## Start here, no credentials needed

    pip install -r requirements.txt
    python reproduce/reproduce_worked_example.py   # public trial, ~20 s
    python reproduce/reproduce_calibration.py      # no data at all

The first reproduces Figure 2a from a publicly redistributable trial. The
second reproduces the null floor that every observed rate must be read against.
Together they cover the paper's central claim and its calibration.

## What is here

| Path | Contents |
|---|---|
| `reproduce/` | Two self-contained scripts, above |
| `code/` | The full analysis pipeline |
| `results/` | Derived summary results, one row per comparison |
| `derived/` | The data files and generated numbers behind every figure and table |

`code/RUN_ORDER.md` gives the order to run things in and what each step writes.

## What is not here, and why

**No participant-level data.** Two of the three trial sources are held under
data use agreements that permit analysis and the publication of aggregate
results but not redistribution of records:

- **Project Data Sphere**, 19 oncology and prevention trials
- **A duloxetine trial in stress urinary incontinence**, obtained separately

`results/` therefore holds endpoint-level summaries only: for each arm
comparison, the sample sizes, the probabilistic index, skewness, kurtosis, the
ratio of arm standard deviations, and the P-value under each specification.
These reproduce every figure and table in the paper and identify no one.

The third source, **14 trials deposited on Dryad under CC0**, is public and
identified by DOI in the paper's Supplementary Information.

## Reproducing the paper's numbers

Every number in the manuscript is generated rather than typed.
`code/headline_numbers.py` writes a file of LaTeX macros that the manuscript
and its Supplementary Information both read, along with the data files behind
the figures. Rerunning the pipeline and recompiling is the only supported way
to change a number; editing one in the manuscript source is not.

The full pipeline needs the restricted data. Without it, `results/` and
`derived/` let you verify every reported quantity, and the two scripts in
`reproduce/` let you rerun the phenomenon itself from scratch.

## Requirements

Python 3.11 or later with numpy, scipy and pandas. See `requirements.txt` or
`environment.yml`. No GPU, no compiled extensions; the full simulation takes
about eight minutes on one core.

## Citation

Archived on Zenodo: https://doi.org/10.5281/zenodo.22281796

See `CITATION.cff` for the full citation metadata.

## License

MIT, see `LICENSE`. The license covers the code and the derived summary
results in this repository. It does not extend to the underlying trial data,
which remain governed by their respective agreements.
