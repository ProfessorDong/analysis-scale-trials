# Order to run the Paper 2 pipeline

Any change to the trial data or the filters requires all four, in order.

```bash
cd Paper2_Reanalysis/code
python simulate_span.py      # 108 conditions x 2,000 reps -> results/simulation_spans.csv (~8 min)
python analyze_pds.py        # trial comparisons  -> results/pds_results.csv
python audit_pds_yield.py    # why each variable was kept or dropped
python headline_numbers.py   # -> NaturePortfolio/data/numbers.tex, .dat files, SI tables
python cluster_inference.py  # clustered inference, printed report
python figure1_data.py       # Figure 1 densities + its label macros (independent of the above)
python figure2_data.py       # Figure 2 specification curves + its label macros
python calgb40603_dichotomy.py # CALGB 40603 dichotomization boundary (needs the DUA data)
python check_si_refs.py      # hard-coded Supplementary Note numbers still point where they should
cd ../../NaturePortfolio/figures && for f in fig1 fig2 fig3 fig4 fig5; do pdflatex $f.tex; done
cd ../si && for f in sfig1 sfig2 sfig3; do pdflatex $f.tex; done
cd .. && pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
cd si && pdflatex natcomms_si.tex && bibtex natcomms_si && pdflatex natcomms_si.tex
```

`simulate_span.py` only needs rerunning when the generating models or the menu change.

`headline_numbers.py` is the only thing that may write a number into the manuscript.
Editing a figure in a `.tex` file by hand reintroduces the drift it exists to prevent.

Every figure is now generated: `headline_numbers.py` writes `simcurve.dat`,
`quartiles.dat`, `flips.dat` and `negcontrol.dat` for figures 4 and 5, including
their cluster bootstrap intervals. Figures 1 and 2 are fully generated: `figure1_data.py` and `figure2_data.py` write both their data and the
labels drawn on them, so panels and captions cannot disagree. `figure2_data.py`
rebuilds panel 2a from the public Beat the Blues CSV, reusing the menu defined in
`multistat-roc/reproduce/reproduce_worked_example.py`; panel 2b comes from the
stored per-specification results, the trial itself being under a data use
agreement.

## Figure files and printed numbers

`figN.tex` is printed as Figure N. They were briefly out of step: `fig5.pdf` was
included before `fig4.pdf`, so the file named fig4 appeared as Figure 5. If you
add or move a figure, check `pdftotext paper.pdf - | grep "^Fig\."` against the
filenames.
