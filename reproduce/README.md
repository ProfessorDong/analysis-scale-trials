# Reproduction without credentials

Two scripts here reproduce central claims of the paper on a laptop, with no
data use agreement and no local data files.

## `reproduce_worked_example.py`

Downloads the Beat the Blues trial, a randomized comparison of computerized
cognitive behavioral therapy against treatment as usual, from the public
Rdatasets mirror. It applies the five pre-specified, sample-preserving
specifications and prints the table behind Figure 2a.

Expected output: the mean-based P-value crosses 0.05 across the menu, moving
from 0.029 under winsorizing to 0.353 on a log scale, while the rank-based
P-value moves by less than 0.02 orders of magnitude.

    python reproduce_worked_example.py

This is the paper's core phenomenon, on public data, in about twenty seconds.

## `reproduce_calibration.py`

Needs no data at all. Simulates normal arms with equal dispersion and no
treatment effect, and reports how often each estimator's verdict at alpha =
0.05 changes across the same menu. This establishes the floor that all observed
rates must be read against: near 4.4% for the mean-based analysis and near 0.1%
for the rank-based one.

    python reproduce_calibration.py

Both scripts require only numpy, scipy and pandas.
