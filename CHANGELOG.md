# Change log of the comparison set

All dates 2026. The rules are in `manifest/RULES.md`; each manifest in `manifest/trials/`
carries its evidence notes. Seeds are fixed in the scripts.

- **31 October, first build.** Rules 1 to 14, 16 and 17 written before any comparison was
  recomputed; 33 datasets manifested; 314 post-randomization comparisons from 20 trials.
- **31 October, rule 15 added.** A numerical audit found eight comparisons in which
  floating-point noise in deposited values (for example `-0.331` beside
  `-0.33099999999999996`) produced ranks that a shifted logarithm then merged, so the
  Mann-Whitney statistic differed across strictly increasing specifications. Rule 15 rounds
  the pooled values of a comparison to nine significant figures relative to their largest
  magnitude before any specification is applied. Rebuilt; the identity now holds in every
  comparison and is asserted in code.
- **1 November, two defects repaired without changing the rules.** (i) A measurement series
  stored in one file per visit (adolescent weight-loss trial) had been treated as separate
  endpoints, so two visits had both counted as first visits; series are now identified by
  endpoint within trial across files. (ii) A site-local participant number (NTrainer) had been
  treated as unique, dropping three participants; participant keys may now be composite.
  Rebuilt: 312 post-randomization comparisons from 20 trials, 275 latest-visit comparisons
  from 13, 406 pre-randomization comparisons from 24. `results/build_v3.log` and
  `results/analyze_v3.log` are the logs of this build.
- **1 November, duloxetine analysis.** `lilly_v3.py` replaces the earlier script: the file is
  described as a pooled complete-case dataset of ten protocol codes; the severity-label
  permutation trend test was removed as uncalibrated; heterogeneity is tested on stratum
  probabilistic indices with two-sample U-statistic variances (`validate_pi_wald.py` checks
  size); the mean-based interaction applies one common transformation to the whole cohort
  with HC3 covariance.
- **1 November, registry.** `registry_classify_v3.py` adds the endpoint codebook and a method
  class with log-rank, Cox and Kaplan-Meier methods in their own class; the reading sample of
  one hundred measures with judgments is in `registry/`.
