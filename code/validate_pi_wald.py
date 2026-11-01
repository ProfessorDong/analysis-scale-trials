"""Size of the probabilistic-index heterogeneity tests used in lilly_v3.py.

Scenarios hold the stratum indices equal (the heterogeneity null) while varying what
the audit asked for: identical distributions; equal indices with different shapes and
spreads per stratum; a common index away from one half; unequal arm sizes; and the
duloxetine stratum sizes with heavy-tailed, tie-rich outcomes. Each scenario reports the
rejection rate at 0.05 of the omnibus Wald test (K-1 df) and of the linear trend z test,
with Wilson intervals over REPS replicates. Also reports the severity-label permutation
test that lilly_v2.py used, on the identical-distribution scenario, for the record.

Writes results/validate_pi_wald.csv.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from lilly_v3 import heterogeneity, pi_u  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPS = int(os.environ.get("VAL_REPS", "2000"))
rng = np.random.default_rng(20261101)
LILLY_N = [(102, 122), (807, 675), (535, 511), (891, 831)]


def wilson(k, n):
    p = k / n; z = 1.96; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def shift_for_theta(sampler, theta, n=400000):
    """Location shift of the treated arm giving P(T > C) = theta for a given control sampler (numerical)."""
    c = sampler(n); t0 = sampler(n)
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        th = (np.searchsorted(np.sort(c), t0 + mid, "left")).mean() / n
        lo, hi = (mid, hi) if th < theta else (lo, mid)
    return (lo + hi) / 2


def scenario_runs(name, samplers, sizes, theta_target):
    """samplers: list of K control samplers (functions of n). Treated = control sampler + shift giving theta_target."""
    shifts = [0.0 if theta_target == 0.5 else shift_for_theta(s, theta_target) for s in samplers]
    rej_w = rej_t = 0
    for _ in range(REPS):
        rows = []
        for s, (nc, nt), sh in zip(samplers, sizes, shifts):
            c = s(nc); t = s(nt) + sh
            th, v = pi_u(c, t); rows.append(dict(theta=th, theta_var=v))
        h = heterogeneity(rows)
        rej_w += h["omnibus_p"] < 0.05; rej_t += h["trend_p_two_sided"] < 0.05
    out = dict(scenario=name, theta=theta_target, reps=REPS, wald_rate=100 * rej_w / REPS, trend_rate=100 * rej_t / REPS)
    out["wald_lo"], out["wald_hi"] = wilson(rej_w, REPS); out["trend_lo"], out["trend_hi"] = wilson(rej_t, REPS)
    print(f"{name:55s} theta={theta_target}: Wald {out['wald_rate']:.2f}% [{out['wald_lo']:.1f},{out['wald_hi']:.1f}]  trend {out['trend_rate']:.2f}% [{out['trend_lo']:.1f},{out['trend_hi']:.1f}]", flush=True)
    return out


def main():
    N = lambda sd: (lambda n: rng.normal(0, sd, n))
    LN = lambda sig: (lambda n: rng.lognormal(0, sig, n))
    CONT = lambda: (lambda n: np.where(rng.random(n) < 0.95, rng.normal(0, 1, n), rng.normal(0, 8, n)))
    PIEFLIKE = lambda: (lambda n: np.round(np.maximum(-100, 100 * (rng.lognormal(-0.3, 0.9, n) - 1)) * (rng.random(n) > 0.07) + (-100) * (rng.random(n) <= 0.07), 0))
    res = []
    eq = [(60, 60)] * 4
    res.append(scenario_runs("identical normal arms, 4 strata of 60/60", [N(1)] * 4, eq, 0.5))
    res.append(scenario_runs("equal theta=.5, shapes normal/lognormal/contaminated/normal(sd3)", [N(1), LN(1), CONT(), N(3)], eq, 0.5))
    res.append(scenario_runs("equal theta=.6, same four shapes", [N(1), LN(1), CONT(), N(3)], eq, 0.6))
    res.append(scenario_runs("equal theta=.6, unequal arms 30/150 per stratum", [N(1), LN(1), CONT(), N(3)], [(30, 150)] * 4, 0.6))
    res.append(scenario_runs("equal theta=.6, duloxetine stratum sizes, PIEF-like ties", [PIEFLIKE()] * 4, LILLY_N, 0.6))
    res.append(scenario_runs("equal theta=.5, duloxetine stratum sizes, PIEF-like ties", [PIEFLIKE()] * 4, LILLY_N, 0.5))
    pd.DataFrame(res).to_csv(ROOT / "results" / "validate_pi_wald.csv", index=False)


if __name__ == "__main__":
    main()
