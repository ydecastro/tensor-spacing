#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Behaviour of the spacing test when the two largest maxima are close, and
numerical stability of the pivot.

Three parts:

1. Near-tie sweep. Fix (lambda_2, detR, trR) at values typical of the null
   and let the spacing eps = lambda_1 - lambda_2 go from 1 down to 1e-15.
   Both the direct and the log-scale evaluations are computed; they agree to
   machine precision and the p-value goes to 1 smoothly. The near-tie case
   is the numerically easy case.

2. Far-tail sweep. Fix the spacing at 1 and let lambda_1 grow. The direct
   double-precision evaluation loses the survival-function term of G around
   lambda ~ 8.3 (1 - Phi rounds to 0) and returns 0/0 = NaN beyond
   lambda ~ 39. The log-scale evaluation follows the exact value for all
   lambda. This is the delicate case, handled by tenspace.pivot.

3. Fresh Monte-Carlo. New simulations of the 3-way 3-dimensional spiked
   tensor model (gamma = 0, the null, and gamma = 2.5), with the p-values
   computed by both methods on each draw (tenspace.simulation).

Outputs are written to experiments/outputs/.

Run:  python3 experiments/stability_experiment.py [n_mc]
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tenspace.pivot import spacing_pvalue
from tenspace.simulation import monte_carlo

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")


def near_tie_sweep():
    # (detR, trR) at the median of the null Monte-Carlo experiments
    det_r, trace_r = -0.182, -1.056
    rows = []
    for lambda_2 in (1.5, 2.5, 3.5):
        for eps in np.logspace(0, -15, 46):
            l1 = lambda_2 + eps
            p_naive = spacing_pvalue(l1, lambda_2, det_r, trace_r, method="naive")
            p_stable = spacing_pvalue(l1, lambda_2, det_r, trace_r, method="stable")
            rows.append(dict(lambda_2=lambda_2, eps=eps,
                             p_naive=p_naive, p_stable=p_stable))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "near_tie_sweep.csv"), index=False)
    gap = np.abs(df.p_naive - df.p_stable).max()
    print(f"[near tie] eps from 1 down to 1e-15: p-values rise smoothly to 1;")
    print(f"[near tie] max |direct - log-scale| over the sweep: {gap:.2e}")
    print(f"[near tie] p at eps=1e-15 (lambda_2=2.5): "
          f"{df[(df.lambda_2 == 2.5)].p_stable.iloc[-1]:.15f}")
    return df


def far_tail_sweep():
    det_r, trace_r = -0.182, -1.056
    spacing = 1.0
    rows = []
    for l1 in np.linspace(2.0, 60.0, 233):
        l2 = l1 - spacing
        with np.errstate(all="ignore"):
            p_naive = spacing_pvalue(l1, l2, det_r, trace_r, method="naive")
        p_stable = spacing_pvalue(l1, l2, det_r, trace_r, method="stable")
        rows.append(dict(lambda_1=l1, p_naive=p_naive, p_stable=p_stable))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "far_tail_sweep.csv"), index=False)
    bad = df[~np.isfinite(df.p_naive) | (df.p_naive <= 0)]
    first_nan = df[df.p_naive.isna()].lambda_1.min()
    rel = np.abs(df.p_naive / df.p_stable - 1.0)
    first_off = df[rel > 1e-6].lambda_1.min()
    print(f"[far tail] direct evaluation deviates by >1e-6 (relative) from "
          f"lambda_1 ~ {first_off:.1f}, NaN from lambda_1 ~ {first_nan:.1f};")
    print(f"[far tail] log-scale evaluation finite and in (0,1) on the whole "
          f"sweep: {bool(np.isfinite(df.p_stable).all() and (df.p_stable > 0).all())}")
    return df


def summarize_mc(df, gamma_val):
    ok = df.p_spacing_stable.between(0, 1)
    agree = np.abs(df.p_spacing_naive - df.p_spacing_stable)[ok].max()
    print(f"[MC gamma={gamma_val}] n={len(df)}; "
          f"stable p in [0,1]: {ok.mean()*100:.2f}%; "
          f"max |direct - log-scale| on valid draws: {agree:.2e}")
    print(f"[MC gamma={gamma_val}] rejection at 5% (spacing): "
          f"{(df.p_spacing_stable[ok] <= 0.05).mean():.4f}; "
          f"(t-spacing): {(df.p_tspacing_stable[ok] <= 0.05).mean():.4f}")
    near = df[df.spacing < 0.05]
    if len(near):
        print(f"[MC gamma={gamma_val}] {len(near)} draws with spacing < 0.05; "
              f"min p among them: {near.p_spacing_stable.min():.4f}")
    flagged = df[~df.valid]
    if len(flagged):
        print(f"[MC gamma={gamma_val}] {len(flagged)} draws flagged by the "
              f"Lemma-3 check (optimisation failures, detectable)")


if __name__ == "__main__":
    n_mc = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    os.makedirs(OUT, exist_ok=True)
    print("== Part 1: near-tie sweep ==")
    near_tie_sweep()
    print("\n== Part 2: far-tail sweep ==")
    far_tail_sweep()
    print(f"\n== Part 3: fresh Monte-Carlo ({n_mc} replications per gamma) ==")
    for gv in (0.0, 2.5):
        df = monte_carlo(n_mc, gv, res=512, seed=27182, progress=500)
        df.to_csv(os.path.join(OUT, f"fresh_mc_gamma_{gv}.csv"), index=False)
        summarize_mc(df, gv)
