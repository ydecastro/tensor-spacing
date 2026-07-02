# Experiments: behaviour of the test when the two maxima are close

Numerical experiments on two questions about the spacing test: *what happens
to the power when the maximum and the second maximum are very close*, and
*how should the p-value be evaluated in floating-point arithmetic*.

## Scripts

1. **`reanalysis_spacing_power.py`** — pools the Monte-Carlo experiments
   (`../data/*.csv`, about 250,000 replications per value of gamma,
   2.79 million in total) and reports:
   - the empirical level under the null (0.0500 at level 5%);
   - the p-value and the power as a function of the observed spacing
     `lambda_1 - lambda_2` (the p-value goes to 1 smoothly as the spacing
     goes to 0; no rejection at 5% occurs below a spacing of 0.6, and 97%
     of the rejections have a spacing larger than 1.5);
   - a re-evaluation of all p-values with the log-scale pivot
     (`tenspace/pivot.py`): no 5% decision ever changes, and the 0.013% of
     draws with a p-value outside [0,1] all come from the optimisation or
     estimation step (peak swaps, undershot second maximum, degenerate
     variance estimate), never from the pivot itself.

2. **`stability_experiment.py`** — targeted experiments:
   - near-tie sweep: spacing from 1 down to 1e-15 at fixed
     `(lambda_2, Omega)`; the direct and log-scale evaluations agree to
     about 1e-14 and the p-value rises smoothly to 1 (the near-tie case is
     the numerically easy case);
   - far-tail sweep: `lambda_1` from 2 to 60 with spacing fixed at 1; the
     direct double-precision evaluation degrades from `lambda_1 ~ 8` and
     returns NaN beyond `lambda_1 ~ 39`, while the log-scale evaluation
     stays accurate;
   - fresh Monte-Carlo: 2000 replications at gamma = 0 and gamma = 2.5
     with the p-values computed by both methods on each draw
     (`tenspace/simulation.py`).

3. **`make_figures.py`** — produces the three panels of the figure of the
   paper (Section 5.2, "Behaviour when the two maxima are close").

4. **`../notebooks/stability_experiments.ipynb`** — executed notebook that
   walks through the findings: the pivot evaluated two ways, the near-tie
   and far-tail sweeps, the fresh Monte-Carlo runs with the Lemma-3 check
   on the flagged draws, and the p-value against the observed spacing.

## Outputs

Written to `outputs/`: `near_tie_sweep.csv`, `far_tail_sweep.csv`,
`fresh_mc_gamma_*.csv`, `pooled_spacing_sample.csv.gz` and the three PNG
panels.

## Run

```bash
python3 experiments/reanalysis_spacing_power.py
python3 experiments/stability_experiment.py 2000
python3 experiments/make_figures.py
```

Requires numpy, scipy, pandas, matplotlib, seaborn (see `requirements.txt`
at the repository root).
