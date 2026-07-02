# tensor-spacing

Python package for the article *Second maximum of a Gaussian random field and exact (t-)spacing test* by Jean-Marc Azaïs, Federico Dalmao and Yohann De Castro ([arXiv:2406.18397](https://arxiv.org/abs/2406.18397)). It illustrates the method of testing for the existence of **low-rank tensors** within a noisy observation. While the procedure is not specifically designed to detect low-rank structures, it is applicable to **all** alternatives; it is particularly powerful on low-rank alternatives.

## Structure

```text
tenspace/         the package
  pivot.py        spacing and t-spacing p-values (direct and log-scale
                  evaluations) and the Lemma-3 check of the optimisation step
  simulation.py   spiked tensor model, Riemannian gradient descents,
                  variance estimator, Monte-Carlo driver
  plots.py        loading and plotting of the Monte-Carlo experiments
notebooks/        executed Jupyter notebooks (see below)
experiments/      experiment scripts and their outputs
data/             Monte-Carlo experiments: about 250,000 replications per
                  value of the signal strength gamma (2.79 million in total)
figures/          figure assets
tests/            pytest suite
```

## Install

```bash
pip install -e .          # or: pip install -r requirements.txt
pytest tests/             # optional check
```

## Quickstart

```python
from tenspace import monte_carlo, spacing_pvalue, is_valid_second_max

# p-value of the spacing test from the observed statistics
# (lambda_1, lambda_2, and the reduced determinant and trace of Omega)
p = spacing_pvalue(3.0, 1.5, det_r=-0.18, trace_r=-1.06)

# check of the gradient descents (Lemma 3: lambda_2 >= lambda_max(Omega))
ok = is_valid_second_max(3.0, 1.5, det_r=-0.18, trace_r=-1.06)

# simulate the 3-way, 3-dimensional spiked tensor model (gamma = 0: null)
df = monte_carlo(n_rep=100, gamma_val=2.5)
```

The p-values are evaluated on the logarithmic scale by default (scaled
complementary error function), which stays accurate for arbitrarily large
maxima; the direct evaluation is available with `method="naive"`.

## Notebooks

1. [**spacing_tensors.ipynb**](notebooks/spacing_tensors.ipynb) — the spacing and t-spacing tests on symmetric tensors, step by step: the random field on the sphere, the gradient descents, the second maximum, the variance estimator, the p-values, and a Monte-Carlo check.
2. [**monte_carlo_results.ipynb**](notebooks/monte_carlo_results.ipynb) — figures over the pooled Monte-Carlo experiments of `data/` (calibration under the null, power, distances, variance estimator, Hessian statistics).
3. [**stability_experiments.ipynb**](notebooks/stability_experiments.ipynb) — behaviour of the test when the two maxima are close: the p-value goes to 1 smoothly as the spacing goes to 0, near ties are the numerically easy case, and the far tail requires the log-scale evaluation.

## Experiments

The scripts of `experiments/` reproduce the study of the test when the two
maxima are close (power as a function of the spacing, near-tie and far-tail
sweeps, fresh Monte-Carlo runs). See [experiments/README.md](experiments/README.md).

## Reference

Azaïs J.-M., Dalmao F., De Castro Y., *Second maximum of a Gaussian random field and exact (t-)spacing test*, [arXiv:2406.18397](https://arxiv.org/abs/2406.18397).
