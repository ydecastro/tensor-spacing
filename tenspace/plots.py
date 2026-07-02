#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loading and plotting helpers for the Monte-Carlo experiments (data/*.csv).

Each CSV records, per replication: alpha (the signal strength gamma),
lambda_0, lambda_1, lambda_2, spacing_pvalue, tspacing_pvalue,
distance_t0t1, distance_t1t2, sigma_estimate, det_R, trace_R.

Typical use:

    from tenspace.plots import load_experiments, pvalue_ecdf
    df = load_experiments("data")
    ax = pvalue_ecdf(df, test="spacing")
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from scipy.stats import chi


def load_experiments(data_dir, clean=True):
    """Pool all CSV files of data_dir into one DataFrame.

    With clean=True (default), the replications with a p-value outside
    [0,1] or NaN are dropped (0.013% of the draws; they come from the
    optimisation or estimation step, see the paper, Section 5.2).
    A "spacing" column (lambda_1 - lambda_2) is added.
    """
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"no CSV file found in {data_dir!r}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if clean:
        ok = (df["spacing_pvalue"].between(0, 1)
              & df["tspacing_pvalue"].between(0, 1))
        df = df[ok].reset_index(drop=True)
    df["spacing"] = df["lambda_1"] - df["lambda_2"]
    return df


def _new_ax(ax, figsize):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def pvalue_violins(df, gammas=None, ax=None):
    """Split violins of the spacing and t-spacing p-values per gamma."""
    ax = _new_ax(ax, (8, 4))
    d = df if gammas is None else df[df["alpha"].isin(gammas)]
    long = d.melt(id_vars=["alpha"],
                  value_vars=["spacing_pvalue", "tspacing_pvalue"],
                  var_name="test", value_name="pvalue")
    sb.violinplot(ax=ax, data=long, x="alpha", y="pvalue", hue="test",
                  palette="summer", inner="quartile", density_norm="count",
                  split=True)
    ax.set(xlabel=r"$\gamma$", ylabel=r"$p$-value", ylim=(0, 1))
    ax.legend(title=None, labels=["Spacing", r"$t$-Spacing"])
    return ax


def pvalue_ecdf(df, test="spacing", gammas=None, ax=None):
    """Empirical CDF of a test's p-value, one curve per gamma."""
    ax = _new_ax(ax, (4, 4))
    d = df if gammas is None else df[df["alpha"].isin(gammas)]
    sb.ecdfplot(ax=ax, data=d.rename(columns={"alpha": r"$\gamma$"}),
                x=f"{test}_pvalue", hue=r"$\gamma$", palette="flare")
    ax.set(xlabel=rf"$p$-value of the {test} test", ylabel="proportion",
           xlim=(0, 1), ylim=(0, 1))
    return ax


def pvalue_hist(df, test="tspacing", gammas=None, ax=None):
    """Histogram (density) of a test's p-value, one colour per gamma."""
    ax = _new_ax(ax, (4, 4))
    d = df if gammas is None else df[df["alpha"].isin(gammas)]
    sb.histplot(ax=ax, data=d.rename(columns={"alpha": r"$\gamma$"}),
                x=f"{test}_pvalue", hue=r"$\gamma$", stat="density",
                kde=True, palette="flare", alpha=0.25, common_norm=False)
    ax.set(xlabel=rf"$p$-value of the {test} test", ylabel="density",
           xlim=(0, 1), ylim=(0, 5))
    return ax


def distance_hist(df, which="distance_t0t1", gammas=None, ax=None):
    """Histogram of the normalised distance d(t_0,t_1) or d(t_1,t_2)."""
    ax = _new_ax(ax, (4, 4))
    d = df if gammas is None else df[df["alpha"].isin(gammas)]
    sb.histplot(ax=ax, data=d.rename(columns={"alpha": r"$\gamma$"}),
                x=which, hue=r"$\gamma$", stat="density", kde=True,
                palette="flare", alpha=0.25, common_norm=False)
    label = r"$d(t_0,t_1)$" if which == "distance_t0t1" else r"$d(t_1,t_2)$"
    ax.set(xlabel=label, ylabel="density", xlim=(0, 1))
    return ax


def sigma_hist(df, gammas=None, m_bar=7, ax=None):
    """Histogram of sigma-hat with the chi(m_bar)/sqrt(m_bar) reference."""
    ax = _new_ax(ax, (4, 4))
    d = df if gammas is None else df[df["alpha"].isin(gammas)]
    sb.histplot(ax=ax, data=d.rename(columns={"alpha": r"$\gamma$"}),
                x="sigma_estimate", hue=r"$\gamma$", stat="density", kde=True,
                palette="flare", alpha=0.25, common_norm=False)
    grid = np.linspace(0, 2.5, 3000)
    ax.plot(grid, np.sqrt(m_bar) * chi.pdf(np.sqrt(m_bar) * grid, df=m_bar),
            "k--", linewidth=2)
    ax.set(xlabel=r"$\hat\sigma$", ylabel="density", xlim=(0, 2.5))
    return ax


def lambda_hist(df, which="lambda_1", gammas=None, ax=None):
    """Histogram of lambda_1 or lambda_2, one colour per gamma."""
    ax = _new_ax(ax, (4, 4))
    d = df if gammas is None else df[df["alpha"].isin(gammas)]
    sb.histplot(ax=ax, data=d.rename(columns={"alpha": r"$\gamma$"}),
                x=which, hue=r"$\gamma$", stat="density", kde=True,
                palette="flare", alpha=0.25, common_norm=False)
    ax.set(xlabel=rf"$\lambda_{which[-1]}$", ylabel="density")
    return ax


def pvalue_vs_spacing(df, gammas=(0, 1, 2, 3, 4, 5), n_per_gamma=3000,
                      test="spacing", ax=None):
    """Scatter of the p-value against the observed spacing lambda_1-lambda_2.

    The p-value goes to 1 as the spacing goes to 0 (see the paper,
    Section 5.2, "Behaviour when the two maxima are close").
    """
    ax = _new_ax(ax, (4.6, 4))
    d = df[df["alpha"].isin(gammas)]
    d = (d.sample(frac=1.0, random_state=0).groupby("alpha").head(n_per_gamma)
         .rename(columns={"alpha": r"$\gamma$"}))
    sb.scatterplot(ax=ax, data=d, x="spacing", y=f"{test}_pvalue",
                   hue=r"$\gamma$", palette="flare", s=6, alpha=0.35,
                   edgecolor=None, rasterized=True)
    ax.axhline(0.05, color="k", linestyle="--", linewidth=1)
    ax.set(xlabel=r"observed spacing $\lambda_1-\lambda_2$",
           ylabel=rf"$p$-value of the {test} test",
           xlim=(0, 10), ylim=(-0.02, 1.02))
    return ax
