#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figures on the behaviour and the numerical stability of the spacing test
when the two largest maxima are close.

Panel (a) pvalue_vs_spacing-min.png
    p-value of the spacing test against the observed spacing
    lambda_1 - lambda_2, on the original Monte-Carlo experiments
    (subsample of 3000 draws per gamma). The p-value goes to 1 smoothly as
    the spacing goes to 0; no rejection at the 5% level occurs below a
    spacing of 0.6, and 97% of the rejections have a spacing larger than 1.5.

Panel (b) near_tie_pvalues-min.png
    p-value as the spacing eps = lambda_1 - lambda_2 goes to 0 at fixed
    (lambda_2, Omega). Lines: log-scale evaluation; circles: direct
    evaluation. The two agree to machine precision: the near-tie case is
    the numerically easy case.

Panel (c) far_tail_pvalues-min.png
    p-value as lambda_1 grows with the spacing fixed at 1. The direct
    double-precision evaluation (circles) returns NaN beyond
    lambda_1 ~ 39 (shaded area); the log-scale evaluation (line) follows
    the exact value for all lambda_1.

Requires: reanalysis_spacing_power.py and stability_experiment.py outputs.
Run:  python3 experiments/make_figures.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")

sb.set_theme(context="paper", style="darkgrid",
             rc={"figure.dpi": 300, "savefig.dpi": 300})

GAMMAS = [0, 1, 2, 3, 4, 5]


def panel_a():
    d = pd.read_csv(os.path.join(OUT, "pooled_spacing_sample.csv.gz"))
    d = d[d.alpha.isin(GAMMAS)]
    d = d.groupby("alpha").head(3000)
    d = d.rename(columns={"alpha": r"$\gamma$"})

    fig, ax = plt.subplots(figsize=(4.6, 4))
    sb.scatterplot(ax=ax, data=d, x="spacing", y="spacing_pvalue",
                   hue=r"$\gamma$", palette="flare", s=6, alpha=0.35,
                   edgecolor=None, rasterized=True)
    ax.axhline(0.05, color="k", linestyle="--", linewidth=1)
    ax.text(9.7, 0.075, r"level $5\%$", ha="right", fontsize=8)
    ax.set(xlabel=r"observed spacing $\lambda_1-\lambda_2$",
           ylabel=r"$p$-value of the spacing test",
           xlim=(0, 10), ylim=(-0.02, 1.02))
    leg = ax.legend(title=r"$\gamma$", loc="upper right", fontsize=7,
                    title_fontsize=8, markerscale=1.6, framealpha=0.9)
    for h in leg.legend_handles:
        h.set_alpha(1)
    sb.despine()
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "pvalue_vs_spacing-min.png"))
    plt.close(fig)


def panel_b():
    d = pd.read_csv(os.path.join(OUT, "near_tie_sweep.csv"))
    colors = sb.color_palette("flare", 3)

    fig, ax = plt.subplots(figsize=(4.6, 4))
    for c, (l2, sub) in zip(colors, d.groupby("lambda_2")):
        sub = sub.sort_values("eps")
        ax.plot(sub.eps, sub.p_stable, color=c, lw=1.8,
                label=rf"$\lambda_2={l2}$")
        ax.plot(sub.eps[::3], sub.p_naive[::3], "o", color=c, ms=4.5,
                mfc="none", mew=1.1)
    ax.set_xscale("log")
    ax.set(xlabel=r"spacing $\varepsilon=\lambda_1-\lambda_2$",
           ylabel=r"$p$-value of the spacing test",
           ylim=(-0.02, 1.02))
    ax.invert_xaxis()
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9,
              title="line: log-scale\ncircle: direct", title_fontsize=8)
    sb.despine()
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "near_tie_pvalues-min.png"))
    plt.close(fig)


def panel_c():
    d = pd.read_csv(os.path.join(OUT, "far_tail_sweep.csv")).sort_values("lambda_1")
    first_nan = d[d.p_naive.isna()].lambda_1.min()
    color = sb.color_palette("flare", 3)[1]

    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.plot(d.lambda_1, d.p_stable, color=color, lw=1.8, label="log-scale")
    ok = d.p_naive > 0
    ax.plot(d.lambda_1[ok][::4], d.p_naive[ok][::4], "o", color=color,
            ms=4.5, mfc="none", mew=1.1, label="direct")
    ax.axvspan(first_nan, 60, color="0.75", alpha=0.5, zorder=0)
    ax.text(first_nan + 1, 1e-12, "direct evaluation\nreturns NaN",
            fontsize=8, va="center")
    ax.set_yscale("log")
    ax.set(xlabel=r"$\lambda_1$ (spacing fixed at $1$)",
           ylabel=r"$p$-value of the spacing test",
           xlim=(2, 60))
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    sb.despine()
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "far_tail_pvalues-min.png"))
    plt.close(fig)


if __name__ == "__main__":
    panel_a()
    panel_b()
    panel_c()
    print("saved pvalue_vs_spacing-min.png, near_tie_pvalues-min.png, "
          "far_tail_pvalues-min.png")
