#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation of the 3-way, 3-dimensional spiked tensor model and computation
of the spacing and t-spacing test statistics.

Model (Section 5.2 of the paper): one observes the symmetric tensor

    Y = lambda_0 * t_0^{otimes 3} + sigma * W,

where W is a standard symmetric Gaussian tensor, and the random field is
X(t) = <Y, t^{otimes 3}> on the sphere S^2. The signal strength is
parameterised by gamma through lambda_0 = gamma * sqrt(3 log 3 + 3 log log 3)
(gamma = 1 is the phase transition of spiked tensor PCA).

The pipeline of one replication:
  1. draw the tensor;
  2. find (lambda_1, t_1) by a mesh warm start and a Riemannian gradient
     descent on the sphere;
  3. build the Riemannian-Hessian matrix R at t_1 (det_R, trace_R);
  4. find (lambda_2, t_2), the second maximum, by a gradient descent on the
     regressed field X^{|t_1};
  5. estimate sigma by the whitening estimator on kappa = 7 random points;
  6. compute the spacing and t-spacing p-values, with both the direct and
     the numerically stable log-scale evaluations (tenspace.pivot), and the
     Lemma-3 validity check.
"""

import time

import numpy as np
import pandas as pd

from tenspace.pivot import is_valid_second_max, spacing_pvalue, tspacing_pvalue

# lambda_0 = gamma * SQRT_SNR; gamma = 1 is the phase transition
SQRT_SNR = np.sqrt(3 * np.log(3) + 3 * np.log(np.log(3)))
M_BAR = 7  # kappa = m - d - 1 = 10 - 2 - 1 points for the variance estimator


def create_tensor(rng, lambda_0, t_0):
    """Draw Y = lambda_0 t_0^{otimes 3} + W, W standard symmetric Gaussian."""
    x_0, y_0, z_0 = t_0
    g = rng.normal(size=10)
    W = np.zeros((3, 3, 3))
    np.fill_diagonal(W, g[:3] + lambda_0 * np.array([x_0**3, y_0**3, z_0**3]))
    sub = [
        ((0, 0, 1), g[3], x_0 * x_0 * y_0), ((0, 0, 2), g[4], x_0 * x_0 * z_0),
        ((1, 1, 0), g[5], y_0 * y_0 * x_0), ((1, 1, 2), g[6], y_0 * y_0 * z_0),
        ((2, 2, 0), g[7], z_0 * z_0 * x_0), ((2, 2, 1), g[8], z_0 * z_0 * y_0),
    ]
    for (i, j, k), gv, mv in sub:
        v = gv / np.sqrt(3) + lambda_0 * mv
        for idx in {(i, j, k), (j, k, i), (k, i, j), (i, k, j), (k, j, i), (j, i, k)}:
            W[idx] = v
    v = g[9] / np.sqrt(6) + lambda_0 * x_0 * y_0 * z_0
    W[0, 1, 2] = W[0, 2, 1] = W[1, 0, 2] = W[1, 2, 0] = W[2, 0, 1] = W[2, 1, 0] = v
    return W


def X_of(W, t):
    """Random field X(t) = <W, t^{otimes 3}>."""
    return np.einsum("ijk,i,j,k->", W, t, t, t)


def grad_of(W, t):
    """Euclidean gradient of X at t."""
    return 3.0 * np.einsum("ijk,j,k->i", W, t, t)


def hess_of(W, t):
    """Euclidean Hessian of X at t."""
    return 6.0 * np.einsum("ijk,k->ij", W, t)


def sphere_mesh(res):
    """(res x res x 3) mesh of points on the unit sphere."""
    phi = np.linspace(0, np.pi, res)
    theta = np.linspace(0, 2 * np.pi, res)
    phi, theta = np.meshgrid(phi, theta)
    return np.stack([np.sin(phi) * np.cos(theta),
                     np.sin(phi) * np.sin(theta),
                     np.cos(phi)], axis=-1)


def mesh_values(W, pts):
    """X evaluated on a mesh of points (warm start for the descents)."""
    return np.einsum("ijk,...i,...j,...k->...", W, pts, pts, pts)


def riemannian_gd(W, t, steps=512, step_size=0.02, regressed_at=None):
    """Projected gradient ascent on the sphere.

    With regressed_at=(t_1, lambda_1), the ascent is on the regressed field
    X^{|t_1} instead of X.
    """
    for _ in range(steps):
        if regressed_at is None:
            grad = grad_of(W, t)
        else:
            t1, x1 = regressed_at
            c = np.dot(t1, t)**3
            dc = 3.0 * np.dot(t1, t)**2 * t1
            f, df = X_of(W, t), grad_of(W, t)
            grad = ((df - dc * x1) * (1 - c) + (f - c * x1) * dc) / (1 - c)**2
        grad = grad - np.dot(grad, t) * t
        t = t + step_size * grad
        t = t / np.linalg.norm(t)
    return t


def sigma_estimate(rng, W, t1, x1):
    """Whitening estimator of sigma on M_BAR random points (Section 4)."""
    V = rng.normal(size=(3, M_BAR))
    V /= np.linalg.norm(V, axis=0)
    Pi = np.outer(t1, t1)
    Sigma = np.zeros((M_BAR, M_BAR))
    for i in range(M_BAR):
        u = V[:, i]
        for j in range(M_BAR):
            v = V[:, j]
            Sigma[i, j] = (np.dot(u, v)**3
                           - (np.dot(u, t1)**3) * (np.dot(v, t1)**3)
                           - 3.0 * np.dot((np.dot(u, t1)**2) * ((np.eye(3) - Pi) @ u),
                                          (np.dot(v, t1)**2) * ((np.eye(3) - Pi) @ v)))
    X2V = np.array([X_of(W, V[:, i]) - (np.dot(t1, V[:, i])**3) * x1
                    for i in range(M_BAR)])
    w, U = np.linalg.eigh(Sigma)
    y = U @ np.diag(1 / np.sqrt(w)) @ U.T @ X2V
    return np.sqrt(np.dot(y, y) / M_BAR)


def one_replication(rng, gamma_val, pts, steps=512):
    """One draw of the model; returns a dict of all recorded statistics."""
    t_0 = np.array([0.0, 0.0, 1.0])
    W = create_tensor(rng, gamma_val * SQRT_SNR, t_0)

    val = mesh_values(W, pts)
    t1 = pts[np.unravel_index(np.argmax(val), val.shape)]
    t1 = riemannian_gd(W, t1, steps=steps)
    lambda_1 = X_of(W, t1)

    Pi = np.outer(t1, t1)
    R = (np.eye(3) - Pi) @ hess_of(W, t1) @ (np.eye(3) - Pi)
    R = R - np.dot(grad_of(W, t1), t1) * (np.eye(3) - Pi)
    R = R / 3.0 + lambda_1 * (np.eye(3) - Pi) + Pi
    det_r = np.linalg.det(R)
    trace_r = np.trace(R) - 1.0

    c = np.einsum("...i,i->...", pts, t1)**3
    val2 = (val - c * lambda_1) / (1.0 - c)
    t2 = pts[np.unravel_index(np.argmax(val2), val2.shape)]
    t2 = riemannian_gd(W, t2, steps=steps, regressed_at=(t1, lambda_1))
    c2 = np.dot(t1, t2)**3
    lambda_2 = (X_of(W, t2) - c2 * lambda_1) / (1.0 - c2)

    sig = sigma_estimate(rng, W, t1, lambda_1)
    with np.errstate(all="ignore"):
        row = dict(
            gamma=gamma_val, lambda_1=lambda_1, lambda_2=lambda_2,
            spacing=lambda_1 - lambda_2, det_R=det_r, trace_R=trace_r,
            sigma_estimate=sig,
            distance_t0t1=(1 - np.dot(t1, t_0)) / 2,
            distance_t1t2=(1 - np.dot(t1, t2)) / 2,
            p_spacing_naive=spacing_pvalue(lambda_1, lambda_2, det_r, trace_r,
                                           method="naive"),
            p_spacing_stable=spacing_pvalue(lambda_1, lambda_2, det_r, trace_r,
                                            method="stable"),
            p_tspacing_naive=tspacing_pvalue(lambda_1, lambda_2, det_r, trace_r,
                                             sig, method="naive"),
            p_tspacing_stable=tspacing_pvalue(lambda_1, lambda_2, det_r, trace_r,
                                              sig, method="stable"),
            valid=bool(is_valid_second_max(lambda_1, lambda_2, det_r, trace_r)),
        )
    return row


def monte_carlo(n_rep, gamma_val, res=512, seed=27182, steps=512, progress=None):
    """n_rep independent replications; returns a pandas DataFrame.

    progress: None, or an int k to print a line every k replications.
    """
    rng = np.random.default_rng(seed + int(10 * gamma_val))
    pts = sphere_mesh(res)
    t0 = time.time()
    rows = []
    for k in range(n_rep):
        rows.append(one_replication(rng, gamma_val, pts, steps=steps))
        if progress and (k + 1) % progress == 0:
            print(f"  gamma={gamma_val}: {k+1}/{n_rep} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)
