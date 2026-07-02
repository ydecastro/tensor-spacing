#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numerically stable evaluation of the spacing-test pivot.

The spacing test of the paper

    Azais J.-M., Dalmao F., De Castro Y.,
    "Second maximum of a Gaussian random field and exact (t-)spacing test",
    arXiv:2406.18397

rejects for small values of the ratio

    p = G(lambda_1) / G(lambda_2),

where, in the 3-way 3-dimensional spiked tensor case (m = 10, kappa = 7),

    G(l) = (l - trR) * phi(l) + (detR + 1) * (1 - Phi(l)),

phi and Phi are the standard normal density and CDF, and (trR, detR) are the
reduced trace and the determinant of the Riemannian-Hessian matrix R at the
maximiser t_1 (see Section 5 of the paper).

Direct evaluation of G in double precision fails in the far tail:

  * 1 - Phi(l) rounds to 0 for l >~ 8.3 (loss of the second term),
  * phi(l) underflows for l >~ 38.5, giving 0/0 = NaN,
  * when the two terms have opposite signs, catastrophic cancellation can
    return a tiny negative value for a quantity that is positive.

The stable evaluation works on the logarithmic scale. Write

    G(l) = phi(l) * [ (l - trR) + (detR + 1) * M(l) ],
    M(l) = (1 - Phi(l)) / phi(l) = sqrt(pi/2) * erfcx(l / sqrt(2)),

where erfcx is the scaled complementary error function, accurate for all l.
The bracket is an O(l) quantity computed without underflow, and

    log G(l) = -l^2/2 - log(sqrt(2*pi)) + log(bracket),
    p = exp(log G(lambda_1) - log G(lambda_2)).

Near-tied maxima (lambda_1 close to lambda_2) are the numerically easy case:
the ratio is close to 1 and both evaluations agree to machine precision.

The t-spacing pivot H uses Student-t tails, which are polynomial. Underflow
is not an issue there; the function h below simply uses the accurate
survival function scipy.stats.t.sf instead of 1 - cdf.
"""

import numpy as np
from scipy.special import erfcx, gamma
from scipy.stats import norm, t

SQRT_2PI = np.sqrt(2.0 * np.pi)

# constants of the 3-way, 3-dimensional case: m = 10 points, kappa = m - 3 = 7
M_POINTS = 10
KAPPA = 7
_C_STUDENT = (7.0 * np.sqrt(7.0)) / (8.0 * np.sqrt(9.0)) \
    * (gamma(5) * gamma(7 / 2)) / (gamma(9 / 2) * gamma(4))


# ---------------------------------------------------------------------------
# Gaussian case (known variance): G and the spacing p-value
# ---------------------------------------------------------------------------

def mills_ratio(l):
    """(1 - Phi(l)) / phi(l), accurate for all l via erfcx."""
    return np.sqrt(np.pi / 2.0) * erfcx(l / np.sqrt(2.0))


def g_naive(l, det_r, trace_r):
    """G(l) evaluated directly, as in the first released code.

    Kept for comparison. Fails in the far tail (see module docstring).
    """
    return (l - trace_r) * norm.pdf(l) + (det_r + 1.0) * (1.0 - norm.cdf(l))


def log_g_stable(l, det_r, trace_r):
    """log G(l) evaluated on the log scale, exact for arbitrarily large l.

    Returns -inf when the bracket is not positive; this only happens when
    (l, det_r, trace_r) do not come from an actual maximiser (for instance
    when the gradient descent returned a wrong critical point), never from
    the evaluation itself.
    """
    l = np.asarray(l, dtype=float)
    bracket = (l - trace_r) + (det_r + 1.0) * mills_ratio(l)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(bracket > 0.0,
                       -0.5 * l ** 2 - np.log(SQRT_2PI) + np.log(np.maximum(bracket, 0.0)),
                       -np.inf)
    return out


def spacing_pvalue(lambda_1, lambda_2, det_r, trace_r, sigma=1.0, method="stable"):
    """Spacing-test p-value p = G(lambda_1/sigma) / G(lambda_2/sigma).

    method = "stable" (log scale, default) or "naive" (direct evaluation,
    as in the first released code, for comparison).
    """
    l1 = np.asarray(lambda_1, dtype=float) / sigma
    l2 = np.asarray(lambda_2, dtype=float) / sigma
    if method == "naive":
        return g_naive(l1, det_r, trace_r) / g_naive(l2, det_r, trace_r)
    if method == "stable":
        return np.exp(log_g_stable(l1, det_r, trace_r)
                      - log_g_stable(l2, det_r, trace_r))
    raise ValueError("method must be 'stable' or 'naive'")


def is_valid_second_max(lambda_1, lambda_2, det_r, trace_r, sigma=1.0, tol=0.0):
    """Check of the optimisation step through a consequence of Lemma 3.

    Lemma 3 of the paper (the helix) implies that the true second maximum
    satisfies lambda_2/sigma >= lambda_max(Omega/sigma), i.e.
    lambda_2 >= lambda_max(Omega), because the radial limits of the
    regressed field at t_1 are the Rayleigh quotients of the normalised
    Hessian (trace_r and det_r are the reduced trace and determinant of
    Omega/sigma). Together with lambda_2 <= lambda_1, this gives a
    necessary condition on the output of the gradient descents.

    In practice the computed lambda_2 can fall slightly short of the bound
    when the second maximum is attained near t_1 (on the helix), because
    the gradient descent stops at finite distance from t_1; on the 2.79
    million Monte-Carlo draws of the paper, these harmless shortfalls have
    median 0.009. The shortfalls that push the pivot out of its domain and
    give p outside [0,1] are one order of magnitude larger (all above
    0.014, median 0.76). A tolerance tol > 0 (for instance 0.05) flags only
    the substantial shortfalls.
    """
    l1 = np.asarray(lambda_1, dtype=float) / sigma
    l2 = np.asarray(lambda_2, dtype=float) / sigma
    disc = np.sqrt(np.maximum(trace_r ** 2 - 4.0 * det_r, 0.0))
    omega_max = (trace_r + disc) / 2.0
    return (l2 <= l1) & (l2 >= omega_max - tol)


# ---------------------------------------------------------------------------
# Student case (estimated variance): H and the t-spacing p-value
# ---------------------------------------------------------------------------

def h_naive(l, det_r, trace_r):
    """H(l) as in the first released code (uses 1 - cdf)."""
    rv9, rv7 = t(9), t(7)
    out = det_r * (np.sqrt(7 / 9) * (1.0 - rv9.cdf(l * np.sqrt(9) / np.sqrt(7))))
    out += (-trace_r) * _C_STUDENT * rv7.pdf(l)
    out += _C_STUDENT * (l * rv7.pdf(l) + 1.0 - rv7.cdf(l))
    return out


def h_stable(l, det_r, trace_r):
    """H(l) with accurate Student-t survival functions.

    Student tails are polynomial, so no log scale is needed; using
    t.sf instead of 1 - t.cdf keeps full relative accuracy in the tail.
    """
    l = np.asarray(l, dtype=float)
    out = det_r * np.sqrt(7 / 9) * t.sf(l * 3.0 / np.sqrt(7.0), 9)
    out += _C_STUDENT * ((l - trace_r) * t.pdf(l, 7) + t.sf(l, 7))
    return out


def tspacing_pvalue(lambda_1, lambda_2, det_r, trace_r, sigma_hat, method="stable"):
    """t-spacing p-value p = H(lambda_1/sigma_hat) / H(lambda_2/sigma_hat)."""
    l1 = np.asarray(lambda_1, dtype=float) / sigma_hat
    l2 = np.asarray(lambda_2, dtype=float) / sigma_hat
    h = h_stable if method == "stable" else h_naive
    return h(l1, det_r, trace_r) / h(l2, det_r, trace_r)


if __name__ == "__main__":
    # quick self-check: stable and naive agree to machine precision in the
    # moderate range, stable keeps working in the far tail
    rng = np.random.default_rng(0)
    for _ in range(5):
        det_r, trace_r = rng.normal(0, 1), rng.normal(0, 1.5)
        l2 = rng.uniform(0.5, 3.0)
        l1 = l2 + rng.uniform(0.0, 3.0)
        p_naive = spacing_pvalue(l1, l2, det_r, trace_r, method="naive")
        p_stable = spacing_pvalue(l1, l2, det_r, trace_r, method="stable")
        assert abs(p_naive - p_stable) < 1e-13 * max(p_stable, 1e-300)
    p_far = spacing_pvalue(50.0, 49.0, 0.1, 1.0, method="stable")
    assert 0.0 < p_far < 1.0 and np.isfinite(p_far)
    print("self-check passed; p(50, 49) =", p_far)
