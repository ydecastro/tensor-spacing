"""Tests of tenspace.pivot: the two evaluations of the spacing pivot,
the Student pivot, and the Lemma-3 validity check."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from tenspace.pivot import (g_naive, h_naive, h_stable, is_valid_second_max,
                            log_g_stable, mills_ratio, spacing_pvalue,
                            tspacing_pvalue)


def test_mills_ratio_matches_definition():
    l = np.array([-2.0, 0.0, 1.0, 5.0, 8.0])
    expected = norm.sf(l) / norm.pdf(l)
    assert np.allclose(mills_ratio(l), expected, rtol=1e-13)


def test_log_g_matches_direct_in_moderate_range():
    rng = np.random.default_rng(0)
    for _ in range(200):
        det_r = rng.normal(0, 1.5)
        trace_r = rng.normal(0, 2.0)
        l = rng.uniform(0.0, 6.0)
        direct = g_naive(l, det_r, trace_r)
        if direct <= 0.0:
            continue  # outside the valid domain of the formula
        assert np.isclose(np.exp(log_g_stable(l, det_r, trace_r)), direct,
                          rtol=1e-11)


def test_g_matches_integral_definition():
    # G(l) = int_l^inf det(u Id - Omega) phi(u) du, with
    # det(u Id - Omega) = u^2 - Tr(Omega) u + det(Omega) for a 2x2 Omega
    det_r, trace_r = -0.182, -1.056
    for l in (0.5, 2.0, 4.0):
        val, _ = quad(lambda u: (u * u - trace_r * u + det_r)
                      * norm.pdf(u), l, np.inf)
        assert np.isclose(np.exp(log_g_stable(l, det_r, trace_r)), val,
                          rtol=1e-9)


def test_spacing_pvalue_methods_agree():
    rng = np.random.default_rng(1)
    for _ in range(100):
        det_r, trace_r = rng.normal(0, 1), rng.normal(0, 1.5)
        l2 = rng.uniform(0.5, 3.0)
        l1 = l2 + rng.uniform(0.0, 3.0)
        p_naive = spacing_pvalue(l1, l2, det_r, trace_r, method="naive")
        p_stable = spacing_pvalue(l1, l2, det_r, trace_r, method="stable")
        if 0.0 <= p_naive <= 1.0:
            assert abs(p_naive - p_stable) < 1e-11


def test_near_tie_goes_to_one():
    det_r, trace_r = -0.182, -1.056
    for eps in (1e-3, 1e-9, 1e-15):
        p = spacing_pvalue(2.5 + eps, 2.5, det_r, trace_r)
        assert 0.99 * (1 - 100 * eps) <= p <= 1.0 + 1e-12


def test_far_tail_stable_where_direct_fails():
    det_r, trace_r = -0.182, -1.056
    with np.errstate(all="ignore"):
        p_naive = spacing_pvalue(50.0, 49.0, det_r, trace_r, method="naive")
    p_stable = spacing_pvalue(50.0, 49.0, det_r, trace_r, method="stable")
    assert np.isnan(p_naive)
    assert 0.0 < p_stable < 1e-20  # tiny but exact


def test_spacing_pvalue_monotone_in_lambda1():
    det_r, trace_r = -0.182, -1.056
    l1 = np.linspace(1.5, 12.0, 200)
    p = spacing_pvalue(l1, 1.5, det_r, trace_r)
    assert np.all(np.diff(p) < 0)


def test_tspacing_methods_agree():
    p_naive = tspacing_pvalue(3.0, 1.5, -0.2, -1.0, 0.9, method="naive")
    p_stable = tspacing_pvalue(3.0, 1.5, -0.2, -1.0, 0.9, method="stable")
    assert np.isclose(p_naive, p_stable, rtol=1e-11)
    assert 0.0 < p_stable < 1.0


def test_h_stable_accurate_in_tail():
    # Student tails are polynomial; h_stable keeps relative accuracy
    val_naive = h_naive(30.0, -0.2, -1.0)
    val_stable = h_stable(30.0, -0.2, -1.0)
    assert val_stable > 0.0
    assert np.isclose(val_naive, val_stable, rtol=1e-6)


def test_certificate_on_known_draws():
    # a valid draw of the fresh Monte-Carlo
    assert is_valid_second_max(3.0, 1.5, -0.182, -1.056)
    # a flagged draw: gradient descent stopped below the second maximum
    assert not is_valid_second_max(4.974, 1.931, 0.090, 3.056)
    # a peak swap: lambda_2 > lambda_1
    assert not is_valid_second_max(1.0, 1.5, -0.182, -1.056)
    # tolerance forgives small shortfalls
    disc = np.sqrt(1.056**2 + 4 * 0.182)
    omega_max = (-1.056 + disc) / 2.0
    assert not is_valid_second_max(omega_max - 0.01, omega_max + 1, -0.182, -1.056)


def test_invalid_method_raises():
    with pytest.raises(ValueError):
        spacing_pvalue(2.0, 1.0, 0.0, 0.0, method="fast")
