"""Smoke tests of tenspace.simulation on a coarse mesh (fast)."""

import numpy as np

from tenspace.simulation import (SQRT_SNR, create_tensor, grad_of, hess_of,
                                 monte_carlo, one_replication, sphere_mesh,
                                 X_of)


def test_tensor_is_symmetric():
    rng = np.random.default_rng(0)
    W = create_tensor(rng, 2.0, np.array([0.0, 0.0, 1.0]))
    for perm in ((0, 2, 1), (1, 0, 2), (2, 1, 0)):
        assert np.allclose(W, np.transpose(W, perm))


def test_gradient_and_hessian_match_finite_differences():
    rng = np.random.default_rng(1)
    W = create_tensor(rng, 0.0, np.array([0.0, 0.0, 1.0]))
    t = rng.normal(size=3)
    t /= np.linalg.norm(t)
    eps = 1e-6
    for i in range(3):
        e = np.zeros(3)
        e[i] = eps
        fd = (X_of(W, t + e) - X_of(W, t - e)) / (2 * eps)
        assert np.isclose(grad_of(W, t)[i], fd, rtol=1e-5)
        fd_h = (grad_of(W, t + e) - grad_of(W, t - e)) / (2 * eps)
        assert np.allclose(hess_of(W, t)[:, i], fd_h, rtol=1e-4, atol=1e-6)


def test_one_replication_returns_sane_row():
    rng = np.random.default_rng(2)
    pts = sphere_mesh(128)
    row = one_replication(rng, 0.0, pts, steps=256)
    assert row["lambda_1"] >= row["lambda_2"]
    assert 0.0 <= row["p_spacing_stable"] <= 1.0
    assert 0.0 <= row["distance_t0t1"] <= 1.0
    assert row["valid"]
    assert abs(row["p_spacing_naive"] - row["p_spacing_stable"]) < 1e-11


def test_monte_carlo_reproducible_and_calibrated():
    df1 = monte_carlo(20, 0.0, res=128, seed=7, steps=128)
    df2 = monte_carlo(20, 0.0, res=128, seed=7, steps=128)
    assert np.allclose(df1.p_spacing_stable, df2.p_spacing_stable)
    assert df1.p_spacing_stable.between(0, 1).all()
    # under a strong alternative the spike is found and p-values are small
    df3 = monte_carlo(10, 4.0, res=128, seed=7, steps=128)
    assert (df3.p_spacing_stable < 0.05).mean() >= 0.8
    assert (df3.distance_t0t1 < 0.05).all()


def test_snr_constant():
    assert np.isclose(SQRT_SNR, np.sqrt(3 * np.log(3) + 3 * np.log(np.log(3))))
