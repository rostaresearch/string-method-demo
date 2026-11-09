"""Fast correctness tests.  `pytest test_demo.py` - a few seconds, no plotting.

These are the checks worth keeping when you port this to a real system: the gradient is
consistent, the reparametrisation does what it claims, the two WHAM solvers agree, and
binless WHAM recovers a free energy whose exact value is known.
"""
from __future__ import annotations

import numpy as np

from potential3d import System, straight_string, densify, project_on_path
from labstring import opt_string, opt_string_info, spline_arc_spacing
from reweight import (reduced_bias_matrix, solve_self_consistent, solve_lbfgs,
                      binless_wham, pmf_from_weights, mean_force_pmf)
from diagnostics import neighbour_overlap, transition_flux, predicted_gaussian_overlap


# ----------------------------------------------------------------- potential
def test_analytic_gradient_matches_finite_differences():
    assert System(kT=0.4).check_gradient(n=128) < 1e-6


def test_straight_line_crosses_the_ridge():
    """The starting guess must be bad, or the demo demonstrates nothing."""
    s = System(kT=0.4)
    mid = s.energy(np.zeros(3))
    well = s.energy(s.minimum_a)
    assert mid - well > 4.0


# ---------------------------------------------------------- reparametrisation
def _helix(n=24):
    t = np.linspace(0.0, 1.0, n)
    return np.stack([2 * t - 1, np.sin(np.pi * t), 0.6 * (1 - np.cos(np.pi * t))], axis=1)


def test_reparametrisation_preserves_the_endpoints():
    path = _helix()
    new = opt_string(path[:, None, :])
    assert np.allclose(new[0], path[0], atol=1e-9)
    assert np.allclose(new[-1], path[-1], atol=1e-9)


def test_reparametrisation_equalises_spline_arc_length():
    """Chords look uneven; spline arc length is equal to within the grid quantisation."""
    _, info = opt_string_info(_helix()[:, None, :])
    sp = spline_arc_spacing(info)
    assert sp["within_quantisation"], sp


def test_reparametrisation_is_idempotent_on_an_even_path():
    """Reparametrising an already equal-arc-length string barely moves it."""
    path = _helix()
    once = opt_string(path[:, None, :])
    twice = opt_string(once[:, None, :])
    assert np.max(np.abs(twice - once)) < 0.02


# ------------------------------------------------------------- binless WHAM
def _umbrella_1d(rng, centres, k, kT, n_frames=4000, dt=2e-3, D=1.0):
    """Sample V(x) = 2 (x^2 - 1)^2 in harmonic windows; returns (n_win, n_frames, 1)."""
    def grad(x):
        return 8.0 * x * (x * x - 1.0)

    x = centres.copy()
    out = np.empty((len(centres), n_frames, 1))
    pref, noise = D / kT, np.sqrt(2.0 * D * dt)
    for _ in range(3000):                                   # equilibrate
        x = x - pref * (grad(x) + k * (x - centres)) * dt + noise * rng.standard_normal(x.shape)
    for j in range(n_frames):
        for _ in range(5):
            x = x - pref * (grad(x) + k * (x - centres)) * dt + noise * rng.standard_normal(x.shape)
        out[:, j, 0] = x
    return out


def test_the_two_solvers_agree():
    rng = np.random.default_rng(7)
    kT, k = 0.5, 30.0
    centres = np.linspace(-1.5, 1.5, 14)
    traj = _umbrella_1d(rng, centres, k, kT, n_frames=1500)
    x = traj.reshape(-1, 1)
    u = reduced_bias_matrix(x, centres[:, None], np.array([k]), kT)
    N_k = np.full(len(centres), traj.shape[1], dtype=float)

    f_sc, info_sc = solve_self_consistent(u, N_k, tol=1e-11, max_iter=20000)
    f_lb, info_lb = solve_lbfgs(u, N_k)
    assert info_sc["converged"], info_sc
    assert np.max(np.abs(f_sc - f_lb)) < 1e-4, (f_sc - f_lb)


def test_binless_wham_recovers_a_known_free_energy():
    """V(x) = 2(x^2-1)^2 has an exact barrier of 2.0 at x = 0 relative to the wells."""
    rng = np.random.default_rng(11)
    kT, k = 0.5, 30.0
    centres = np.linspace(-1.5, 1.5, 16)
    traj = _umbrella_1d(rng, centres, k, kT, n_frames=4000)
    x = traj.reshape(-1, 1)

    res = binless_wham(x, centres[:, None], np.array([k]), kT,
                       n_per_window=traj.shape[1], method="lbfgs")
    q, A, _ = pmf_from_weights(x[:, 0], res["weights"], kT, bins=70, range_=(-1.6, 1.6))

    exact = 2.0 * (q ** 2 - 1.0) ** 2
    ok = np.isfinite(A) & (np.abs(q) < 1.45)
    A = A - A[ok].min()
    exact = exact - exact[ok].min()
    rms = float(np.sqrt(np.mean((A[ok] - exact[ok]) ** 2)))
    assert rms < 0.25, "rms error %.3f kT" % rms

    barrier = A[ok][np.argmin(np.abs(q[ok]))]
    assert abs(barrier - 2.0) < 0.35, "barrier %.3f vs exact 2.0" % barrier


def test_mean_force_agrees_with_binless_wham_on_the_same_data():
    """The independent cross-check that should always be run."""
    rng = np.random.default_rng(13)
    kT, k = 0.5, 30.0
    centres = np.linspace(-1.5, 1.5, 16)
    traj = _umbrella_1d(rng, centres, k, kT, n_frames=4000)

    s_mf, A_mf, _ = mean_force_pmf(traj, centres[:, None], np.array([k]))
    res = binless_wham(traj.reshape(-1, 1), centres[:, None], np.array([k]), kT,
                       n_per_window=traj.shape[1])
    q, A_w, _ = pmf_from_weights(traj.reshape(-1), res["weights"], kT, bins=60,
                                 range_=(-1.5, 1.5))
    # Compare INTERIOR windows only.  The outermost image has no neighbour on one
    # side, so its mean-force tangent is one-sided and the WHAM histogram there is at
    # the edge of the sampled range; both are least reliable exactly there.
    A_on_centres = np.interp(centres, q, A_w)
    A_on_centres -= A_on_centres.min()
    A_mf = A_mf - A_mf.min()
    d = np.abs(A_on_centres - A_mf)[1:-1]
    assert np.max(d) < 0.45, "interior max |mean force - WHAM| = %.3f kT" % np.max(d)


# ------------------------------------------------------------- diagnostics
def test_overlap_and_flux_detect_a_deliberate_gap():
    """Pull two windows far apart: overlap must collapse, flux must collapse."""
    rng = np.random.default_rng(3)
    kT, k = 0.5, 30.0
    close = np.array([-0.2, 0.0, 0.2])
    far = np.array([-1.4, 0.0, 1.4])
    t_close = _umbrella_1d(rng, close, k, kT, n_frames=1500)
    t_far = _umbrella_1d(rng, far, k, kT, n_frames=1500)

    ov_c = neighbour_overlap(t_close, close[:, None], np.array([k]), kT)
    ov_f = neighbour_overlap(t_far, far[:, None], np.array([k]), kT)
    assert ov_c.min() > 0.3
    assert ov_f.min() < 0.05

    edges = np.linspace(-1.8, 1.8, 40)
    fl_c = transition_flux(t_close[:, :, 0], edges)
    fl_f = transition_flux(t_far[:, :, 0], edges)
    assert fl_c.min() >= fl_f.min()


def test_predicted_overlap_is_a_usable_design_rule():
    centres = np.linspace(0, 1, 6)[:, None]
    pred, D, sig = predicted_gaussian_overlap(centres, np.array([30.0]), 0.5)
    assert np.all(pred > 0) and np.all(pred <= 1)
    assert np.isclose(sig, np.sqrt(0.5 / 30.0))


# --------------------------------------------------------------- projection
def test_path_projection_is_monotone_on_the_path_itself():
    s = System(kT=0.4)
    path = straight_string(s, 12)
    dense, s_dense = densify(path, n_per_segment=10)
    got = project_on_path(path, dense, s_dense)
    assert np.all(np.diff(got) > 0)
