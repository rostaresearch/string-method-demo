"""Tests for the replica-exchange part.  `pytest test_hrex.py`

The important one is test_exchange_preserves_the_free_energy: HREX is only worth using
if it leaves the equilibrium distribution of every window exactly as it was.
"""
from __future__ import annotations

import numpy as np

from hrex import (pair_indices, exchange_pass, ExchangeStats, propagate_hrex,
                  sample_window_set_hrex, predicted_acceptance,
                  contaminated_transition_fraction)
from sampler import sample_window_set, restraint_energy
from reweight import binless_wham, pmf_from_weights


class DoubleWell1D:
    """V(x) = 2 (x^2 - 1)^2 in one dimension, with the System interface."""
    def __init__(self, kT=0.5):
        self.kT = kT
        self.n_dim = 1

    def energy(self, r):
        x = np.asarray(r)[..., 0]
        return 2.0 * (x * x - 1.0) ** 2

    def gradient(self, r):
        x = np.asarray(r, dtype=float)
        return (8.0 * x * (x * x - 1.0)).reshape(x.shape)


# ------------------------------------------------------------- the sweep
def test_pairs_are_the_alternating_neighbour_sweep():
    """parity 0 -> 1-2, 3-4, ... ; parity 1 -> 2-3, 4-5, ... (1-based labels)."""
    lo0, hi0 = pair_indices(8, 0)
    lo1, hi1 = pair_indices(8, 1)
    assert list(zip(lo0, hi0)) == [(0, 1), (2, 3), (4, 5), (6, 7)]
    assert list(zip(lo1, hi1)) == [(1, 2), (3, 4), (5, 6)]
    # no window may appear twice within one pass
    for lo, hi in ((lo0, hi0), (lo1, hi1)):
        allw = np.concatenate([lo, hi])
        assert len(allw) == len(set(allw.tolist()))


def test_odd_window_count_leaves_one_out_per_pass():
    lo0, _ = pair_indices(7, 0)
    lo1, _ = pair_indices(7, 1)
    assert list(lo0) == [0, 2, 4]
    assert list(lo1) == [1, 3, 5]


# -------------------------------------------------------- the move itself
def test_exchange_is_a_permutation_of_the_configurations():
    """Nothing may be created or destroyed - only swapped."""
    rng = np.random.default_rng(0)
    centres = np.linspace(-1, 1, 6)[:, None]
    x = centres + 0.05 * rng.standard_normal(centres.shape)
    before = np.sort(x.ravel())
    out, accept, lo, rep = exchange_pass(x, centres, np.array([30.0]), 0.5, rng, 0,
                                         replica=np.arange(6))
    assert np.allclose(np.sort(out.ravel()), before)
    assert sorted(rep.tolist()) == list(range(6))


def test_identical_windows_always_accept():
    """If two windows have the same centre, Delta = 0 and every swap is accepted."""
    rng = np.random.default_rng(1)
    centres = np.zeros((4, 1))
    x = rng.standard_normal((4, 1))
    _, accept, _, _ = exchange_pass(x, centres, np.array([30.0]), 0.5, rng, 0)
    assert accept.all()


def test_acceptance_falls_as_windows_separate():
    rng = np.random.default_rng(2)
    k, kT = np.array([30.0]), 0.5
    acc = []
    for spacing in (0.05, 0.2, 0.8):
        centres = (np.arange(6) * spacing)[:, None]
        hits = 0
        for _ in range(400):
            x = centres + np.sqrt(kT / k) * rng.standard_normal(centres.shape)
            _, a, _, _ = exchange_pass(x, centres, k, kT, rng, 0)
            hits += a.sum()
        acc.append(hits)
    assert acc[0] > acc[1] > acc[2], acc


def test_predicted_acceptance_brackets_the_measurement():
    """The exp(-0.5 (D/sigma)^2) guide should be the right order of magnitude."""
    rng = np.random.default_rng(3)
    k, kT = np.array([30.0]), 0.5
    centres = (np.arange(6) * 0.2)[:, None]
    pred, D, sig = predicted_acceptance(centres, k, kT)
    hits = tot = 0
    for _ in range(600):
        x = centres + np.sqrt(kT / k) * rng.standard_normal(centres.shape)
        _, a, lo, _ = exchange_pass(x, centres, k, kT, rng, 0)
        hits += a.sum(); tot += len(lo)
    measured = hits / tot
    assert 0.2 * pred.mean() < measured < 5.0 * pred.mean(), (measured, pred.mean())


# ------------------------------------------- the property that actually matters
def _umbrella(system, centres, k, rng, exchange_every=None, sample_steps=60_000):
    kw = dict(equil_steps=4000, sample_steps=sample_steps, dt=2e-3, D=1.0, stride=20)
    if exchange_every is None:
        traj, _ = sample_window_set(system, centres, k, rng, **kw)
        return traj, None
    stats = ExchangeStats(len(centres))
    traj, _, _ = sample_window_set_hrex(system, centres, k, rng,
                                        exchange_every=exchange_every, stats=stats, **kw)
    return traj, stats


def test_exchange_preserves_the_free_energy():
    """HREX must not change the PMF.  V(x) = 2(x^2-1)^2 has an exact barrier of 2.0."""
    system = DoubleWell1D(kT=0.5)
    k = np.array([30.0])
    centres = np.linspace(-1.5, 1.5, 14)[:, None]

    out = {}
    for tag, ex in (("plain", None), ("hrex", 200)):
        traj, _ = _umbrella(system, centres, k, np.random.default_rng(17), ex)
        x = traj.reshape(-1, 1)
        w = binless_wham(x, centres, k, system.kT,
                         n_per_window=traj.shape[1])["weights"]
        q, A, _ = pmf_from_weights(x[:, 0], w, system.kT, bins=60, range_=(-1.5, 1.5))
        ok = np.isfinite(A) & (np.abs(q) < 1.4)
        out[tag] = (q, A - A[ok].min(), ok)

    q, A_p, ok_p = out["plain"]
    _, A_h, ok_h = out["hrex"]
    ok = ok_p & ok_h
    exact = 2.0 * (q ** 2 - 1.0) ** 2
    exact = exact - exact[ok].min()

    rms_p = float(np.sqrt(np.mean((A_p[ok] - exact[ok]) ** 2)))
    rms_h = float(np.sqrt(np.mean((A_h[ok] - exact[ok]) ** 2)))
    between = float(np.sqrt(np.mean((A_h[ok] - A_p[ok]) ** 2)))

    assert rms_p < 0.3, "plain rms %.3f" % rms_p
    assert rms_h < 0.3, "hrex rms %.3f (HREX changed the free energy)" % rms_h
    assert between < 0.3, "hrex vs plain rms %.3f" % between


def test_exchange_actually_mixes_the_replicas():
    system = DoubleWell1D(kT=0.5)
    centres = np.linspace(-1.5, 1.5, 10)[:, None]
    _, stats = _umbrella(system, centres, np.array([30.0]),
                         np.random.default_rng(23), exchange_every=200,
                         sample_steps=40_000)
    s = stats.summary()
    assert s["total_attempts"] > 0
    assert s["acceptance_mean"] > 0.05, s
    assert s["replica_span_mean"] > 1.0, s


# --------------------------------------------- the cost to kinetic estimators
def test_contaminated_fraction_tracks_the_exchange_schedule():
    """More frequent exchange -> more teleported transitions in the saved trajectory."""
    system = DoubleWell1D(kT=0.5)
    centres = np.linspace(-1.0, 1.0, 8)[:, None]
    fracs = []
    for every in (200, 1000):
        traj, stats = _umbrella(system, centres, np.array([30.0]),
                                np.random.default_rng(29), exchange_every=every,
                                sample_steps=40_000)
        _, overall = contaminated_transition_fraction(
            stats, len(centres), traj.shape[1], every, 20)
        fracs.append(overall)
    assert 0.0 < fracs[1] < fracs[0] < 1.0, fracs
