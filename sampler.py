"""Restrained Brownian (overdamped Langevin) dynamics - the "MD engine" of the demo.

Each string image is one umbrella window, restrained by

    U_i(x) = sum_j  0.5 * k_j * (x_j - c_ij)^2

which is exactly the restraint form we use on the protein (one harmonic term per
collective variable, with the centres taken from a row of the string).

All windows are propagated simultaneously as one (n_images, n_dim) array, so a full
string iteration is a handful of numpy operations.  Replace `step` with a call to your
MD engine and the rest of the package is unchanged.

Integrator:   x <- x - (D/kT) * grad U_total * dt + sqrt(2 D dt) * N(0, 1)
This samples exp(-U_total/kT) exactly in the dt -> 0 limit, and has a well-defined
dynamics, which matters for the transition-flux diagnostic in diagnostics.py.
"""
from __future__ import annotations

import numpy as np


def restraint_energy(x, centres, k):
    """0.5 * sum_j k_j (x_j - c_j)^2, per window."""
    d = np.asarray(x) - np.asarray(centres)
    return 0.5 * np.sum(np.asarray(k) * d * d, axis=-1)


def restraint_gradient(x, centres, k):
    return np.asarray(k) * (np.asarray(x) - np.asarray(centres))


def stability_hint(system, k, D, dt):
    """Crude explicit-Euler stability guide; returns the ratio that should stay < ~0.1."""
    kmax = float(np.max(k))
    return float((D / system.kT) * kmax * dt)


def propagate(system, centres, k, n_steps, dt, D, rng, x0=None, stride=1,
              return_traj=True):
    """Run every window for n_steps, saving every `stride` steps.

    centres : (n_images, n_dim) restraint centres
    k       : scalar or (n_dim,) or (n_images, n_dim) force constants
    x0      : (n_images, n_dim) start coordinates; defaults to the centres
    returns : (traj, x_final) with traj (n_images, n_saved, n_dim), or (None, x_final)
    """
    centres = np.asarray(centres, dtype=float)
    n_img, n_dim = centres.shape
    k = np.broadcast_to(np.asarray(k, dtype=float), (n_img, n_dim))
    x = centres.copy() if x0 is None else np.array(x0, dtype=float, copy=True)

    pref = D / system.kT
    noise = np.sqrt(2.0 * D * dt)
    n_saved = n_steps // stride
    traj = np.empty((n_img, n_saved, n_dim)) if return_traj else None

    j = 0
    for step in range(n_steps):
        g = system.gradient(x) + restraint_gradient(x, centres, k)
        x = x - pref * g * dt + noise * rng.standard_normal(x.shape)
        if return_traj and (step + 1) % stride == 0 and j < n_saved:
            traj[:, j, :] = x
            j += 1
    return traj, x


def sample_window_set(system, centres, k, rng, equil_steps, sample_steps, dt, D,
                      stride=1, x0=None):
    """Equilibrate (discarded) then sample - the protocol used on the protein string.

    Returns (traj, x_final).  traj is (n_images, n_frames, n_dim).
    """
    _, x = propagate(system, centres, k, equil_steps, dt, D, rng, x0=x0,
                     return_traj=False)
    traj, x = propagate(system, centres, k, sample_steps, dt, D, rng, x0=x,
                        stride=stride, return_traj=True)
    return traj, x
