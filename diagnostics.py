"""Sampling diagnostics: do the windows actually connect?

Two different questions, often confused:

  neighbour_overlap   - do adjacent windows share CONFIGURATIONS?
                        This is what WHAM / binless WHAM needs to tie windows together.

  transition_flux     - do trajectories actually CROSS between them?
                        This is what DHAM / MSM / any transition-matrix estimator needs.

They are not the same.  A single window whose trajectory wanders back and forth across
a region generates flux there even if no *pair* of windows overlaps in that region, and
conversely two windows can overlap statically while almost nothing crosses between them.

A region can show zero pairwise histogram overlap and still be fully connected in the
transition graph, and a boundary can be connected by very few crossings while its
neighbours carry thousands.  A transition-matrix estimator will report a barrier straight
through such a bottleneck without complaint.  So: check both quantities, and quote the
minimum over boundaries, not the average.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _overlap_coefficient(a, b, bins=60):
    """Integral of min(p_a, p_b): 0 = disjoint, 1 = identical distributions."""
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(a, bins=edges, density=True)
    pb, _ = np.histogram(b, bins=edges, density=True)
    w = np.diff(edges)
    return float(np.sum(np.minimum(pa, pb) * w))


def neighbour_overlap(traj, centres, k, kT, bins=60):
    """Configuration overlap between each adjacent pair of windows.

    Both windows' samples are projected onto the unit vector joining their centres -
    the direction along which they are supposed to overlap - and the overlap
    coefficient of the two 1-D distributions is returned.

    traj    : (n_img, n_frames, n_dim)
    returns : (n_img - 1,) in [0, 1]
    """
    traj = np.asarray(traj, dtype=float)
    centres = np.asarray(centres, dtype=float)
    out = np.empty(len(centres) - 1)
    for i in range(len(centres) - 1):
        e = centres[i + 1] - centres[i]
        n = np.linalg.norm(e)
        if n == 0:
            out[i] = 1.0
            continue
        e = e / n
        out[i] = _overlap_coefficient(traj[i] @ e, traj[i + 1] @ e, bins=bins)
    return out


def predicted_gaussian_overlap(centres, k, kT):
    """2 * Phi(-D / 2 sigma): the overlap two harmonic windows SHOULD have.

    Useful as a design target before you run anything - it tells you whether your
    chosen force constant and image spacing can possibly overlap.
    """
    centres = np.asarray(centres, dtype=float)
    k = np.asarray(k, dtype=float)
    sigma = np.sqrt(kT / np.broadcast_to(k, centres.shape[1:]))
    sig = float(np.mean(sigma))
    D = np.linalg.norm(np.diff(centres, axis=0), axis=1)
    return 2.0 * norm.cdf(-D / (2.0 * sig)), D, sig


def transition_flux(s_traj, edges, lag=1):
    """Transitions crossing every bin boundary, summed over all windows.

    s_traj : (n_windows, n_frames) a 1-D coordinate, time ordered
    returns: (n_bins - 1,) counts crossing each interior boundary

    Report the MINIMUM over boundaries, not the mean.  A transition-matrix free energy
    is only as well determined as its weakest crossing.
    """
    s_traj = np.atleast_2d(np.asarray(s_traj, dtype=float))
    n_bins = len(edges) - 1
    A = np.zeros((n_bins, n_bins))
    for row in s_traj:
        b = np.clip(np.searchsorted(edges, row, side="right") - 1, 0, n_bins - 1)
        src, dst = b[:-lag], b[lag:]
        np.add.at(A, (src, dst), 1.0)
    np.fill_diagonal(A, 0.0)
    csum = np.cumsum(np.cumsum(A, axis=0), axis=1)
    flux = np.empty(n_bins - 1)
    for bnd in range(1, n_bins):
        lower_upper = csum[bnd - 1, -1] - csum[bnd - 1, bnd - 1]
        upper_lower = csum[-1, bnd - 1] - csum[bnd - 1, bnd - 1]
        flux[bnd - 1] = lower_upper + upper_lower
    return flux


def summarise(traj, centres, k, kT, s_traj=None, edges=None):
    """One dict with the numbers worth printing after every production run."""
    ov = neighbour_overlap(traj, centres, k, kT)
    pred, D, sig = predicted_gaussian_overlap(centres, k, kT)
    out = {
        "overlap_min": float(ov.min()), "overlap_median": float(np.median(ov)),
        "overlap_predicted_min": float(pred.min()),
        "spacing_median": float(np.median(D)), "restraint_sigma": sig,
        "spacing_over_sigma_max": float(D.max() / sig),
    }
    if s_traj is not None and edges is not None:
        fl = transition_flux(s_traj, edges)
        out.update({"flux_min": float(fl.min()), "flux_median": float(np.median(fl)),
                    "flux_n_below_50": int((fl < 50).sum())})
    return out
