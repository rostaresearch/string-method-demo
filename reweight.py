"""Binless WHAM (= MBAR) and, as an independent cross-check, mean-force integration.

-------------------------------------------------------------------------------
WHY BINLESS
-------------------------------------------------------------------------------
Ordinary WHAM histograms the samples first, then solves for one free-energy offset
f_k per window.  You must choose the bins *before* you solve, and you must redo the
whole solve if you later want the free energy along a different coordinate.

Binless WHAM never histograms.  It solves the same self-consistent equations using
the reduced bias evaluated at *every individual sample*:

      f_k  =  -ln  sum_n   exp(-u_k(x_n))  /  sum_j  N_j exp(f_j - u_j(x_n))        (1)

and then assigns every sample an unbiased weight

      w_n  proportional to  1 / sum_j  N_j exp(f_j - u_j(x_n))                      (2)

Once you have w_n you are done with the estimator.  The free energy along ANY
coordinate q is just a weighted histogram (or KDE) of q(x_n) with weights w_n:

      A(q)  =  -kT ln  sum_n  w_n  delta(q(x_n) - q)                                (3)

So you solve once and can then project onto the path coordinate, onto a single CV,
onto a 2-D plane, or onto a coordinate you only thought of afterwards - without
touching the estimator again.  That is the practical reason we use it.

These are the MBAR equations (Shirts & Chodera 2008); "binless WHAM" is the older
name for the same thing.  Nothing here is novel - the value is seeing it in 40 lines.

-------------------------------------------------------------------------------
GAUGE, UNITS, AND WHAT CAN GO WRONG
-------------------------------------------------------------------------------
* Equation (1) is invariant under f -> f + const, so one offset is arbitrary.  We fix
  f[0] = 0.  Never compare f values between two separate solves without re-zeroing.
* u_k must be the REDUCED bias, i.e. U_k/kT, dimensionless.
* Every window must use its own force constants and its own centre.  Pooling windows
  that had different restraints under one assumed restraint is the single most common
  way to get a confidently wrong barrier.
* Binless WHAM needs the windows to overlap in *configuration* space.  If adjacent
  windows share no samples, the relative offset between them is poorly determined and
  no amount of iterating will fix it - see diagnostics.neighbour_overlap.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


# --------------------------------------------------------------- reduced bias
def reduced_bias_matrix(x, centres, k, kT):
    """u[j, n] = U_j(x_n) / kT  for the harmonic restraint of every window j.

    x       : (N, n_dim) all samples, pooled
    centres : (K, n_dim) one restraint centre per window
    k       : (n_dim,) or (K, n_dim) force constants
    returns : (K, N) float64   -- K*N*8 bytes, so subsample if K*N gets large
    """
    x = np.asarray(x, dtype=float)
    centres = np.asarray(centres, dtype=float)
    K, n_dim = centres.shape
    k = np.broadcast_to(np.asarray(k, dtype=float), (K, n_dim))
    u = np.empty((K, len(x)))
    for j in range(K):
        d = x - centres[j]
        u[j] = 0.5 * np.sum(k[j] * d * d, axis=1) / kT
    return u


# ------------------------------------------------------ the estimator, twice
def solve_self_consistent(u, N_k, tol=1e-10, max_iter=5000):
    """Equation (1), iterated directly.  Short and literal; converges linearly."""
    u = np.asarray(u, dtype=float)
    N_k = np.asarray(N_k, dtype=float)
    K = u.shape[0]
    logN = np.log(N_k)
    f = np.zeros(K)
    for it in range(max_iter):
        log_denom = logsumexp(logN[:, None] + f[:, None] - u, axis=0)   # (N,)
        f_new = -logsumexp(-u - log_denom[None, :], axis=1)             # (K,)
        f_new -= f_new[0]
        delta = float(np.max(np.abs(f_new - f)))
        f = f_new
        if delta < tol:
            return f, {"iterations": it + 1, "max_change": delta, "converged": True}
    return f, {"iterations": max_iter, "max_change": delta, "converged": False}


def _mbar_objective(f, u, N_k, logN):
    """Shirts-Chodera convex objective; its stationary point solves equation (1)."""
    log_denom = logsumexp(logN[:, None] + f[:, None] - u, axis=0)
    phi = float(np.sum(log_denom) - np.dot(N_k, f))
    # p[k, n] = N_k exp(f_k - u_kn) / denom_n ; grad_k = sum_n p[k,n] - N_k
    p = np.exp(logN[:, None] + f[:, None] - u - log_denom[None, :])
    grad = p.sum(axis=1) - N_k
    return phi, grad


def solve_lbfgs(u, N_k, tol=1e-12):
    """Same equations, minimised with L-BFGS.  Much faster for many windows."""
    u = np.asarray(u, dtype=float)
    N_k = np.asarray(N_k, dtype=float)
    logN = np.log(N_k)
    res = minimize(_mbar_objective, np.zeros(u.shape[0]), args=(u, N_k, logN),
                   jac=True, method="L-BFGS-B",
                   options={"maxiter": 10000, "ftol": tol, "gtol": 1e-10})
    f = res.x - res.x[0]
    return f, {"iterations": int(res.nit), "converged": bool(res.success),
               "max_gradient": float(np.max(np.abs(res.jac)))}


def binless_wham(x, centres, k, kT, n_per_window=None, method="lbfgs", **kw):
    """Solve for the window free energies and the per-sample unbiased weights.

    Returns dict with:
        f        (K,)  window free energies, f[0] = 0, in units of kT
        weights  (N,)  normalised unbiased weights, equation (2)
        log_w    (N,)  their logs (use these if you need to reweight again)
        info     solver diagnostics
    """
    u = reduced_bias_matrix(x, centres, k, kT)
    K, N = u.shape
    if n_per_window is None:
        if N % K:
            raise ValueError("give n_per_window when windows have unequal sample counts")
        N_k = np.full(K, N // K, dtype=float)
    else:
        N_k = np.broadcast_to(np.asarray(n_per_window, dtype=float), (K,)).astype(float)
    if not np.isclose(N_k.sum(), N):
        raise ValueError("sum(n_per_window) = %g does not match N = %d" % (N_k.sum(), N))

    solver = {"lbfgs": solve_lbfgs, "self-consistent": solve_self_consistent}[method]
    f, info = solver(u, N_k, **kw)

    log_denom = logsumexp(np.log(N_k)[:, None] + f[:, None] - u, axis=0)
    log_w = -log_denom
    log_w -= logsumexp(log_w)
    return {"f": f, "log_w": log_w, "weights": np.exp(log_w), "info": info,
            "N_k": N_k, "u": u}


def weight_diagnostics(weights, top_k: int = 10, min_kish_fraction: float = 5e-3,
                       max_single_weight: float = 0.05):
    """Health check on the per-sample weights.  ALWAYS look at this.

    Binless WHAM gives every sample a weight w_n proportional to
    1 / sum_j N_j exp(f_j - u_j(x_n)).  If a sample lands where EVERY window has
    negligible probability - a rare excursion outside the union of the window supports -
    the denominator collapses and that one sample can carry a large share of the total
    probability.  The free energy then rests on a handful of frames no matter how many
    you collected, and a reweighted PMF will show narrow spikes where those frames fall.

    With restraints on every CV this is easy to hit: the windows cover a thin tube, so
    anything that wanders out of the tube is outside everything.

    Returns the Kish effective sample size (1 / sum w^2), the largest single weight, and
    the share carried by the top_k samples.

    The defaults ask that no single frame carry more than a twentieth of the
    distribution (`max_single_weight`) and that there be at least one effective sample
    per two hundred collected (`min_kish_fraction`).  Treat `healthy` as a coarse screen
    and read the numbers: what matters is `kish_ess` in absolute terms - a few hundred
    effective samples is a reasonable basis for a PMF, a handful is not.
    """
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    n = len(w)
    kish = 1.0 / float(np.sum(w ** 2))
    srt = np.sort(w)
    return {"n_samples": int(n),
            "kish_ess": kish,
            "kish_fraction": kish / n,
            "max_weight": float(srt[-1]),
            "top_k_share": float(srt[-top_k:].sum()),
            "healthy": bool(kish / n > min_kish_fraction
                            and srt[-1] < max_single_weight)}


# ------------------------------------------------------------- projections
def pmf_from_weights(q, weights, kT, bins=60, range_=None):
    """Equation (3): free energy along ANY coordinate, from the sample weights.

    This is the payoff of going binless - q can be anything you can compute per
    sample, and you never re-solve the estimator.
    """
    q = np.asarray(q, dtype=float)
    hist, edges = np.histogram(q, bins=bins, range=range_, weights=weights)
    centres = 0.5 * (edges[:-1] + edges[1:])
    with np.errstate(divide="ignore"):
        A = -kT * np.log(hist)
    finite = np.isfinite(A)
    A = A - np.min(A[finite]) if finite.any() else A
    return centres, A, hist


def pmf_2d_from_weights(q1, q2, weights, kT, bins=40):
    """The same, in two dimensions.  Same weights, no re-solve."""
    H, e1, e2 = np.histogram2d(np.asarray(q1), np.asarray(q2), bins=bins,
                               weights=weights)
    with np.errstate(divide="ignore"):
        A = -kT * np.log(H)
    finite = np.isfinite(A)
    A = A - np.min(A[finite]) if finite.any() else A
    return 0.5 * (e1[:-1] + e1[1:]), 0.5 * (e2[:-1] + e2[1:]), A


# --------------------------------------------------------------- mean force
def mean_force_pmf(samples, centres, k, path=None):
    """Independent estimator: integrate  dA/ds = k (c - <x>) . t  along the string.

    This uses only each window's OWN average, so it needs no overlap between windows
    at all.  That makes it the right sanity check on a WHAM/MBAR or transition-matrix
    result: if they disagree badly, suspect the overlap, not the physics.

    samples : (n_img, n_frames, n_dim)
    centres : (n_img, n_dim)
    returns : (s, A, dAds)
    """
    samples = np.asarray(samples, dtype=float)
    centres = np.asarray(centres, dtype=float)
    k = np.asarray(k, dtype=float)
    path = centres if path is None else np.asarray(path, dtype=float)

    mean_x = samples.mean(axis=1)                       # (n_img, n_dim)
    force = k * (centres - mean_x)                      # dA/dx at each image

    tang = np.gradient(path, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)

    dAds = np.sum(force * tang, axis=1)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])

    A = np.concatenate([[0.0], np.cumsum(0.5 * (dAds[:-1] + dAds[1:]) * seg)])
    return s, A - A.min(), dAds
