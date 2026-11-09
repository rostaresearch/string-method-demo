"""Hamiltonian replica exchange between the string's umbrella windows.

This is the second ingredient of the 2011 ribonuclease H study: the string images are
not sampled independently, they periodically swap configurations with their neighbours.
The exchange is a plain Metropolis move in the space of (configuration, window) pairs.

-------------------------------------------------------------------------------
THE MOVE
-------------------------------------------------------------------------------
Every window runs the same unbiased potential V and differs only in its restraint
U_i(x) = sum_j 0.5 k_j (x_j - c_ij)^2.  Proposing to swap the configurations x_i and
x_j held by windows i and j, the V terms cancel and the acceptance is

    Delta = [ U_i(x_j) + U_j(x_i) - U_i(x_i) - U_j(x_j) ] / kT
    accept with probability min(1, exp(-Delta))                                   (1)

That is the whole thing.  No energy of the unbiased potential is needed, which is why
it is cheap: you already know every window's centres and force constants.

-------------------------------------------------------------------------------
THE SWEEP PATTERN
-------------------------------------------------------------------------------
Neighbour pairs are attempted in two alternating passes, so that no window appears in
two pairs of the same pass:

    parity 0 :  (1,2) (3,4) (5,6) ...
    parity 1 :  (2,3) (4,5) (6,7) ...

(1-based above to match how the pairs are usually written; the code is 0-based.)
Alternating the parity between exchange attempts is what lets a configuration migrate
along the whole string rather than rattling inside one pair.

-------------------------------------------------------------------------------
TWO THINGS TO BE CLEAR ABOUT
-------------------------------------------------------------------------------
1. WHAT IS SWAPPED.  We swap the *configurations* and leave the window identities
   fixed, so window k always holds samples generated under bias k.  Record your samples
   per WINDOW, not per replica, and the binless WHAM in reweight.py needs no change at
   all - every sample still carries the bias of the window it was recorded in.  Swapping
   the labels instead is formally equivalent but makes the bookkeeping easy to get
   wrong.

2. THE DYNAMICS IS NO LONGER CONTINUOUS.  An accepted exchange teleports a
   configuration from one window to another.  Free energies are unaffected - detailed
   balance is satisfied and the equilibrium distribution of each window is exactly the
   one it had before - but any estimator that reads *dynamical* information off the
   trajectory is now invalid.  That includes transition counts, so
   diagnostics.transition_flux and any DHAM / MSM / rate estimate must be computed from
   exchange-free runs.  `run_demo_hrex.py` demonstrates both halves of this: the PMF is
   unchanged, the flux statistics are not meaningful.

With overdamped dynamics there are no velocities, so an exchange is purely positional.
With a velocity-Verlet / Langevin integrator at a single temperature the velocities can
simply travel with the configuration (all replicas share kT here, so no rescaling).
"""
from __future__ import annotations

import numpy as np

from sampler import restraint_energy


def pair_indices(n_windows: int, parity: int):
    """Neighbour pairs of the given parity, 0-based.

    parity 0 -> (0,1), (2,3), ...   which is "1-2, 3-4, ..." in 1-based labels
    parity 1 -> (1,2), (3,4), ...   which is "2-3, 4-5, ..."
    """
    lo = np.arange(parity, n_windows - 1, 2)
    return lo, lo + 1


def exchange_pass(x, centres, k, kT, rng, parity, replica=None):
    """One Metropolis pass over the neighbour pairs of `parity`, in place-safe form.

    x        (n_windows, n_dim) current configuration held by each window
    centres  (n_windows, n_dim) restraint centres
    returns  (x_new, accepted_mask, lo, replica_new)
    """
    x = np.array(x, dtype=float, copy=True)
    centres = np.asarray(centres, dtype=float)
    lo, hi = pair_indices(len(centres), parity)
    if len(lo) == 0:
        return x, np.zeros(0, dtype=bool), lo, replica

    k = np.broadcast_to(np.asarray(k, dtype=float), x.shape)

    # equation (1): only the restraint energies enter, V cancels
    u_ii = restraint_energy(x[lo], centres[lo], k[lo])
    u_jj = restraint_energy(x[hi], centres[hi], k[hi])
    u_ij = restraint_energy(x[hi], centres[lo], k[lo])   # window i evaluating x_j
    u_ji = restraint_energy(x[lo], centres[hi], k[hi])   # window j evaluating x_i
    delta = (u_ij + u_ji - u_ii - u_jj) / kT

    accept = rng.random(len(lo)) < np.exp(np.minimum(0.0, -delta))

    a_lo, a_hi = lo[accept], hi[accept]
    x[[a_lo, a_hi]] = x[[a_hi, a_lo]]
    if replica is not None:
        replica = np.array(replica, copy=True)
        replica[[a_lo, a_hi]] = replica[[a_hi, a_lo]]
    return x, accept, lo, replica


class ExchangeStats:
    """Per-pair attempt/accept counters plus the replica permutation history."""

    def __init__(self, n_windows: int):
        self.n_windows = n_windows
        self.attempts = np.zeros(n_windows - 1)
        self.accepts = np.zeros(n_windows - 1)
        self.replica_history = []
        self.swapped_per_pass = []     # windows that actually swapped, per pass

    def record(self, lo, accept):
        np.add.at(self.attempts, lo, 1.0)
        np.add.at(self.accepts, lo, accept.astype(float))
        a_lo = np.asarray(lo)[accept]
        self.swapped_per_pass.append(np.concatenate([a_lo, a_lo + 1])
                                     if len(a_lo) else np.zeros(0, dtype=int))

    def snapshot(self, replica):
        if replica is not None:
            self.replica_history.append(np.array(replica, copy=True))

    @property
    def acceptance(self):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.attempts > 0, self.accepts / self.attempts, np.nan)

    def summary(self):
        a = self.acceptance
        good = np.isfinite(a)
        out = {"acceptance_min": float(np.min(a[good])) if good.any() else float("nan"),
               "acceptance_mean": float(np.mean(a[good])) if good.any() else float("nan"),
               "acceptance_max": float(np.max(a[good])) if good.any() else float("nan"),
               "total_attempts": float(self.attempts.sum())}
        if self.replica_history:
            H = np.array(self.replica_history)              # (n_snapshots, n_windows)
            # how far each replica roamed, in window units
            pos = np.argsort(H, axis=1)                     # window holding each replica
            out["replica_span_mean"] = float(np.mean(pos.max(axis=0) - pos.min(axis=0)))
            out["replica_span_max"] = float(np.max(pos.max(axis=0) - pos.min(axis=0)))
            out["n_full_traversals"] = int(np.sum(
                (pos.min(axis=0) == 0) & (pos.max(axis=0) == self.n_windows - 1)))
        return out


def contaminated_transition_fraction(stats, n_windows, n_frames, exchange_every,
                                     stride):
    """Fraction of consecutive-frame transitions that are teleports, not dynamics.

    An accepted exchange at the end of segment p means the FIRST saved frame of segment
    p+1 did not evolve continuously from the last frame of segment p - it belongs to a
    different replica.  Any transition counted across that boundary is an artefact.

    This is the honest way to state what HREX costs a kinetic estimator: not "the flux
    numbers look different" (they may not, if the windows overlap well and a swap moves
    a configuration about as far as ordinary diffusion would), but "this percentage of
    your recorded transitions never happened".

    returns (per_window_fraction, overall_fraction)
    """
    frames_per_segment = exchange_every // stride
    bad = np.zeros(n_windows)
    for p, swapped in enumerate(stats.swapped_per_pass):
        boundary_frame = (p + 1) * frames_per_segment
        if 0 < boundary_frame < n_frames and len(swapped):
            np.add.at(bad, np.asarray(swapped), 1.0)
    total = max(n_frames - 1, 1)
    return bad / total, float(bad.sum() / (n_windows * total))


def predicted_acceptance(centres, k, kT):
    """Rough guide: acceptance falls off with the squared centre separation.

    For one CV with spacing D and sigma = sqrt(kT/k), Delta has mean k D^2 / kT =
    (D/sigma)^2, so acceptance collapses once neighbours are more than ~1-2 sigma apart.
    Use this the same way as diagnostics.predicted_gaussian_overlap: as a design check
    BEFORE spending any sampling.
    """
    centres = np.asarray(centres, dtype=float)
    k = np.broadcast_to(np.asarray(k, dtype=float), centres.shape[1:])
    sigma = float(np.mean(np.sqrt(kT / k)))
    D = np.linalg.norm(np.diff(centres, axis=0), axis=1)
    return np.exp(-0.5 * (D / sigma) ** 2), D, sigma


# --------------------------------------------------------------- propagation
def propagate_hrex(system, centres, k, n_steps, dt, D, rng, x0=None, stride=1,
                   exchange_every=200, stats=None, replica=None, return_traj=True):
    """MD in segments of `exchange_every` steps, with one exchange pass between them.

    The integrator is sampler.propagate, called unchanged, so the only difference from
    an exchange-free run is what happens at the segment boundaries.  Parity alternates
    every pass: (0,1)(2,3)... then (1,2)(3,4)... as described in the module docstring.
    """
    from sampler import propagate            # local import: sampler must not import us

    centres = np.asarray(centres, dtype=float)
    if exchange_every % stride:
        raise ValueError("exchange_every (%d) must be a multiple of stride (%d) so that "
                         "frame counts stay even across segments" % (exchange_every, stride))

    # Segment lengths: full segments of exchange_every, plus any remainder as a final
    # shorter segment (with no exchange after it).  n_steps need not divide evenly.
    n_full, rest = divmod(n_steps, exchange_every)
    segments = [exchange_every] * n_full + ([rest] if rest else [])

    x = centres.copy() if x0 is None else np.array(x0, dtype=float, copy=True)
    if replica is None:
        replica = np.arange(len(centres))
    parts, parity = [], 0
    for seg_i, n_seg in enumerate(segments):
        t, x = propagate(system, centres, k, n_seg, dt, D, rng, x0=x,
                         stride=stride, return_traj=return_traj)
        if return_traj and t is not None and t.shape[1]:
            parts.append(t)
        if seg_i < len(segments) - 1 or rest == 0:
            x, accept, lo, replica = exchange_pass(x, centres, k, system.kT, rng,
                                                   parity, replica)
            if stats is not None:
                stats.record(lo, accept)
                stats.snapshot(replica)
            parity ^= 1
    traj = np.concatenate(parts, axis=1) if (return_traj and parts) else None
    return traj, x, replica


def sample_window_set_hrex(system, centres, k, rng, equil_steps, sample_steps, dt, D,
                           stride=1, x0=None, exchange_every=200, stats=None,
                           replica=None):
    """Equilibrate (discarded) then sample, with replica exchange throughout.

    Mirrors sampler.sample_window_set so the two can be swapped behind one flag.
    Returns (traj, x_final, replica).
    """
    _, x, replica = propagate_hrex(system, centres, k, equil_steps, dt, D, rng, x0=x0,
                                   stride=stride, exchange_every=exchange_every,
                                   stats=None, replica=replica, return_traj=False)
    traj, x, replica = propagate_hrex(system, centres, k, sample_steps, dt, D, rng, x0=x,
                                      stride=stride, exchange_every=exchange_every,
                                      stats=stats, replica=replica, return_traj=True)
    return traj, x, replica
