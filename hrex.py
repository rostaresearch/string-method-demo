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

2. THE TRAJECTORY IS PIECEWISE CONTINUOUS.  An accepted exchange teleports a
   configuration from one window to another, so a frame pair that straddles a swap did
   not evolve from one to the other.  This does NOT stop you using HREX data for
   kinetics: DHAM and MSM estimators work perfectly well on it, provided transition
   pairs are taken only WITHIN a continuous segment between exchanges.  Use
   `transition_counts_continuous` below instead of a naive count, and you lose only the
   handful of pairs that cross a swap (about 6% in the demo - see
   `contaminated_transition_fraction`).  Free energies need no such care at all:
   detailed balance holds and every window's equilibrium distribution is exactly the one
   it had before.

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

    This is how many transition pairs you DISCARD when feeding HREX data to DHAM or an
    MSM - not a reason to discard the run.  Use transition_counts_continuous to do the
    counting properly; this function just tells you the price, which is small.

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


def segment_of_frame(n_frames, exchange_every, stride):
    """Which continuous segment each saved frame belongs to.

    Frames within one segment evolved continuously; a change of segment index means an
    exchange pass happened in between, so a transition across that boundary is not
    dynamics.
    """
    frames_per_segment = max(exchange_every // stride, 1)
    return np.arange(n_frames) // frames_per_segment


def transition_counts_continuous(s_traj, edges, exchange_every, stride, lag=1,
                                 stats=None):
    """Transition counts from HREX data, skipping only genuinely discontinuous pairs.

    This is the correct way to feed replica-exchange trajectories to DHAM or an MSM.
    The exchange is a valid Monte Carlo move on the extended ensemble and the dynamics
    inside each segment is untouched, so you simply refuse to count pairs that straddle
    a swap.

    stats
        An ExchangeStats from the run.  With it, a pair is dropped only when THAT window
        actually swapped at the intervening boundary - which is the exact criterion and
        costs only a few per cent of pairs.  Without it, every pair crossing a segment
        boundary is dropped, which is safe but throws away roughly stride/exchange_every
        of the data whether or not a swap was accepted.

    s_traj : (n_windows, n_frames) a 1-D coordinate, time ordered
    returns (flux, n_used, n_dropped), flux per interior boundary as in
    diagnostics.transition_flux
    """
    s_traj = np.atleast_2d(np.asarray(s_traj, dtype=float))
    n_win, n_frames = s_traj.shape
    n_bins = len(edges) - 1
    fps = max(exchange_every // stride, 1)

    # discontinuity[w, t] = True if frame t did not evolve from frame t-1 for window w
    disc = np.zeros((n_win, n_frames), dtype=bool)
    n_boundaries = (n_frames - 1) // fps
    for p in range(n_boundaries):
        t = (p + 1) * fps
        if t >= n_frames:
            break
        if stats is not None and p < len(stats.swapped_per_pass):
            w = np.asarray(stats.swapped_per_pass[p], dtype=int)
            if len(w):
                disc[w[w < n_win], t] = True
        elif stats is None:
            disc[:, t] = True

    # a pair (t, t+lag) is invalid if any discontinuity falls in (t, t+lag]
    cum = np.cumsum(disc, axis=1)
    bad = (cum[:, lag:] - cum[:, :-lag]) > 0

    A = np.zeros((n_bins, n_bins))
    n_used = n_dropped = 0
    for w in range(n_win):
        b = np.clip(np.searchsorted(edges, s_traj[w], side="right") - 1, 0, n_bins - 1)
        keep = ~bad[w]
        np.add.at(A, (b[:-lag][keep], b[lag:][keep]), 1.0)
        n_used += int(keep.sum())
        n_dropped += int((~keep).sum())
    np.fill_diagonal(A, 0.0)
    csum = np.cumsum(np.cumsum(A, axis=0), axis=1)
    flux = np.empty(n_bins - 1)
    for bnd in range(1, n_bins):
        flux[bnd - 1] = ((csum[bnd - 1, -1] - csum[bnd - 1, bnd - 1])
                         + (csum[-1, bnd - 1] - csum[bnd - 1, bnd - 1]))
    return flux, n_used, n_dropped


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
