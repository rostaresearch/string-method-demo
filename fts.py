"""The finite-temperature string driver.

One iteration, exactly as on the protein:

    1. set each image's restraint centres to its row of the current string
    2. continue that image from its own end state of the previous iteration
    3. equilibrate (discarded), then sample
    4. when every image is done, reparametrise with the lab opt_string

Step 2 matters: images are NOT restarted from the centres each time.  Carrying the
state forward is what lets the orthogonal degrees of freedom stay equilibrated as the
string moves, and it is cheap.

The per-iteration record is deliberately the same set of numbers we watch in
production: arc length, the largest centre move, and the worst neighbour overlap.
Arc length creeping up every iteration is the classic sign that the string is
lengthening into poorly sampled territory.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from labstring import reparametrize, spline_arc_spacing
from sampler import sample_window_set
from hrex import ExchangeStats, sample_window_set_hrex
from diagnostics import neighbour_overlap


@dataclass
class StringRun:
    """Everything a string run produces."""
    centres_history: list = field(default_factory=list)   # [n_iter+1] of (n_img, n_dim)
    samples: list = field(default_factory=list)           # [n_iter] of (n_img, n_frames, n_dim)
    arc_length: list = field(default_factory=list)
    max_centre_move: list = field(default_factory=list)
    min_overlap: list = field(default_factory=list)
    exchange: list = field(default_factory=list)   # per-iteration ExchangeStats.summary()
    k: np.ndarray = None
    kT: float = None

    @property
    def final_centres(self):
        return self.centres_history[-1]

    def pooled(self, burn_in: int = 0):
        """Flatten iterations >= burn_in into one umbrella data set.

        Returns (x, centres, n_per_window) ready to hand to reweight.binless_wham:
        x is (N_total, n_dim) and centres is (K, n_dim).

        Each *iteration* of each *image* is its OWN umbrella window, because its
        restraint centre differs.  Pooling several iterations under one centre would
        be exactly the "assume one restraint when the data had several" mistake that
        the reweight module warns about.
        """
        xs, cs = [], []
        for it in range(burn_in, len(self.samples)):
            traj = self.samples[it]                 # (n_img, n_frames, n_dim)
            cen = self.centres_history[it]          # the centres that produced it
            for i in range(traj.shape[0]):
                xs.append(traj[i])
                cs.append(cen[i])
        n_per_window = xs[0].shape[0]
        return np.concatenate(xs, axis=0), np.array(cs), n_per_window


def run_string(system, initial_centres, k, n_iterations, rng,
               equil_steps=2000, sample_steps=8000, dt=2e-3, D=1.0, stride=4,
               smooth_kappa=0.0, verbose=True, exchange_every=None):
    """Iterate the finite-temperature string to convergence.

    exchange_every
        None  -> independent windows (part 1 of the demo).
        int   -> Hamiltonian replica exchange every that many MD steps, with the
                 alternating neighbour-pair sweep of hrex.py (part 2).  Free energies
                 are unaffected; transition-flux diagnostics become invalid.
    """
    centres = np.array(initial_centres, dtype=float)
    n_img, n_dim = centres.shape
    k_arr = np.broadcast_to(np.asarray(k, dtype=float), (n_dim,)).copy()

    run = StringRun(k=k_arr, kT=system.kT)
    run.centres_history.append(centres.copy())
    x_state = None

    replica = None
    for it in range(n_iterations):
        if exchange_every is None:
            traj, x_state = sample_window_set(
                system, centres, k_arr, rng,
                equil_steps=equil_steps, sample_steps=sample_steps,
                dt=dt, D=D, stride=stride, x0=x_state)
            ex_summary = None
        else:
            stats = ExchangeStats(n_img)
            traj, x_state, replica = sample_window_set_hrex(
                system, centres, k_arr, rng,
                equil_steps=equil_steps, sample_steps=sample_steps,
                dt=dt, D=D, stride=stride, x0=x_state,
                exchange_every=exchange_every, stats=stats, replica=replica)
            ex_summary = stats.summary()

        new_centres, info = reparametrize(traj, smooth_kappa=smooth_kappa)
        move = float(np.max(np.abs(new_centres - centres)))
        ov = neighbour_overlap(traj, centres, k_arr, system.kT)

        run.samples.append(traj)
        run.exchange.append(ex_summary)
        run.arc_length.append(float(info["arc_length"]))
        run.max_centre_move.append(move)
        run.min_overlap.append(float(np.min(ov)))
        run.centres_history.append(new_centres.copy())

        if verbose:
            extra = ("" if ex_summary is None
                     else "   acc %.2f" % ex_summary["acceptance_mean"])
            print("  iter %3d   arc %7.4f   max move %7.4f   min overlap %.3f%s"
                  % (it + 1, info["arc_length"], move, np.min(ov), extra))

        centres = new_centres

    return run


def converged_path(run: StringRun, last: int = 10):
    """Average the last `last` strings - a cheap way to damp reparametrisation noise."""
    hist = np.array(run.centres_history[-last:])
    return hist.mean(axis=0)
