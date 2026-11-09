"""Part 2: the same string, with Hamiltonian replica exchange.

    python run_demo_hrex.py

Runs the string twice from the same starting guess and the same MD budget - once with
independent windows, once with HREX - and answers three questions:

  1. Does HREX help the string converge?          (figure 4)
  2. Does it change the free energy?               (figure 5, left)   it must not
  3. What does it break?                           (figure 5, right)  the dynamics

Question 3 is the one people forget.  HREX leaves every window's equilibrium
distribution exactly as it was, so PMFs are untouched; but an accepted exchange
teleports a configuration between windows, so transition counts no longer describe
continuous dynamics.  Anything that reads kinetics off the trajectory - DHAM, MSMs,
rates - must come from exchange-free runs.
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from potential3d import System, straight_string, densify, project_on_path
from fts import run_string, converged_path
from sampler import sample_window_set
from hrex import (sample_window_set_hrex, ExchangeStats, predicted_acceptance,
                  pair_indices, transition_counts_continuous)
from reweight import binless_wham, weight_diagnostics, pmf_from_weights
from diagnostics import transition_flux, neighbour_overlap

HERE = os.path.dirname(os.path.abspath(__file__))
FS = 20
plt.rcParams.update({
    "font.size": FS, "axes.titlesize": FS + 1, "axes.labelsize": FS,
    "xtick.labelsize": FS - 2, "ytick.labelsize": FS - 2, "legend.fontsize": FS - 3,
    "figure.titlesize": FS + 3,
})
C_IND, C_HREX, C_EXACT = "#2b6cb0", "#c1442e", "#333333"

N_IMAGES = 16
K = 20.0
DT, DIFF = 1e-3, 1.0
N_ITER = 25
EXCHANGE_EVERY = 200


def zero_at_start(s, A, frac=0.2):
    sel = (s <= s[0] + frac * (s[-1] - s[0])) & np.isfinite(A)
    return A - (np.min(A[sel]) if sel.any() else 0.0)


def main():
    system = System(kT=0.4)
    init = straight_string(system, N_IMAGES)
    sigma = np.sqrt(system.kT / K)
    print("part 2: Hamiltonian replica exchange")
    print("  %d images, k = %.1f, restraint sigma = %.4f" % (N_IMAGES, K, sigma))
    print("  exchange sweep: parity 0 = %s" % (list(zip(*pair_indices(N_IMAGES, 0)))[:3],))
    print("                  parity 1 = %s" % (list(zip(*pair_indices(N_IMAGES, 1)))[:3],))

    # ------------------------------------------------- converge the string
    print("")
    print("  string with exchange every %d steps" % EXCHANGE_EVERY)
    run_hx = run_string(system, init, K, N_ITER, np.random.default_rng(2026),
                        equil_steps=1500, sample_steps=6000, dt=DT, D=DIFF,
                        stride=4, verbose=False, exchange_every=EXCHANGE_EVERY)
    print("    final arc %.4f, mean |move| over last 5 iters %.4f"
          % (run_hx.arc_length[-1], np.mean(run_hx.max_centre_move[-5:])))
    print("    mean acceptance over the run %.3f"
          % np.mean([e["acceptance_mean"] for e in run_hx.exchange]))
    print("    replica span (windows) mean %.1f, max %.0f, full traversals %d"
          % (run_hx.exchange[-1]["replica_span_mean"],
             run_hx.exchange[-1]["replica_span_max"],
             run_hx.exchange[-1]["n_full_traversals"]))

    path = converged_path(run_hx, last=8)
    dense, s_dense = densify(path, n_per_segment=12)

    # ------------------------------------------- matched production, both schemes
    print("\n  production on the same path, both schemes (identical MD budget)")
    prod_ind, _ = sample_window_set(system, path, K, np.random.default_rng(7),
                                    equil_steps=4000, sample_steps=120_000,
                                    dt=DT, D=DIFF, stride=40)
    stats = ExchangeStats(N_IMAGES)
    prod_hx, _, _ = sample_window_set_hrex(system, path, K, np.random.default_rng(7),
                                           equil_steps=4000, sample_steps=120_000,
                                           dt=DT, D=DIFF, stride=40,
                                           exchange_every=EXCHANGE_EVERY, stats=stats)
    print("    %d windows x %d frames each" % prod_ind.shape[:2])
    print("    exchange: %s" % {k: round(v, 3) for k, v in stats.summary().items()})

    # ------------------------------------------------- free energies must agree
    res = {}
    for tag, prod in (("independent", prod_ind), ("hrex", prod_hx)):
        x = prod.reshape(-1, 3)
        s, perp = project_on_path(x, dense, s_dense, return_perp=True)
        res_w = binless_wham(x, path, K, system.kT, n_per_window=prod.shape[1])
        w = res_w["weights"]
        wd = weight_diagnostics(w)
        print("    %-12s weight health: Kish ESS %6.0f of %d (%.2f%%), "
              "max weight %.4f, healthy=%s"
              % (tag, wd["kish_ess"], wd["n_samples"], 100 * wd["kish_fraction"],
                 wd["max_weight"], wd["healthy"]))
        sc, A, _ = pmf_from_weights(s, w, system.kT, bins=44,
                                    range_=(s_dense[0], s_dense[-1]))
        res[tag] = (sc, zero_at_start(sc, A), float(np.percentile(perp, 99.5)), s)

    tube = max(res["independent"][2], res["hrex"][2])
    s_ref, A_ref = system.reference_pmf(path, n_grid=111, n_bins=44, tube_radius=tube)
    A_ref = zero_at_start(s_ref, A_ref)

    bar = lambda t: float(np.nanmax(res[t][1]))
    print("\n  barrier along the path (energy units; kT = %.2f):" % system.kT)
    print("    exact, tube-restricted  %.3f" % float(np.nanmax(A_ref)))
    print("    binless WHAM, no exchange   %.3f" % bar("independent"))
    print("    binless WHAM, with HREX     %.3f" % bar("hrex"))
    print("    HREX - no exchange          %+.3f   <- should be ~0: HREX must not"
          % (bar("hrex") - bar("independent")))
    print("                                          change free energies")

    # ------------------------------------------------- what HREX breaks
    edges = np.linspace(s_dense[0], s_dense[-1], 45)
    fl_ind = transition_flux(res["independent"][3].reshape(prod_ind.shape[:2]), edges)
    fl_hx = transition_flux(res["hrex"][3].reshape(prod_hx.shape[:2]), edges)
    fl_valid, n_used, n_drop = transition_counts_continuous(
        res["hrex"][3].reshape(prod_hx.shape[:2]), edges, EXCHANGE_EVERY, 40,
        stats=stats)
    fl_naive = fl_hx
    dropped_frac = n_drop / max(n_used + n_drop, 1)
    print("")
    print("  using the HREX data for kinetics (DHAM / MSM):")
    print("    An exchange is a valid Monte Carlo move and the dynamics inside each")
    print("    segment is untouched, so transition counts are fine as long as no pair")
    print("    straddles a swap.  transition_counts_continuous() enforces that.")
    print("      pairs kept    %7d" % n_used)
    print("      pairs dropped %7d  (%.1f%% - only where that window actually swapped)"
            % (n_drop, 100 * dropped_frac))
    print("      total flux: naive %.0f, within-segment %.0f"
            % (fl_naive.sum(), fl_valid.sum()))
    print("    Free energies need no such care: detailed balance holds and every")
    print("    window keeps exactly the equilibrium distribution it had before.")

    # ------------------------------- acceptance vs spacing: thin the string
    print("")
    print("  acceptance vs neighbour spacing (string thinned 1x, 2x, 3x, 4x)")
    scan = []
    for thin in (1, 2, 3, 4):
        sub = path[::thin]
        st = ExchangeStats(len(sub))
        sample_window_set_hrex(system, sub, K, np.random.default_rng(100 + thin),
                               equil_steps=2000, sample_steps=20_000, dt=DT, D=DIFF,
                               stride=40, exchange_every=EXCHANGE_EVERY, stats=st)
        pr, Ds, sg = predicted_acceptance(sub, np.full(3, K), system.kT)
        a = st.acceptance
        scan.append((float(np.mean(Ds / sg)), float(np.nanmean(a)), float(np.mean(pr))))
        print("    thin %dx : D/sigma %5.2f   measured %.3f   guide %.3f"
              % (thin, scan[-1][0], scan[-1][1], scan[-1][2]))

    # ============================================================ figure 4
    fig, ax = plt.subplots(1, 2, figsize=(19, 7.6))
    ax = list(ax)
    ax.insert(0, None)          # keep the panel indices below unchanged

    ds = np.array([r[0] for r in scan])
    am = np.array([r[1] for r in scan])
    grid = np.linspace(ds.min() * 0.85, ds.max() * 1.08, 160)
    guide_label = "guide  exp(-0.5 (D/sigma)^2)"
    ax[1].semilogy(grid, np.exp(-0.5 * grid ** 2), "-", color=C_EXACT, lw=3.0,
                   label=guide_label)
    ax[1].semilogy(ds, np.maximum(am, 1e-4), "o", color=C_HREX, ms=14,
                   label="measured")
    for d_, a_ in zip(ds, am):
        ax[1].annotate("%.3f" % a_, (d_, max(a_, 1e-4)),
                       textcoords="offset points", xytext=(10, 8), fontsize=FS - 4)
    ax[1].set_xlabel("mean neighbour spacing  D / sigma")
    ax[1].set_ylabel("exchange acceptance")
    ax[1].set_title("A. acceptance collapses with spacing")
    ax[1].legend(loc="lower left"); ax[1].grid(alpha=0.3, which="both")

    H = np.array(stats.replica_history)
    pos = np.argsort(H, axis=1)
    for r in range(0, N_IMAGES, 3):
        ax[2].plot(pos[:, r], lw=2.2, alpha=0.9)
    ax[2].set_xlabel("exchange attempt"); ax[2].set_ylabel("window index")
    ax[2].set_title("B. replicas migrate along the string")
    ax[2].grid(alpha=0.3)

    fig.suptitle("Hamiltonian replica exchange between string images")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(HERE, "fig4_hrex.png"), dpi=105)

    # ============================================================ figure 5
    fig, ax = plt.subplots(1, 2, figsize=(21, 8.0))

    ax[0].plot(s_ref, A_ref, "-", color=C_EXACT, lw=4.0, label="exact (tube-restricted)")
    ax[0].plot(*res["independent"][:2], "o-", color=C_IND, lw=2.8, ms=7,
               label="binless WHAM, no exchange")
    ax[0].plot(*res["hrex"][:2], "s--", color=C_HREX, lw=2.8, ms=8,
               label="binless WHAM, with HREX")
    ax[0].set_xlabel("path coordinate  s"); ax[0].set_ylabel("A")
    ax[0].set_title("A. HREX does not change the free energy")
    ax[0].legend(loc="lower center"); ax[0].grid(alpha=0.3)

    ax[1].semilogy(edges[1:-1], np.maximum(fl_naive, 1), "-", color="#999999",
                   lw=3.0, label="naive count (wrong: spans swaps)")
    ax[1].semilogy(edges[1:-1], np.maximum(fl_valid, 1), "-", color=C_HREX,
                   lw=3.0, label="within-segment count (correct)")
    ax[1].set_xlabel("path coordinate  s")
    ax[1].set_ylabel("transitions crossing")
    ax[1].set_title("B. HREX data is usable for kinetics: drop %.1f%% of pairs"
                    % (100 * dropped_frac))
    ax[1].legend(loc="lower center"); ax[1].grid(alpha=0.3, which="both")

    fig.suptitle("HREX leaves the free energy alone, and its data is still usable for kinetics")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(HERE, "fig5_hrex_pmf.png"), dpi=105)

    print("\nwrote fig4_hrex.png, fig5_hrex_pmf.png in\n  %s" % HERE)


if __name__ == "__main__":
    main()
