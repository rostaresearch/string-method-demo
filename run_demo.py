"""End-to-end demonstration: string method + binless WHAM on a 3-D analytical system.

    python run_demo.py

Produces three figures and prints the numbers worth checking.  Runs in a couple of
minutes on one core; everything is numpy/scipy.

The structure mirrors a real campaign:

    phase 1   converge the string          many cheap iterations
    phase 2   production umbrella sampling  on the FINAL string only
    phase 3   estimate the free energy      binless WHAM, cross-checked by mean force
    phase 4   check the sampling            overlap and transition flux

Phase 4 is not optional.  A free energy from windows that do not overlap is a number,
not a measurement.
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from potential3d import System, straight_string, densify, project_on_path
from fts import run_string, converged_path
from sampler import sample_window_set, stability_hint
from reweight import (binless_wham, weight_diagnostics, pmf_from_weights, pmf_2d_from_weights,
                      mean_force_pmf, solve_self_consistent, solve_lbfgs,
                      reduced_bias_matrix)
from diagnostics import summarise, neighbour_overlap, transition_flux

HERE = os.path.dirname(os.path.abspath(__file__))
FS = 20
plt.rcParams.update({
    "font.size": FS, "axes.titlesize": FS + 1, "axes.labelsize": FS,
    "xtick.labelsize": FS - 2, "ytick.labelsize": FS - 2, "legend.fontsize": FS - 3,
    "figure.titlesize": FS + 3,
})
C_EXACT, C_WHAM, C_MF, C_INIT = "#333333", "#c1442e", "#1b7f5a", "#2b6cb0"


def zero_at_start(s, A, frac=0.2):
    """Common zero: the minimum over the first `frac` of the path (well A)."""
    sel = (s <= s[0] + frac * (s[-1] - s[0])) & np.isfinite(A)
    return A - (np.min(A[sel]) if sel.any() else 0.0)


def main():
    rng = np.random.default_rng(20261108)

    # ---------------------------------------------------------------- setup
    system = System(kT=0.4)
    k = 20.0
    dt, D = 1e-3, 1.0
    n_images = 20

    print("3-D analytical system")
    print("  kT                      %.3f" % system.kT)
    print("  analytic gradient check %.2e (max abs vs central differences)"
          % system.check_gradient())
    print("  restraint sigma         %.4f   (sqrt(kT/k), k = %.1f)"
          % (np.sqrt(system.kT / k), k))
    print("  integrator stability    %.3f   (keep well below 1)"
          % stability_hint(system, k, D, dt))

    init = straight_string(system, n_images)
    print("  straight-line guess passes through V(0,0,0) = %.2f, wells at %.2f"
          % (system.energy(np.zeros(3)), system.energy(system.minimum_a)))

    # ------------------------------------------------- phase 1: converge the string
    print("\nphase 1: converging the string (%d images)" % n_images)
    run = run_string(system, init, k, n_iterations=40, rng=rng,
                     equil_steps=2000, sample_steps=8000, dt=dt, D=D, stride=4)
    path = converged_path(run, last=10)

    # ------------------------------------------------- phase 2: production sampling
    print("\nphase 2: production umbrella sampling on the converged string")
    prod, _ = sample_window_set(system, path, k, rng,
                                equil_steps=5000, sample_steps=200_000,
                                dt=dt, D=D, stride=50)
    print("  %d windows x %d frames = %d samples"
          % (prod.shape[0], prod.shape[1], prod.size // 3))

    # a common path coordinate for everything that follows
    dense, s_dense = densify(path, n_per_segment=12)
    s_img = project_on_path(path, dense, s_dense)
    x_all = prod.reshape(-1, 3)
    s_all, perp_all = project_on_path(x_all, dense, s_dense, return_perp=True)
    tube = float(np.percentile(perp_all, 99.5))
    print("  restraints sample a tube of perpendicular radius %.3f "
          "(99.5th pct; restraint sigma %.3f)" % (tube, np.sqrt(system.kT / k)))

    # ------------------------------------------------- phase 3: binless WHAM
    print("\nphase 3: binless WHAM")
    wham = binless_wham(x_all, path, k, system.kT,
                        n_per_window=prod.shape[1], method="lbfgs")
    print("  solver: %s" % wham["info"])
    wd = weight_diagnostics(wham["weights"])
    print("  weight health: Kish ESS %.0f of %d (%.3f%%), max weight %.4f, "
          "healthy=%s" % (wd["kish_ess"], wd["n_samples"],
          100 * wd["kish_fraction"], wd["max_weight"], wd["healthy"]))
    print("  window free energies f (kT), first five: %s"
          % np.round(wham["f"][:5], 3).tolist())

    # the same equations solved the slow, literal way, on the SAME data
    sub = x_all[::20]
    u_sub = reduced_bias_matrix(sub, path, k, system.kT)
    N_sub = np.full(len(path), u_sub.shape[1] / len(path))
    f_sc, info_sc = solve_self_consistent(u_sub, N_sub, tol=1e-11, max_iter=40000)
    f_lb, _ = solve_lbfgs(u_sub, N_sub)
    print("  self-consistent vs L-BFGS on identical data: %.2e kT (%d iterations)"
          % (np.max(np.abs(f_sc - f_lb)), info_sc["iterations"]))

    w = wham["weights"]
    s_w, A_w, counts = pmf_from_weights(s_all, w, system.kT, bins=48,
                                        range_=(s_dense[0], s_dense[-1]))
    A_w = zero_at_start(s_w, A_w)

    s_ref, A_ref = system.reference_pmf(path, n_grid=121, n_bins=48, tube_radius=tube)
    A_ref = zero_at_start(s_ref, A_ref)
    s_all_sp, A_all_sp = system.reference_pmf(path, n_grid=121, n_bins=48)
    A_all_sp = zero_at_start(s_all_sp, A_all_sp)

    s_mf, A_mf, _ = mean_force_pmf(prod, path, np.full(3, k), path=path)
    A_mf = zero_at_start(s_mf, A_mf)

    def barrier(s, A):
        good = np.isfinite(A)
        return float(np.nanmax(A[good]))

    print("\n  barrier along the path (energy units; kT = %.2f):" % system.kT)
    print("    exact, tube-restricted  %.3f   <- like-for-like reference for umbrellas"
          % barrier(s_ref, A_ref))
    print("    binless WHAM            %.3f   <- should match the line above"
          % barrier(s_w, A_w))
    print("    exact, ALL space        %.3f   <- a different quantity (see README)"
          % barrier(s_all_sp, A_all_sp))
    print("    mean force              %.3f   <- path-tangent integral, not the"
          % barrier(s_mf, A_mf))
    print("                                      projected PMF (see README)")

    # ------------------------------------------------- phase 4: diagnostics
    print("\nphase 4: sampling diagnostics")
    edges = np.linspace(s_dense[0], s_dense[-1], 61)
    s_traj = project_on_path(prod.reshape(-1, 3), dense, s_dense).reshape(prod.shape[:2])
    diag = summarise(prod, path, np.full(3, k), system.kT, s_traj=s_traj, edges=edges)
    for key in ("overlap_min", "overlap_median", "overlap_predicted_min",
                "spacing_median", "restraint_sigma", "spacing_over_sigma_max",
                "flux_min", "flux_median", "flux_n_below_50"):
        print("  %-24s %s" % (key, np.round(diag[key], 4)))

    # ======================================================== figure 1: the string
    fig, ax = plt.subplots(1, 3, figsize=(24, 7.4))
    gx = np.linspace(-2.0, 2.0, 220)

    GY, GX = np.meshgrid(gx, gx, indexing="ij")
    Vxy = system.energy(np.stack([GX, GY, np.full_like(GX, 0.3)], axis=-1))
    ax[0].contourf(GX, GY, np.minimum(Vxy, 4.0), levels=28, cmap="viridis")
    ax[0].plot(init[:, 0], init[:, 1], "o--", color=C_INIT, lw=3, ms=9, label="initial (straight)")
    ax[0].plot(path[:, 0], path[:, 1], "o-", color=C_WHAM, lw=3.4, ms=10, label="converged string")
    ax[0].set_xlabel("x"); ax[0].set_ylabel("y")
    ax[0].set_title("A. x-y   (slice z = 0.3)")
    ax[0].legend(loc="upper center")

    GZ, GX2 = np.meshgrid(gx, gx, indexing="ij")
    Vxz = system.energy(np.stack([GX2, np.full_like(GX2, 0.5), GZ], axis=-1))
    ax[1].contourf(GX2, GZ, np.minimum(Vxz, 4.0), levels=28, cmap="viridis")
    ax[1].plot(init[:, 0], init[:, 2], "o--", color=C_INIT, lw=3, ms=9)
    ax[1].plot(path[:, 0], path[:, 2], "o-", color=C_WHAM, lw=3.4, ms=10)
    ax[1].set_xlabel("x"); ax[1].set_ylabel("z")
    ax[1].set_title("B. x-z   (slice y = 0.5)")

    it = np.arange(1, len(run.arc_length) + 1)
    ax[2].plot(it, run.arc_length, "-", color=C_WHAM, lw=3.2, label="arc length")
    ax[2].set_xlabel("string iteration"); ax[2].set_ylabel("arc length")
    a2 = ax[2].twinx()
    a2.semilogy(it, run.max_centre_move, "-", color=C_MF, lw=3.0, label="max centre move")
    a2.set_ylabel("max centre move", fontsize=FS)
    a2.tick_params(labelsize=FS - 2)
    ax[2].set_title("C. convergence")
    h1, l1 = ax[2].get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    ax[2].legend(h1 + h2, l1 + l2, loc="center right")
    ax[2].grid(alpha=0.3)

    fig.suptitle("Finite-temperature string on a 3-D analytical potential")
    fig.tight_layout(rect=[0, 0, 0.995, 0.94])
    fig.savefig(os.path.join(HERE, "fig1_string.png"), dpi=105)

    # =================================================== figure 2: the free energy
    fig, ax = plt.subplots(1, 2, figsize=(21, 8.0))

    ax[0].plot(s_ref, A_ref, "-", color=C_EXACT, lw=4.0, label="exact (tube-restricted)")
    ax[0].plot(s_all_sp, A_all_sp, ":", color=C_EXACT, lw=3.0, label="exact (all space)")
    ax[0].plot(s_w, A_w, "o-", color=C_WHAM, lw=3.0, ms=7, label="binless WHAM")
    ax[0].plot(s_mf, A_mf, "s--", color=C_MF, lw=3.0, ms=9, label="mean force")
    ax[0].set_xlabel("path coordinate  s"); ax[0].set_ylabel("A")
    ax[0].set_title("A. free energy along the string")
    ax[0].legend(loc="lower center", ncol=2); ax[0].grid(alpha=0.3)

    good = np.isfinite(A_w) & np.isfinite(np.interp(s_w, s_ref, A_ref))
    err = A_w[good] - np.interp(s_w[good], s_ref, A_ref)
    ax[1].axhline(0, color="k", lw=2)
    ax[1].plot(s_w[good], err, "o-", color=C_WHAM, lw=3.0, ms=7, label="WHAM - exact")
    err_mf = A_mf - np.interp(s_mf, s_ref, A_ref)
    ax[1].plot(s_mf, err_mf, "s--", color=C_MF, lw=3.0, ms=9, label="mean force - exact")
    ax[1].axhspan(-system.kT, system.kT, color="#cccccc", alpha=0.5, label="+- kT")
    ax[1].set_xlabel("path coordinate  s"); ax[1].set_ylabel("error")
    ax[1].set_title("B. deviation from the exact answer")
    ax[1].legend(loc="best"); ax[1].grid(alpha=0.3)

    fig.suptitle("Estimators vs the exact answer - compared on the region actually sampled")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(HERE, "fig2_free_energy.png"), dpi=105)

    # ====================== figure 3: reprojection payoff + sampling diagnostics
    fig, ax = plt.subplots(1, 3, figsize=(24, 7.4))

    sx, Ax, _ = pmf_from_weights(x_all[:, 0], w, system.kT, bins=48)
    ax[0].plot(sx, Ax, "-", color=C_WHAM, lw=3.4)
    ax[0].set_xlabel("x"); ax[0].set_ylabel("A")
    ax[0].set_title("A. same weights, projected on x")
    ax[0].grid(alpha=0.3)

    q1, q2, A2 = pmf_2d_from_weights(x_all[:, 0], x_all[:, 1], w, system.kT, bins=44)
    Q1, Q2 = np.meshgrid(q1, q2, indexing="ij")
    m = np.isfinite(A2)
    pc = ax[1].contourf(Q1, Q2, np.where(m, A2, np.nan),
                        levels=np.linspace(0, np.nanpercentile(A2[m], 97), 22),
                        cmap="viridis")
    cb = fig.colorbar(pc, ax=ax[1]); cb.set_label("A", fontsize=FS)
    cb.ax.tick_params(labelsize=FS - 3)
    ax[1].plot(path[:, 0], path[:, 1], "o-", color="#c1442e", lw=3.0, ms=8)
    ax[1].set_xlabel("x"); ax[1].set_ylabel("y")
    ax[1].set_title("B. and on (x, y)")

    ov = neighbour_overlap(prod, path, np.full(3, k), system.kT)
    fl = transition_flux(s_traj, edges)
    mid = 0.5 * (s_img[:-1] + s_img[1:])
    ax[2].plot(mid, ov, "o-", color=C_MF, lw=3.0, ms=9, label="neighbour overlap")
    ax[2].set_xlabel("path coordinate  s"); ax[2].set_ylabel("overlap coefficient")
    ax[2].set_ylim(0, 1)
    a2 = ax[2].twinx()
    # transition_flux returns one value per INTERIOR boundary -> edges[1:-1]
    a2.semilogy(edges[1:-1], np.maximum(fl, 1), "-", color="#2b6cb0", lw=2.6,
                label="transition flux")
    a2.set_ylabel("transitions crossing", fontsize=FS)
    a2.tick_params(labelsize=FS - 2)
    ax[2].set_title("C. do the windows connect?")
    h1, l1 = ax[2].get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    ax[2].legend(h1 + h2, l1 + l2, loc="lower center")
    ax[2].grid(alpha=0.3)

    fig.suptitle("Binless WHAM: solve once, project anywhere - then check the sampling")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(HERE, "fig3_reprojection.png"), dpi=105)

    print("\nwrote fig1_string.png, fig2_free_energy.png, fig3_reprojection.png in\n  %s"
          % HERE)
    return dict(system=system, run=run, path=path, wham=wham,
                exact=(s_ref, A_ref), wham_pmf=(s_w, A_w), mf=(s_mf, A_mf), diag=diag)


if __name__ == "__main__":
    main()
