"""A 3-D analytical test system for the finite-temperature string method.

The potential is a sum of isotropic/anisotropic Gaussians plus a quartic confinement.
It is built so that the *straight line* between the two minima runs straight over a
high central barrier, while the true minimum-free-energy path arcs out through +y and
+z.  A string initialised on the straight line therefore has to discover the detour,
which is the whole point of the demonstration.

Everything is in reduced units.  Set kT when you construct `System`.

Nothing here is specific to any particular system; it is a stand-in for "a handful of
collective variables measured on a protein", with the advantage that the exact answer
is computable.
"""
from __future__ import annotations

import numpy as np

# (amplitude, (x0, y0, z0), (ax, ay, az))  ->  amplitude * exp(-[ax(x-x0)^2 + ...])
_TERMS = (
    (-3.0, (-1.0, 0.0, 0.0), (4.0, 4.0, 4.0)),   # well A
    (-3.0, (+1.0, 0.0, 0.0), (4.0, 4.0, 4.0)),   # well B
    (+4.0, (0.0, 0.0, 0.0), (3.0, 6.0, 6.0)),    # central ridge: blocks the straight line
    (-1.2, (0.0, 1.0, 0.6), (2.0, 2.0, 2.0)),    # the curved channel the MFEP must find
)
_QUARTIC = 0.08      # confinement, keeps the partition function finite


class System:
    """V(r), grad V(r), and an exact reference free energy by 3-D quadrature."""

    def __init__(self, kT: float = 0.4):
        self.kT = float(kT)
        self.n_dim = 3
        self.minimum_a = np.array([-1.0, 0.0, 0.0])
        self.minimum_b = np.array([+1.0, 0.0, 0.0])

    # ---------------------------------------------------------------- energy
    def energy(self, r):
        """r: (..., 3) -> (...,)"""
        r = np.asarray(r, dtype=float)
        v = _QUARTIC * np.sum(r ** 4, axis=-1)
        for amp, c, a in _TERMS:
            d2 = np.zeros(r.shape[:-1])
            for k in range(3):
                d2 = d2 + a[k] * (r[..., k] - c[k]) ** 2
            v = v + amp * np.exp(-d2)
        return v

    def gradient(self, r):
        """Analytic dV/dr.  r: (..., 3) -> (..., 3)"""
        r = np.asarray(r, dtype=float)
        g = 4.0 * _QUARTIC * r ** 3
        for amp, c, a in _TERMS:
            d2 = np.zeros(r.shape[:-1])
            for k in range(3):
                d2 = d2 + a[k] * (r[..., k] - c[k]) ** 2
            e = amp * np.exp(-d2)
            for k in range(3):
                g[..., k] += e * (-2.0 * a[k] * (r[..., k] - c[k]))
        return g

    def check_gradient(self, seed: int = 0, n: int = 64, h: float = 1e-5) -> float:
        """Central finite differences vs the analytic gradient; returns max abs error."""
        rng = np.random.default_rng(seed)
        r = rng.uniform(-1.8, 1.8, size=(n, 3))
        ga = self.gradient(r)
        gn = np.empty_like(ga)
        for k in range(3):
            dp = r.copy(); dp[:, k] += h
            dm = r.copy(); dm[:, k] -= h
            gn[:, k] = (self.energy(dp) - self.energy(dm)) / (2 * h)
        return float(np.max(np.abs(ga - gn)))

    # ------------------------------------------------------- exact reference
    def reference_pmf(self, path, n_grid: int = 141, extent: float = 2.2,
                      n_bins: int = 60, tube_radius: float | None = None):
        """Exact A(s) along `path`, by Boltzmann quadrature on a regular 3-D grid.

        s(r) is the arc length of the nearest point on the (densely interpolated) path -
        the same "project onto the path" convention we use for the protein string.

        tube_radius
            If given, the quadrature is restricted to points within that perpendicular
            distance of the path.  THIS IS USUALLY WHAT YOU WANT.  The lab restraint
            acts on every CV, so the windows sample a narrow tube around the string,
            and umbrella sampling can only report the free energy of the region it
            visited.  Comparing a tube-sampled estimate against an all-space integral
            charges the estimator for perpendicular entropy it never had access to.
            Pass the perpendicular extent your production run actually reached.

        Returns (s_centres, A) with A in the same units as kT, zeroed at its minimum.
        """
        path = np.asarray(path, dtype=float)
        dense, s_dense = densify(path, n_per_segment=24)

        ax = np.linspace(-extent, extent, n_grid)
        X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
        R = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
        V = self.energy(R)

        # Boltzmann weight, stabilised
        w = np.exp(-(V - V.min()) / self.kT)

        s, perp = project_on_path(R, dense, s_dense, chunk=200_000, return_perp=True)
        if tube_radius is not None:
            w = np.where(perp <= tube_radius, w, 0.0)

        edges = np.linspace(s_dense[0], s_dense[-1], n_bins + 1)
        hist, _ = np.histogram(s, bins=edges, weights=w)
        centres = 0.5 * (edges[:-1] + edges[1:])
        with np.errstate(divide="ignore"):
            A = -self.kT * np.log(hist)
        A = A - np.nanmin(A[np.isfinite(A)])
        return centres, A


# --------------------------------------------------------------- path helpers
def densify(path, n_per_segment: int = 24):
    """Linearly interpolate a polyline and return (points, cumulative arc length)."""
    path = np.asarray(path, dtype=float)
    pts = [path[0]]
    for i in range(len(path) - 1):
        t = np.linspace(0.0, 1.0, n_per_segment + 1)[1:]
        pts.append(path[i] + t[:, None] * (path[i + 1] - path[i]))
    dense = np.vstack(pts)
    seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return dense, s


def project_on_path(points, dense, s_dense, chunk: int = 200_000,
                    return_perp: bool = False):
    """Arc length of the nearest dense-path point, for each row of `points`.

    With return_perp=True also returns the perpendicular distance to the path, which
    is what tells you how wide a tube your restraints are actually sampling.
    """
    points = np.asarray(points, dtype=float)
    out = np.empty(len(points))
    perp = np.empty(len(points))
    for lo in range(0, len(points), chunk):
        hi = min(lo + chunk, len(points))
        d = points[lo:hi, None, :] - dense[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", d, d)
        j = np.argmin(d2, axis=1)
        out[lo:hi] = s_dense[j]
        perp[lo:hi] = np.sqrt(d2[np.arange(hi - lo), j])
    return (out, perp) if return_perp else out


def straight_string(system: System, n_images: int = 20):
    """The deliberately bad starting guess: a straight line through the ridge."""
    t = np.linspace(0.0, 1.0, n_images)[:, None]
    return system.minimum_a + t * (system.minimum_b - system.minimum_a)
