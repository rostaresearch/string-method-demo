"""The Rosta-lab string update (reparametrisation), unchanged.

This is the same arithmetic we run on the protein strings.  Source lineage:

    the Rosta lab's original string.py
      -> the copy used for production protein runs
      -> this file

Nothing numerical has been changed.  The algorithm, operation by operation:

1. Average each window's samples to one point per image:  avg[i, :] = <x>_i
2. Fit a natural ("not-a-knot") cubic spline to each CV separately, as a function of
   the image index, with knots at linspace(1, new_nw, nw).
3. Evaluate |dx/di| on a fine grid x = arange(1, new_nw + 0.1, 0.1), i.e. 10 grid
   points per image interval, and accumulate the arc length by the trapezoid rule.
4. Place the new images at equal arc length - but *snapped to the nearest grid point*,
   not at the exact target.  This quantises image placement to 1/10 of an image
   interval and is the reason the resulting chord lengths are never perfectly equal.
5. The new centres are the spline evaluated at those grid points.

Point 4 is worth knowing about: if you measure the spacing of a reparametrised string
by straight-line (chord) distance you will find it noticeably non-uniform, and conclude
the reparametrisation is broken.  It is not.  The algorithm equalises *spline arc
length*, and chords differ from arcs wherever the path curves.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

_trapz = getattr(np, "trapz", None) or getattr(np, "trapezoid")


def _core(data, order="c", new_nw=None):
    nw, n_frames, nc = data.shape
    avg = np.mean(data, axis=1)
    if new_nw is None:
        new_nw = nw
    x = np.arange(1.0, new_nw + 0.1, 0.1)
    y = np.zeros(shape=x.shape, dtype=np.float64)
    if order == "c":
        p = []
        for i in range(nc):
            p.append(CubicSpline(np.linspace(1, new_nw, nw), avg[:, i]))
            y += np.square(p[i](x, 1))
    else:
        p = np.empty(shape=(nc, order + 1), dtype=np.float64)
        for i in range(nc):
            p[i] = np.polyfit(np.linspace(1, new_nw, nw), avg[:, i], order)
            y += np.square(np.polyval(np.polyder(p[i]), x))
    y = np.sqrt(y)
    l = _trapz(y, x)
    li = np.linspace(0, l, new_nw)
    flen = np.empty(shape=x.shape, dtype=np.float64)
    for i in range(flen.shape[0]):
        flen[i] = _trapz(y[:i + 1], x[:i + 1])
    pt = np.empty(shape=new_nw, dtype=np.int_)
    for i in range(new_nw):
        v = np.abs(flen - li[i])
        hits = np.where(v == np.min(v))[0]
        if hits.shape[0] != 1:
            raise ValueError(
                "opt_string: equal-arc target %d is equidistant from grid points %s; "
                "the string has zero-length segments" % (i, hits.tolist()))
        pt[i] = hits[0]
    g = np.empty(shape=(nc, x.shape[0]), dtype=np.float64)
    newconstr = np.empty(shape=(nc, new_nw), dtype=np.float64)
    for i in range(nc):
        if order == "c":
            g[i] = p[i](x)
        else:
            g[i] = np.polyval(p[i], x)
        newconstr[i] = g[i, pt]
    newconstr = np.transpose(newconstr)
    info = {"arc_length": float(l), "grid_x": x, "grid_index": pt,
            "grid_x_selected": x[pt], "cum_length": flen,
            "equal_arc_targets": li, "avg": avg}
    return newconstr, info


def opt_string(data, order="c", new_nw=None):
    """data[n_windows, n_frames, n_cv] -> new centres [new_nw, n_cv]."""
    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError("data must be [n_windows, n_frames, n_cv]")
    return _core(data, order=order, new_nw=new_nw)[0]


def opt_string_info(data, order="c", new_nw=None):
    """Same numbers as opt_string, plus the arc-length diagnostics."""
    return _core(np.asarray(data), order=order, new_nw=new_nw)


def smooth_interior(avg, kappa):
    """Optional mild smoothing (NOT in the lab code; default off).

    Interior images -> (1-kappa)*x_i + kappa/2*(x_{i-1} + x_{i+1}); endpoints fixed.
    Useful if the string develops kinks on noisy data.  kappa = 0 is a no-op.
    """
    avg = np.array(avg, dtype=np.float64, copy=True)
    if kappa and avg.shape[0] > 2:
        avg[1:-1] = (1.0 - kappa) * avg[1:-1] + 0.5 * kappa * (avg[:-2] + avg[2:])
    return avg


def reparametrize(data, order="c", new_nw=None, smooth_kappa=0.0):
    """Driver entry point.  smooth_kappa == 0 reproduces the lab opt_string exactly."""
    data = np.asarray(data, dtype=np.float64)
    if not smooth_kappa:
        return opt_string_info(data, order=order, new_nw=new_nw)
    avg = smooth_interior(np.mean(data, axis=1), smooth_kappa)
    return opt_string_info(avg[:, None, :], order=order, new_nw=new_nw)


def spline_arc_spacing(info):
    """True spline arc length between consecutive placed images.

    Use this, not chord length, to check that reparametrisation worked.  The residual
    non-uniformity should sit inside one grid step of arc length (see module docstring).
    """
    flen, pt = info["cum_length"], info["grid_index"]
    seg = np.diff(flen[pt])
    target = info["arc_length"] / (len(pt) - 1)
    # The grid is uniform in IMAGE INDEX (0.1 steps), not in arc length, so one grid
    # step spans more arc where |dx/di| is large.  Each placement can be off by up to
    # half a local step, so a segment can be off by up to one full step: use the
    # largest step as the bound, not target*0.1.
    grid_quantum = float(np.max(np.diff(flen)))
    dev = float(np.max(np.abs(seg - target)))
    return {"segments": seg, "target": target, "grid_quantum": grid_quantum,
            "max_abs_deviation": dev,
            "within_quantisation": bool(dev <= grid_quantum)}
