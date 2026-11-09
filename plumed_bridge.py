"""Bridge to a real MD engine via PLUMED.

The string logic in this package never looks at Cartesian coordinates - it only needs
CV timeseries of shape (n_images, n_frames, n_cv).  So to run the same method on a real
system you replace `sampler.propagate` with "run MD under PLUMED restraints and read
back the COLVAR files", and everything else is unchanged.

The loop becomes:

    for iteration in range(n_iterations):
        write_plumed_inputs(centres, kappa, cv_defs, outdir)   # one file per image
        ... launch n_images MD runs (your queue system) ...
        traj = read_colvars(outdir, n_cv)                      # (n_images, n_frames, n_cv)
        centres, info = labstring.reparametrize(traj)

PLUMED does the restraints and the CV evaluation.  It does NOT do the reparametrisation
or the binless reweighting - those stay here.

Units: PLUMED defaults to kJ/mol and nm.  Decide once and be consistent; the biggest
single source of wrong barriers in this kind of calculation is a kappa that is in the
wrong units for the CV it is attached to.
"""
from __future__ import annotations

import os
import glob

import numpy as np

HEADER = """# Generated for string image {image:03d}, iteration {iteration:03d}
# centres below are row {image} of allconstr_{iteration}
UNITS LENGTH=nm ENERGY=kj/mol TIME=ps
"""


def write_plumed_input(path, cv_defs, centres_row, kappa, image, iteration,
                       stride=100, colvar="COLVAR"):
    """Write one plumed.dat restraining every CV at its string centre.

    cv_defs      list of PLUMED CV lines, e.g.
                 ["d1: DISTANCE ATOMS=12,47", "d2: DISTANCE ATOMS=88,131"]
                 The label before the colon is reused in ARG.
    centres_row  (n_cv,) the restraint centres for this image
    kappa        scalar or (n_cv,) force constants, in PLUMED units
    """
    labels = [line.split(":")[0].strip() for line in cv_defs]
    if len(labels) != len(centres_row):
        raise ValueError("%d CV definitions but %d centres" % (len(labels), len(centres_row)))
    kappa = np.broadcast_to(np.asarray(kappa, dtype=float), (len(labels),))

    lines = [HEADER.format(image=image, iteration=iteration)]
    lines.extend(cv_defs)
    lines.append("")
    lines.append("restraint: RESTRAINT ...")
    lines.append("   ARG=%s" % ",".join(labels))
    lines.append("   AT=%s" % ",".join("%.6f" % v for v in centres_row))
    lines.append("   KAPPA=%s" % ",".join("%.6f" % v for v in kappa))
    lines.append("...")
    lines.append("")
    lines.append("PRINT ARG=%s FILE=%s STRIDE=%d" % (",".join(labels), colvar, stride))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def write_plumed_inputs(outdir, cv_defs, centres, kappa, iteration, **kw):
    """One plumed.dat per image, in outdir/img_000/, outdir/img_001/, ..."""
    centres = np.asarray(centres, dtype=float)
    made = []
    for i, row in enumerate(centres):
        d = os.path.join(outdir, "img_%03d" % i)
        os.makedirs(d, exist_ok=True)
        made.append(write_plumed_input(os.path.join(d, "plumed.dat"), cv_defs, row,
                                       kappa, image=i, iteration=iteration, **kw))
    return made


def read_colvar(path, n_cv, discard=0):
    """Read a PLUMED COLVAR file -> (n_frames, n_cv).

    Assumes the first column is time and the next n_cv columns are the CVs, which is
    what write_plumed_input produces.  `discard` drops leading frames (equilibration).
    """
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < n_cv + 1:
        raise ValueError("%s has %d columns, expected at least %d"
                         % (path, data.shape[1], n_cv + 1))
    return data[discard:, 1:1 + n_cv]


def read_colvars(outdir, n_cv, discard=0, colvar="COLVAR"):
    """Collect every image's COLVAR into (n_images, n_frames, n_cv).

    Raises if the images have different frame counts - that usually means one run died,
    and silently truncating would quietly bias the string.
    """
    dirs = sorted(glob.glob(os.path.join(outdir, "img_*")))
    if not dirs:
        raise FileNotFoundError("no img_* directories under %s" % outdir)
    per = [read_colvar(os.path.join(d, colvar), n_cv, discard=discard) for d in dirs]
    n = {a.shape[0] for a in per}
    if len(n) != 1:
        raise ValueError("images have different frame counts %s; check for failed runs"
                         % sorted(n))
    return np.stack(per, axis=0)


if __name__ == "__main__":
    # A three-distance example, written to ./plumed_example/.  Substitute your own
    # atom indices; a real run typically uses of order ten distances.
    cv_defs = [
        "d1: DISTANCE ATOMS=12,47",       # e.g. a contact that breaks
        "d2: DISTANCE ATOMS=88,131",      # e.g. a contact that forms
        "d3: DISTANCE ATOMS=64,152",      # e.g. a loop-closure distance
    ]
    centres = np.array([[0.27, 1.18, 1.13],
                        [0.31, 1.15, 1.14],
                        [0.38, 1.10, 1.15]])          # nm
    files = write_plumed_inputs("plumed_example", cv_defs, centres,
                                kappa=2092.0, iteration=0)   # kJ/mol/nm^2
    print("wrote:")
    for f in files:
        print("  ", f)
    print()
    print(open(files[0]).read())
