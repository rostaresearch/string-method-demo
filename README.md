# Finite-temperature string method + binless WHAM, on a 3-D analytical system

A small, self-contained, dependency-light implementation of the string method we use
for protein free-energy pathways, demonstrated on a 3-D toy potential where the exact
answer can be computed and compared against.

The reparametrisation in `labstring.py` is the same code used for the finite-temperature
string calculations in [Rosta, Nowotny, Yang & Hummer, *J. Am. Chem. Soc.* **133**,
8934-8941 (2011)](https://doi.org/10.1021/ja200173a) - see
[Origin and references](#origin-and-references).

Everything is `numpy` + `scipy` + `matplotlib`. No MD engine, no PLUMED, no compilation.
A full run takes a couple of minutes on one core.

```bash
python run_demo.py
```

---

## What it produces

**The string finds the curved path from a straight-line start.** Left and middle are
slices through the potential; the blue line is the initial guess, red the converged
string. Right: arc length plateaus and the per-iteration centre movement falls.

![string convergence](fig1_string.png)

**Binless WHAM reproduces the exact free energy.** On this run the barrier is 2.332
against an exact tube-restricted 2.326 - agreement to 0.015 kT. Mean force sits lower
because it integrates the path-tangent force rather than the projected PMF; see below.

![free energy](fig2_free_energy.png)

**One solve, three projections.** The same per-sample weights give the free energy along
the path, along x, and on the (x, y) plane - with no re-solve. Right-hand panel is the
sampling check that should accompany any of them.

![reprojection](fig3_reprojection.png)

---

## Why a 3-D analytical system

The potential (`potential3d.py`) is a sum of Gaussians plus a quartic confinement, with
two minima at `(-1, 0, 0)` and `(+1, 0, 0)`. The straight line between them runs
directly over a high central ridge; the true minimum-free-energy path arcs out through
`+y` and `+z` through a channel.

So a string initialised on the straight line **has to discover the detour**. That is the
behaviour worth seeing, and because the system is analytical we can compute the exact
free energy along the path by Boltzmann quadrature on a 3-D grid and check the answer.

Think of the three coordinates as standing in for "the collective variables you measured
on your protein" - with the advantage that here you know the truth.

---

## The method

### 1. Umbrella windows along a path

The string is `N` images. Image `i` is an umbrella window restrained at centre `c_i`:

```
U_i(x) = sum_j  0.5 * k_j * (x_j - c_ij)^2
```

one harmonic term per collective variable. This is exactly the restraint form used in
the protein work.

### 2. One string iteration

```
for each image i:
    set the restraint centres to row i of the current string
    continue image i from its OWN end state of the previous iteration
    equilibrate (discard), then sample
when every image is done:
    reparametrise
```

Carrying each image's state forward (rather than restarting from the centre) is what
keeps the orthogonal degrees of freedom equilibrated as the string moves. It costs
nothing and matters.

### 3. Reparametrisation - `labstring.py`

This is the lab's `opt_string`, copied operation for operation from the original
`string.py` used for production protein runs. Nothing numerical has been changed.

1. average each window's samples to one point per image
2. fit a cubic spline (not-a-knot) to each CV against image index
3. evaluate `|dx/di|` on a grid of 10 points per image interval, accumulate arc length
   by the trapezoid rule
4. place the new images at **equal arc length**, snapped to the nearest grid point
5. new centres = the spline at those grid points

**Gotcha worth knowing (step 4).** Image placement is quantised to 1/10 of an image
interval. If you measure the spacing of a reparametrised string by straight-line
**chord** distance you will find it visibly non-uniform and conclude the code is broken.
It is not: the algorithm equalises **spline arc length**, and chords differ from arcs
wherever the path curves. Use `labstring.spline_arc_spacing(info)`, which checks the
residual against the grid quantisation.

---

## Binless WHAM - `reweight.py`

Ordinary WHAM histograms the samples, then solves for one free-energy offset per window.
You must choose the bins before you solve, and you must redo the whole solve to get the
free energy along a different coordinate.

Binless WHAM never histograms. It uses the reduced bias at every individual sample:

```
f_k  =  -ln  sum_n  exp(-u_k(x_n)) / sum_j N_j exp(f_j - u_j(x_n))        (1)

w_n  proportional to  1 / sum_j N_j exp(f_j - u_j(x_n))                   (2)

A(q) =  -kT ln sum_n w_n delta(q(x_n) - q)                                (3)
```

**Solve once, project anywhere.** After (1) and (2) you hold one weight per sample. The
free energy along *any* coordinate is a weighted histogram, equation (3) - the path
coordinate, a single CV, a 2-D plane, or something you only thought of later. You never
touch the estimator again. `run_demo.py` figure 3 shows the same weights projected three
ways.

These are the MBAR equations (Shirts & Chodera 2008); "binless WHAM" is the older name
for the same thing. Nothing here is novel - the point is seeing it in about forty lines.

Two solvers are provided and the test asserts they agree:

* `solve_self_consistent` - equation (1) iterated literally. Short, readable, linear
  convergence. Read this one to understand the method.
* `solve_lbfgs` - the same stationary point found by minimising the Shirts-Chodera
  convex objective. Use this one in anger.

### Always check the weight health

Binless WHAM gives every sample a weight `w_n` proportional to
`1 / sum_j N_j exp(f_j - u_j(x_n))`. If a sample lands where **every** window has
negligible probability - a rare excursion outside the union of the window supports - that
denominator collapses and the one sample can carry a large share of the total
probability. The free energy then rests on a handful of frames however many you
collected, and the reweighted PMF shows narrow spikes where they fall.

With restraints on every CV this is easy to hit, because the windows only cover a thin
tube and anything that leaves the tube is outside all of them.

`reweight.weight_diagnostics` reports the Kish effective sample size `1 / sum w^2`, the
largest single weight, and the share held by the top few. The two runs in this package
bracket the useful range:

| run | Kish ESS | largest single weight | barrier |
|---|---|---|---|
| 20 images, long production | **1147 of 80000** (1.4%) | 1.5% | correct to 0.015 kT |
| fewer images, short production | **3.9** | **50.4%** | worthless |

Same code, same estimator. The second one still produces a smooth-looking curve and a
plausible number - which is exactly why you check the weights rather than the plot. A few
hundred effective samples is a reasonable basis for a PMF; four is not.

### Things that will bite you

* Equation (1) is invariant under `f -> f + const`. One offset is arbitrary; we fix
  `f[0] = 0`. Never compare `f` between two separate solves without re-zeroing.
* `u_k` must be **reduced** (divided by kT), dimensionless.
* Every window must carry **its own** force constants and centre. Pooling windows that
  had different restraints under one assumed restraint is the commonest route to a
  confidently wrong barrier. In a validation against a synthetic barrier of known
  height, assuming one global `k` when the data had two nearly doubled the result.
* Binless WHAM needs the windows to overlap **in configuration space**. If adjacent
  windows share no samples, their relative offset is poorly determined and iterating
  harder will not help.

---

## What free energy are you actually measuring? (read this one)

The lab restraint acts on **every** CV:

```
U_i(x) = sum_j 0.5 * k_j * (x_j - c_ij)^2
```

so the windows do not just localise you *along* the string - they also confine you
*perpendicular* to it, inside a tube of radius about `sigma = sqrt(kT/k)`.

That means umbrella sampling reports the free energy of **the tube**, not of all space.
The two agree only when the potential's own perpendicular confinement is tighter than
your restraint. If the restraint is the tighter one - which is the normal situation -
the tube estimate is missing perpendicular entropy, and will differ.

`run_demo.py` prints both, and they differ by a visible amount on this system:

```
exact, tube-restricted   <- the like-for-like reference for umbrella sampling
binless WHAM
mean force
exact, ALL space         <- a different quantity
```

This is not a bug in anyone's estimator and it does not go away with more sampling.
It is a statement about what the restrained ensemble contains. Two practical
consequences:

1. When you validate an estimator against a known answer, restrict the reference to
   the region your windows actually visited - `System.reference_pmf(..., tube_radius=)`
   takes the perpendicular extent from your production run.
2. When you quote a barrier from a string calculation, it is a barrier *within the
   sampled tube*. If the path is wrong, or the tube too narrow to contain the real
   transition region, no estimator will tell you - they will all agree on the wrong
   number. Only widening the restraints, or a better path, will.

---

## Mean force - the independent cross-check

`reweight.mean_force_pmf` integrates

```
dA/ds = k (c - <x>) . t
```

along the string, using only each window's **own** average. It needs no overlap
between windows at all, which is what makes it valuable: when WHAM and mean force
disagree badly, it is usually WHAM's overlap that has failed, not the physics.

**But it is not the same quantity as the projected PMF, and the demo shows that.**

Integrating the mean force along the path tangent gives the free energy change *along
the path*. The projected PMF, `A(s) = -kT ln integral exp(-V/kT) delta(s(r)-s) dr`,
additionally contains the perpendicular partition function at each `s` - a Jacobian /
entropy term that varies when the path curves or the tube width changes. Integrating
the tangential force does not pick that up.

On this system the two differ by a visible amount, and `run_demo.py` reports both. The
check that settles which effect it is: differentiate the exact reference and compare
*pointwise* against `k(c - <x>).t`. Integrating the exact gradient on the same 20
images reproduces the exact barrier, while the mean-force gradient correlates only
weakly with it - so the gap is the projection, not the integration step size.

Practical reading:

* use mean force as a **robustness** check (it survives poor overlap), not as a
  second opinion on the absolute projected barrier;
* if you need them to agree, either compare both against a reference computed the
  same way, or add the Jacobian term explicitly;
* the agreement to compare is **binless WHAM against the tube-restricted exact
  answer**, which in the demo matches to well under kT.

---

## Diagnostics - `diagnostics.py`

Two different questions that are easy to conflate:

| function | question | which estimators need it |
|---|---|---|
| `neighbour_overlap` | do adjacent windows share **configurations**? | WHAM, binless WHAM |
| `transition_flux` | do trajectories actually **cross** between them? | DHAM, MSMs, anything transition-based |

They are not the same thing. One window wandering back and forth across a region
generates flux there even when no *pair* of windows overlaps; and two windows can
overlap statically while almost nothing crosses between them.

This distinction costs real time if you miss it. A region can show zero pairwise
histogram overlap yet still be fully connected in the transition graph - while its
weakest boundary carries **three orders of magnitude fewer crossings than the median**.
A transition-matrix estimator will report a confident barrier straight through such a
bottleneck, while mean force disagrees by a factor of two.

**Report the minimum over boundaries, never the mean.** `predicted_gaussian_overlap`
also lets you check, before running anything, whether your chosen force constant and
image spacing can overlap at all: `2*Phi(-D/2sigma)` with `sigma = sqrt(kT/k)`.

---

## Part 2: Hamiltonian replica exchange

```bash
python run_demo_hrex.py
```

The 2011 study combined the string method with Hamiltonian replica exchange, and `hrex.py`
adds that second ingredient. Every window runs the same unbiased potential and differs
only in its restraint, so proposing to swap the configurations held by windows *i* and
*j* costs nothing but two restraint evaluations - the `V` terms cancel:

```
Delta  =  [ U_i(x_j) + U_j(x_i) - U_i(x_i) - U_j(x_j) ] / kT
accept with probability min(1, exp(-Delta))
```

Neighbour pairs are swept in two alternating passes, so no window appears twice in one
pass:

```
parity 0 :  1-2, 3-4, 5-6, ...
parity 1 :  2-3, 4-5, 6-7, ...
```

Alternating the parity is what lets a configuration migrate along the whole string
rather than rattling inside one pair. In the demo (16 windows, exchange every 200 steps,
~32% acceptance) replicas span 13.4 of 15 windows on average, with 8 full traversals -
figure 4B.

### Two things to be clear about

**What is swapped.** We swap the *configurations* and leave window identities fixed, so
window *k* always holds samples generated under bias *k*. **Record samples per window,
not per replica, and the binless WHAM needs no change at all** - every sample still
carries the bias of the window it was recorded in. Swapping the labels instead is
formally equivalent but much easier to get wrong.

**The free energy must be unchanged - and it is.** On the demo:

| | barrier | Kish ESS | largest weight |
|---|---|---|---|
| exact, tube-restricted | 2.295 | - | - |
| binless WHAM, no exchange | 2.301 | **5** of 48000 | **43%** |
| binless WHAM, with HREX | 2.241 | **1361** of 48000 | 1.9% |

The barriers differ by 0.06, inside statistical error - so HREX has not shifted the free
energy, which is the thing to check. If it shifts your PMF by more than the noise,
something is wrong with the implementation, not the physics.

**But look at the other two columns, because they are the real result here.** The
exchange-free run has one frame carrying 43% of the total probability and an effective
sample size of *five*: its 2.301 happens to be close to the truth, and that is luck, not
evidence. HREX raises the effective sample size by a factor of ~270 and leaves no frame
above 2%.

That is the concrete benefit on this system. Because configurations keep circulating, no
window wanders out of the tube covered by the others and sits there generating a
pathological weight. It is also what produces the spike in figure 5A: that narrow dive in
the no-exchange curve *is* the 43% sample. See
[Always check the weight health](#always-check-the-weight-health).

### Using the HREX data for kinetics (DHAM / MSM)

An accepted exchange teleports a configuration between windows, so a frame pair that
straddles a swap did not evolve from one to the other. This does **not** stop you using
HREX data for rates. The exchange is a valid Monte Carlo move on the extended ensemble
and the dynamics inside each segment is untouched, so you simply refuse to count the
pairs that cross a swap.

`hrex.transition_counts_continuous` does exactly that. Pass it the `ExchangeStats` from
the run and it drops a pair only where **that window actually swapped**, which is the
exact criterion:

```
pairs kept      45102
pairs dropped    2882   (6.0%)
total flux: naive 90992, within-segment 85099
```

So the price is a few per cent of counts, not the run. Without the statistics the
function falls back to dropping every pair that crosses a segment boundary, which is
safe but discards roughly `stride / exchange_every` of the data (about 20% here) whether
or not a swap was accepted - so pass the statistics.

Free energies need no such care at all: detailed balance holds and every window keeps
exactly the equilibrium distribution it had before.

One trap worth naming: the naive flux histogram does **not** look wrong. On this system
the totals are 90992 against 85099, and the per-boundary minima are similar, because with
well-overlapping windows a swap displaces a configuration about as far as ordinary
diffusion does over one save stride. A histogram that looks fine is not evidence that
the counting was done correctly - use the segment-aware function regardless.

### What HREX can hide

Exchange can mask a slow coordinate orthogonal to the string. A window that would have
sat stuck in one conformation instead has that configuration swapped away and replaced by
a neighbour's, so the sampling looks converged while the slow coordinate never actually
relaxed in place. Acceptance rates and replica traversals will both look healthy.

This is not really a defect of HREX. It is the string method's standing assumption
surfacing: **every important coordinate should be in the string definition.** A degree of
freedom that is slow and matters belongs in the CV set, where the string can resolve it
and the restraints can hold it, not left orthogonal to the path where neither the
reparametrisation nor the reweighting can see it.

So the diagnostics to trust here are the ones on the CVs themselves - overlap, flux,
weight health - not the exchange statistics. Good mixing between windows says nothing
about whether the coordinates you left out have equilibrated.

### Design guide

`hrex.predicted_acceptance` gives `exp(-(D/sigma)^2 / 2)` with `sigma = sqrt(kT/k)`.
The demo tests it by thinning the converged string by 1x, 2x, 3x and 4x, which scales the
neighbour spacing while changing nothing else:

| thinning | D / sigma | measured acceptance | guide |
|---|---|---|---|
| 1x | 1.47 | **0.316** | 0.339 |
| 2x | 2.92 | **0.031** | 0.014 |
| 3x | 4.30 | **0.004** | 0.000 |
| 4x | 5.77 | **0.000** | 0.000 |

Acceptance falls by two orders of magnitude between 1.5 and 4.3 sigma and is gone by 5.8;
figure 4A plots this against the guide.

This is standard practice rather than a surprise: force constants and window spacings are
normally tuned together so that neighbours stay close enough to exchange. The point of
having the number is that you can check it cheaply before committing - if `D/sigma` is
five or six, no amount of attempting will help, and the fix is more images or a softer
restraint.

![hrex](fig4_hrex.png)

![hrex pmf](fig5_hrex_pmf.png)

---

## Files

| file | contents |
|---|---|
| `potential3d.py` | the 3-D potential, analytic gradient, exact reference free energy |
| `labstring.py` | the lab reparametrisation, unchanged, plus an arc-spacing check |
| `sampler.py` | restrained overdamped Langevin; swap this for your MD engine |
| `fts.py` | the string iteration driver and its per-iteration record |
| `reweight.py` | binless WHAM (two solvers), projections, mean force |
| `diagnostics.py` | overlap and transition-flux diagnostics |
| `plumed_bridge.py` | write per-image `plumed.dat`, read COLVARs back - the MD-engine bridge |
| `hrex.py` | Hamiltonian replica exchange: the Metropolis move, the alternating sweep, diagnostics |
| `run_demo.py` | part 1 end to end: converge, produce, estimate, check, plot |
| `run_demo_hrex.py` | part 2: the same string with HREX, and what it does and does not change |
| `test_demo.py` | `pytest` - gradient, reparametrisation, and WHAM vs exact (11 tests) |
| `test_hrex.py` | `pytest` - the sweep, the move, and that HREX preserves the PMF (9 tests) |

## Adapting it to a real system

Replace `sampler.propagate` with a call to your MD engine, returning CV timeseries of
shape `(n_images, n_frames, n_cv)`. Everything else is dimension-agnostic and takes the
CVs as given - `labstring`, `reweight` and `diagnostics` never look at Cartesian
coordinates.

The two things to carry over are the protocol (continue each image from its own state;
reparametrise only when every image is done) and the habit of checking overlap and flux
before believing a barrier.

## PLUMED

The demo deliberately does not need PLUMED - a toy potential and 40 lines of Langevin
make every test fast. But `plumed_bridge.py` is the bridge for when you move to a real
system:

```bash
python plumed_bridge.py        # writes an example plumed.dat per image
```

```
restraint: RESTRAINT ...
   ARG=d1,d2,d3
   AT=0.270000,1.180000,1.130000
   KAPPA=2092.000000,2092.000000,2092.000000
...
PRINT ARG=d1,d2,d3 FILE=COLVAR STRIDE=100
```

The iteration becomes:

```python
for iteration in range(n_iterations):
    write_plumed_inputs(outdir, cv_defs, centres, kappa, iteration)
    ...   launch n_images MD runs on your queue ...
    traj = read_colvars(outdir, n_cv, discard=n_equil)
    centres, info = labstring.reparametrize(traj)
```

PLUMED handles the CVs and the restraints. It does **not** do the reparametrisation or
the binless reweighting - those stay in this package, unchanged, because they only ever
see the CV timeseries.

`read_colvars` deliberately raises if the images came back with different frame counts.
That almost always means a run died, and silently truncating to the shortest would bias
the string without telling you.

One unit warning: PLUMED defaults to kJ/mol and nm. The single commonest way to get a
wrong barrier in this kind of calculation is a `KAPPA` in the wrong units for the CV it
is attached to. Decide once, write it in the header, and check `sqrt(kT/k)` against the
window widths you actually observe.

---

## Origin and references

The string-method implementation here descends directly from the one used in:

> Rosta, E.; Nowotny, M.; Yang, W.; Hummer, G.
> **Catalytic mechanism of RNA backbone cleavage by ribonuclease H from quantum
> mechanics/molecular mechanics simulations.**
> *Journal of the American Chemical Society* **2011**, *133* (23), 8934-8941.
> DOI: [10.1021/ja200173a](https://doi.org/10.1021/ja200173a) ·
> PMID: [21539371](https://pubmed.ncbi.nlm.nih.gov/21539371/) · PMCID: PMC3110985

That study combined Hamiltonian replica exchange with a finite-temperature string method
to obtain the free-energy surface for RNA backbone cleavage by ribonuclease H. The
reparametrisation step in `labstring.py` is that same algorithm, copied operation for
operation and unchanged; `test_demo.py` checks it against the behaviour documented in
the module docstring.

The reweighting side of the package implements the binless WHAM / MBAR equations:

> Shirts, M. R.; Chodera, J. D. **Statistically optimal analysis of samples from multiple
> equilibrium states.** *Journal of Chemical Physics* **2008**, *129*, 124105.
> DOI: [10.1063/1.2978177](https://doi.org/10.1063/1.2978177)

Related estimator work from the same group, for anyone comparing transition-based
methods against the histogram and mean-force routes used here:

> Rosta, E.; Hummer, G. **Free energies from dynamic weighted histogram analysis using
> unbiased Markov state model.** *Journal of Chemical Theory and Computation* **2015**,
> *11* (1), 276-285. PMID: [26574225](https://pubmed.ncbi.nlm.nih.gov/26574225/)

> Stelzl, L. S.; Kells, A.; Rosta, E.; Hummer, G. **Dynamic histogram analysis to
> determine free energies and rates from biased simulations.** *Journal of Chemical
> Theory and Computation* **2017**. PMID:
> [29059525](https://pubmed.ncbi.nlm.nih.gov/29059525/)
