"""Cyclic inhibition: Figure S1.1 of the Supplementary Material and c*.

Integrates the three-population neural field

    xdot_i = -x_i + sigma(-c x_{i-1} + b0),   i = 1,2,3 mod 3,

with sigma = tanh and b0 = 1/2, and writes the tables read by
tex/ring-cycle.tex:

  results/ring-timeseries.dat   t x1 x2 x3       the attracting periodic orbit
                                                 at c = 5
  results/ring-bifurcation.dat  c xmin xmax xeq  the range of x_1 on the
                                                 attractor and the
                                                 equilibrium branch x*(c)

It also writes results/ring-threshold.tex, a one-macro file holding the
threshold c* at which psi(c) = c sigma'(sigma^{-1}(x*)) crosses 2, so that the
figure and the text cannot drift apart, and results/ring-results.json with
c* and the numbers quoted in the text: the peak-to-peak range and the period
of the orbit at c = 2.1, 5, 10, 40, the spread of the extreme values of x_1
over 22 initial conditions at c = 2.1, 5, 10, and the distance to the
equilibrium reached from far-away initial conditions at c = 1.5, 2.

Above c* the range of x_1 is measured by integration, sweeping c upwards and
starting each run from the previous periodic orbit so that the transients
stay short.  At and below c* the table is not integrated: xmin and xmax are
set to the equilibrium x*(c), which is locally asymptotically stable below c*
by the theorem.  That it is also the attractor of far-away initial conditions
is checked separately, by integration at c = 1.5 and c = 2 only, and the
result goes to ring-results.json without affecting the table.

Finally it builds figures/ring-cycle.pdf with latexmk.  The full run takes
about fifteen minutes, most of it in the checks.  With --replot the script
only rebuilds the PDF from the cached tables, in a few seconds.

Usage (from the repository root):

  python3 ring_cycle.py            compute the tables, then build the PDF
  python3 ring_cycle.py --replot   build the PDF from results/ only
"""

import argparse
import json
import pathlib
import shutil
import subprocess

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

B0 = 0.5
C_SHOW = 5.0
RTOL, ATOL = 1e-10, 1e-12
HERE = pathlib.Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
FIG_DIR = HERE / "figures"
TEX_DIR = HERE / "tex"


def rhs(t, x, c):
    """Right-hand side.  np.roll(x, 1)[i] equals x_{i-1}, indices mod 3."""
    return -x + np.tanh(-c * np.roll(x, 1) + B0)


def equilibrium(c):
    """The unique solution of x = tanh(-c x + b0)."""
    return brentq(lambda s: np.tanh(-c * s + B0) - s, -1.0, 1.0,
                  xtol=1e-15, rtol=8.9e-16)


def psi(c):
    """psi(c) = c sigma'(theta*), the quantity that crosses 2 at c*."""
    return c * (1.0 - equilibrium(c) ** 2)


def flow(x0, c, horizon):
    """Endpoint of the orbit through x0 after time horizon."""
    return solve_ivp(rhs, (0.0, horizon), np.asarray(x0), args=(c,),
                     rtol=RTOL, atol=ATOL).y[:, -1]


def sample(x0, c, window, n):
    """Sample the orbit through x0 on [0, window]."""
    t = np.linspace(0.0, window, n)
    return t, solve_ivp(rhs, (0.0, window), np.asarray(x0), args=(c,),
                        rtol=RTOL, atol=ATOL, t_eval=t).y


def write(name, header, rows):
    with open(RESULTS_DIR / name, "w") as fh:
        fh.write(header + "\n")
        for row in rows:
            fh.write(" ".join(f"{v:.6f}" for v in row) + "\n")
    print(f"wrote results/{name} ({len(rows)} rows)")


def period(t, x):
    """Median time between upward crossings of the mean of x."""
    level = x.mean()
    ids = np.flatnonzero((x[:-1] < level) & (x[1:] >= level))
    alpha = (level - x[ids]) / (x[ids + 1] - x[ids])
    crossings = t[ids] + alpha * (t[ids + 1] - t[ids])
    return float(np.median(np.diff(crossings)))


def spread_above_threshold(c, n_random=15):
    """Check that off-diagonal initial conditions reach one common orbit.

    Above c* the periodic orbit appears to be the attractor for every
    initial condition off the diagonal, not only for the one used to draw
    the figure.
    """
    xs = equilibrium(c)
    rng = np.random.default_rng(1)
    ics = list(rng.uniform(-0.95, 0.95, size=(n_random, 3)))
    ics.append(np.array([xs, xs, xs]) + 1e-6 * np.array([1.0, -1.0, 0.0]))
    ics += [e * 0.95 for e in np.eye(3)] + [-e * 0.95 for e in np.eye(3)]
    ext = []
    for x0 in ics:
        _, z = sample(flow(x0, c, 3000.0), c, 60.0, 30001)
        ext.append((z[0].min(), z[0].max()))
    ext = np.array(ext)
    return len(ics), max(np.ptp(ext[:, 0]), np.ptp(ext[:, 1]))


def check_below_threshold(c):
    """Confirm that far-away initial conditions still reach the equilibrium."""
    xs = equilibrium(c)
    worst = 0.0
    rng = np.random.default_rng(0)
    for x0 in np.vstack([rng.uniform(-0.95, 0.95, size=(12, 3)),
                         np.eye(3) * 0.9, -np.eye(3) * 0.9]):
        worst = max(worst, abs(flow(x0, c, 4000.0) - xs).max())
    return worst


def compute():
    RESULTS_DIR.mkdir(exist_ok=True)
    c_star = brentq(lambda c: psi(c) - 2.0, 0.5, 10.0,
                    xtol=1e-14, rtol=8.9e-16)
    print(f"c* = {c_star:.10f}")

    # ---- panel (a): the periodic orbit at c = C_SHOW ------------------
    t, x = sample(flow((0.9, -0.9, 0.3), C_SHOW, 800.0), C_SHOW, 12.0, 1601)
    write("ring-timeseries.dat", "t x1 x2 x3",
          np.column_stack([t, x[0], x[1], x[2]]))

    # ---- panel (b): the attractor as c varies -------------------------
    below = np.linspace(1.0, c_star, 40)
    above = np.linspace(c_star, 10.0, 110)[1:]

    rows = [(c, equilibrium(c), equilibrium(c), equilibrium(c)) for c in below]

    y = flow((0.9, -0.9, 0.3), above[0], 3000.0)   # settle onto the cycle once
    for c in above:
        y = flow(y, c, 400.0)                      # continuation in c
        _, z = sample(y, c, 40.0, 20001)
        y = z[:, -1]
        rows.append((c, z[0].min(), z[0].max(), equilibrium(c)))
    write("ring-bifurcation.dat", "c xmin xmax xeq", rows)

    with open(RESULTS_DIR / "ring-threshold.tex", "w") as fh:
        fh.write(f"\\def\\ringcstar{{{c_star:.4f}}}%\n")
    print("wrote results/ring-threshold.tex")

    # ---- reported numbers and checks ---------------------------------
    summary = {"sigma": "tanh", "b0": B0, "c_star": c_star,
               "orbit": [], "below_threshold": [], "spread": []}
    for c in (2.1, 5.0, 10.0, 40.0):
        t, z = sample(flow((0.9, -0.9, 0.3), c, 1500.0), c, 60.0, 30001)
        row = {"c": c, "peak_to_peak": float(z[0].max() - z[0].min()),
               "period": period(t, z[0])}
        summary["orbit"].append(row)
        print(f"  c = {c:>5}   peak-to-peak = {row['peak_to_peak']:.5f}"
              f"   period = {row['period']:.5f}")
    for c in (1.5, 2.0):
        worst = check_below_threshold(c)
        summary["below_threshold"].append(
            {"c": c, "initial_conditions": 18,
             "worst_distance_to_equilibrium": float(worst)})
        print(f"  c = {c:>5}   worst distance to equilibrium = {worst:.2e}")
    for c in (2.1, 5.0, 10.0):
        n, spread = spread_above_threshold(c)
        summary["spread"].append(
            {"c": c, "initial_conditions": n,
             "extreme_value_spread": float(spread)})
        print(f"  c = {c:>5}   {n} initial conditions, spread of the "
              f"extreme values of x1 = {spread:.2e}")

    with open(RESULTS_DIR / "ring-results.json", "w") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    print("wrote results/ring-results.json")


def build_pdf(name):
    """Compile tex/<name>.tex with latexmk and move the PDF to figures/."""
    subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode",
                    "-halt-on-error", f"{name}.tex"],
                   cwd=TEX_DIR, check=True, stdout=subprocess.DEVNULL)
    FIG_DIR.mkdir(exist_ok=True)
    shutil.move(TEX_DIR / f"{name}.pdf", FIG_DIR / f"{name}.pdf")
    print(f"wrote figures/{name}.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replot", action="store_true",
                        help="only rebuild the PDF from the cached tables")
    args = parser.parse_args()
    if not args.replot:
        compute()
    build_pdf("ring-cycle")


if __name__ == "__main__":
    main()
