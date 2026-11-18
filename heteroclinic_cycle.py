"""Heteroclinic cycle of a May-Leonard field: Figure 2.1(b) of the paper.

Uses the May-Leonard field

    g_i = x_i (1 - x_i - beta x_{i+1} - alpha x_{i+2}),   alpha = 0.8, beta = 1.3,

whose saddles e_1, e_2, e_3 are joined by the cycle e_1 -> e_2 -> e_3 -> e_1
in the boundary of X = R^3_{>=0}.  Writes the tables read by
tex/heteroclinic-cycle.tex, all in results/ and with columns x1 x2 x3:

  hc-gamma-i.dat   the connection Gamma_i^+ from e_i to e_{i+1}
  hc-minus.dat     the branch of W^u(e_1) leaving X
  hc-stable-k.dat  four orbits in the face x_3 = 0, which is the part of
                   W^s(e_2) in X, converging to e_2
  hc-orbit.dat     an orbit of g from the interior of X, which visits the
                   saddles in turn and lingers longer at each pass

then builds figures/heteroclinic-cycle.pdf with latexmk.  With --replot it
only rebuilds the PDF from the cached tables, in a few seconds.

Usage (from the repository root):

  python3 heteroclinic_cycle.py            compute the tables, build the PDF
  python3 heteroclinic_cycle.py --replot   build the PDF from results/ only
"""

import argparse
import pathlib
import shutil
import subprocess

import numpy as np
from scipy.integrate import solve_ivp

AL, BE = 0.8, 1.3
X0 = [0.45, 0.3, 0.35]
HERE = pathlib.Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
FIG_DIR = HERE / "figures"
TEX_DIR = HERE / "tex"


def field(t, x):
    x1, x2, x3 = x
    return np.array([x1 * (1 - x1 - BE * x2 - AL * x3),
                     x2 * (1 - x2 - BE * x3 - AL * x1),
                     x3 * (1 - x3 - BE * x1 - AL * x2)])


def jac(x):
    h = 1e-7
    return np.column_stack([(field(0, x + h * e) - field(0, x - h * e))
                            / (2 * h) for e in np.eye(3)])


def save(name, y):
    np.savetxt(RESULTS_DIR / name, y, header="x1 x2 x3", comments="",
               fmt="%.5g")
    print(f"wrote results/{name}")


def compute():
    RESULTS_DIR.mkdir(exist_ok=True)

    # both branches of W^u(e_i): Gamma_i^+ runs until it is within 1e-3 of
    # e_{i+1}, the other branch, kept for e_1 only, until it is at distance
    # 0.8 from e_1
    opts = dict(rtol=1e-11, atol=1e-14, max_step=0.05)
    for i in range(3):
        e = np.eye(3)[i]
        w, v = np.linalg.eig(jac(e))
        u = np.real(v[:, np.argmax(np.real(w))])
        u = u if u[(i + 1) % 3] > 0 else -u

        def near(t, x):
            return np.linalg.norm(x - np.eye(3)[(i + 1) % 3]) - 1e-3

        def far(t, x):
            return np.linalg.norm(x - e) - 0.8

        near.terminal = far.terminal = True
        s = solve_ivp(field, [0, 200], e + 1e-4 * u, events=near, **opts)
        save(f"hc-gamma-{i + 1}.dat",
             np.vstack([e, s.y.T[::4], np.eye(3)[(i + 1) % 3]]))
        if i == 0:
            s = solve_ivp(field, [0, 200], e - 1e-4 * u, events=far, **opts)
            save("hc-minus.dat", np.vstack([e, s.y.T[::4]]))
        print(f"e_{i + 1}: lambda = {np.sort(np.real(w))}")

    # orbits in W^s(e_2) cap X, from the edges of the square drawn in the
    # figure until they are within 0.02 of e_2
    def arrive(t, x):
        return np.linalg.norm(x - np.eye(3)[1]) - 0.02

    arrive.terminal = True
    for k, x0 in enumerate([(1.25, 0.25, 0), (1.25, 0.7, 0), (1.25, 1.25, 0),
                            (0.55, 1.25, 0)]):
        s = solve_ivp(field, [0, 200], x0, events=arrive, **opts)
        save(f"hc-stable-{k + 1}.dat", s.y.T[::4])

    # interior orbit, a few passes near the cycle once the transient has gone
    s = solve_ivp(field, [0, 160], X0, rtol=1e-10, atol=1e-13, max_step=0.05)
    save("hc-orbit.dat", s.y.T[s.t >= 45][::4])


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
    build_pdf("heteroclinic-cycle")


if __name__ == "__main__":
    main()
