"""Checks on the two trained neural fields of Section 5.

Runs on the weights cached by train_lv.py (results/lv-training-*.npz) and
never retrains.  For each configuration (symmetric and asymmetric) it
computes:

  1. the field and Jacobian errors |f_theta - g| and |Df_theta - Dg|_F on
     3000 uniform test points of K = [-0.05, 1.15]^3, and on the points of a
     uniform sample of 400000 points of K lying within distance 0.1 of the
     heteroclinic cycle Gamma of g;
  2. the equilibria of f_theta within distance 0.5 of each saddle e_i, with
     the eigenvalues of Df_theta there, next to those of Dg(e_i);
  3. the plane condition: the minimum of (f_theta)_k on a 401 x 401 grid of
     each face K cap {x_k = 0}, and the fraction of grid points where
     (f_theta)_k <= 0;
  4. a sweep of 30 initial conditions: six fixed ones and 24 uniform in
     [0.02, 1]^3 drawn with np.random.default_rng(7).  Each is integrated on
     [0, 500] like the trajectory of the figures; the script records its
     period (same section estimate as train_lv.py) and the largest distance
     of its tail t >= 480 to the reference periodic orbit, which is the
     orbit through x_theta(500) for x(0) = (0.89, 0.04, 0.07), sampled
     densely over one period;
  5. the Floquet multipliers of that periodic orbit, from the variational
     equation integrated over one period.

Everything is written to results/lv-verify.json.  The run takes a minute or
two and has no figure, hence no --replot flag.

Usage (from the repository root):

  python3 verify_lv.py
"""

import json
import pathlib

import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree

from lvnet import (
    CONDITIONS, K_MAX, K_MIN, NETWORKS, SEED, T_END, TrainingConfig,
    complex_list, empirical_c1_diagnostics, equilibria_report,
    estimate_period, heteroclinic_cycle, integrate, load_model,
    near_gamma_diagnostics, times, x_initial,
)

HERE = pathlib.Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
OUTPUT = RESULTS_DIR / "lv-verify.json"

PLANE_GRID = 401
TAIL_START = 480.0
REFERENCE_DT = 0.001


def initial_conditions():
    fixed = [(0.89, 0.04, 0.07), (0.95, 0.06, 0.04), (0.3, 0.3, 0.3),
             (0.1, 0.1, 0.1), (1.0, 1.0, 1.0), (0.02, 0.5, 0.9)]
    rng = np.random.default_rng(7)
    return [np.array(x) for x in fixed] + list(
        rng.uniform(0.02, 1.0, size=(24, 3)))


def plane_condition(model):
    """Minimum and sign of (f_theta)_k on a grid of K cap {x_k = 0}."""
    s = np.linspace(K_MIN, K_MAX, PLANE_GRID)
    u, v = np.meshgrid(s, s, indexing="ij")
    rows = []
    for k in range(3):
        others = [j for j in range(3) if j != k]
        x = np.zeros((PLANE_GRID**2, 3))
        x[:, others[0]] = u.ravel()
        x[:, others[1]] = v.ravel()
        fk = model.field_batch(x)[:, k]
        rows.append({
            "face": f"x_{k + 1} = 0 ({NETWORKS[k]})",
            "grid": [PLANE_GRID, PLANE_GRID],
            "min": float(fk.min()),
            "argmin": x[np.argmin(fk)].tolist(),
            "fraction_nonpositive": float(np.mean(fk <= 0.0)),
        })
    return rows


def polyline_distance(points, vertices):
    """Distance of each point to the polyline through the vertices."""
    tree = cKDTree(vertices)
    _, nearest = tree.query(points)
    best = np.full(len(points), np.inf)
    for shift in (-1, 0):
        i = np.clip(nearest + shift, 0, len(vertices) - 2)
        p, q = vertices[i], vertices[i + 1]
        d = q - p
        s = np.clip(np.sum((points - p) * d, axis=1)
                    / np.maximum(np.sum(d * d, axis=1), 1e-300), 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(points - p - s[:, None] * d,
                                               axis=1))
    return best


def period_or_none(x):
    try:
        T, _, spread, recurrence, component, _ = estimate_period(times, x)
    except RuntimeError as err:
        return {"period": None, "reason": str(err)}
    return {"period": T, "relative_period_spread": spread,
            "relative_return_error": recurrence,
            "section_component": NETWORKS[component]}


def sweep(model):
    """Period and distance to the reference orbit for 30 initial data."""
    ics = initial_conditions()
    runs = [integrate(model.field, x0) for x0 in ics]

    T_ref = estimate_period(times, runs[0])[0]
    p0 = runs[0][-1]
    n = int(np.ceil(1.01 * T_ref / REFERENCE_DT))
    reference = integrate(model.field, p0,
                          t_eval=np.linspace(0.0, n * REFERENCE_DT, n + 1))

    tail = times >= TAIL_START
    rows = []
    for x0, x in zip(ics, runs):
        row = {"x0": [float(v) for v in x0]}
        row.update(period_or_none(x))
        row["tail_distance_to_reference"] = float(
            polyline_distance(x[tail], reference).max())
        rows.append(row)

    periods = np.array([r["period"] for r in rows if r["period"] is not None])
    summary = {
        "count": len(rows),
        "periodic": int(len(periods)),
        "reference_period": T_ref,
        "reference_point": p0.tolist(),
        "tail_start": TAIL_START,
        "reference_dt": REFERENCE_DT,
        "period_min": float(periods.min()),
        "period_max": float(periods.max()),
        "relative_period_range": float(np.ptp(periods) / T_ref),
        "max_tail_distance": max(r["tail_distance_to_reference"]
                                 for r in rows),
        "runs": rows,
    }
    return summary, T_ref, p0


def floquet(model, T, p0):
    """Monodromy matrix of the orbit through p0 over one period T."""
    def variational(_, z):
        x, M = z[:3], z[3:].reshape(3, 3)
        return np.concatenate([model.field(x),
                               (model.jacobian(x) @ M).ravel()])

    sol = solve_ivp(variational, (0.0, T),
                    np.concatenate([p0, np.eye(3).ravel()]),
                    method="DOP853", rtol=1e-11, atol=1e-13, max_step=0.05)
    if not sol.success:
        raise RuntimeError(sol.message)
    monodromy = sol.y[3:, -1].reshape(3, 3)
    mu = np.linalg.eigvals(monodromy)
    mu = mu[np.argsort(-np.abs(mu))]
    return {
        "period": T,
        "point": p0.tolist(),
        "closure_error": float(np.linalg.norm(sol.y[:3, -1] - p0)),
        "multipliers": [[float(m.real), float(m.imag)] for m in mu],
        "moduli": [float(abs(m)) for m in mu],
        "integration": {"method": "DOP853", "rtol": 1e-11, "atol": 1e-13,
                        "max_step": 0.05},
    }


def main():
    config = TrainingConfig()
    out = {}
    for key, (label, eigs) in CONDITIONS.items():
        print(f"{label}: lambda_u={eigs}", flush=True)
        model, stored = load_model(RESULTS_DIR / f"lv-training-{key}.npz")
        assert np.allclose(stored, eigs)
        tree = cKDTree(heteroclinic_cycle(eigs))

        report = {
            "lambda_u": eigs.tolist(),
            "test_errors_K": dict(
                empirical_c1_diagnostics(model, eigs, config.test_size,
                                         SEED + 100),
                points=config.test_size, seed=SEED + 100),
            "errors_near_Gamma": near_gamma_diagnostics(model, eigs,
                                                        tree=tree),
            "equilibria": equilibria_report(model, eigs),
            "plane_condition": plane_condition(model),
        }
        print("  errors, equilibria and plane condition done", flush=True)
        report["initial_conditions"], T, p0 = sweep(model)
        print("  sweep of initial conditions done", flush=True)
        report["floquet"] = floquet(model, T, p0)

        for name in ("test_errors_K", "errors_near_Gamma"):
            print(f"  {name}: " + ", ".join(
                f"{k} {v:.3g}" for k, v in report[name].items()
                if isinstance(v, float)))
        for row in report["equilibria"]:
            for eq in row["f_theta_equilibria"]:
                print(f"  equilibrium near {row['near']}: eigenvalues "
                      + " ".join(f"{v[0]:+.4f}" for v in eq["eigenvalues"]))
        for row in report["plane_condition"]:
            print(f"  plane {row['face']}: min {row['min']:.4f}, "
                  f"fraction <= 0: {row['fraction_nonpositive']:.4f}")
        s = report["initial_conditions"]
        print(f"  {s['periodic']}/{s['count']} initial conditions periodic, "
              f"periods in [{s['period_min']:.6f}, {s['period_max']:.6f}], "
              f"max tail distance {s['max_tail_distance']:.2e}")
        f = report["floquet"]
        print("  Floquet multipliers: " + ", ".join(
            f"{m[0]:.6g}{m[1]:+.2g}i" for m in f["multipliers"])
            + f" (closure {f['closure_error']:.1e})")
        out[key] = report

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(OUTPUT, "w") as fh:
        json.dump({"T_end": T_END, "x0": x_initial.tolist(), **out}, fh,
                  indent=2)
        fh.write("\n")
    print(f"wrote {OUTPUT.relative_to(HERE)}")


if __name__ == "__main__":
    main()
