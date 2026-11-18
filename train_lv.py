"""Trained neural fields approximating the Lotka-Volterra cycle (Section 5).

For each configuration of unstable eigenvalues,

    symmetric   lambda_u = (0.7, 0.7, 0.7)
    asymmetric  lambda_u = (0.3, 0.7, 0.7)

this script trains the block neural field

    f_theta(x) = -x + P tanh(W x + b),   N = 90 units in three blocks of 30,

on the target g_k(x) = x_k (1 - sum_j rho_kj x_j), with the C^1 loss
mean|f_theta - g|^2 + lambda_J mean|Df_theta - Dg|^2 on 8000 points of
K = [-0.05, 1.15]^3 and the penalty pushing (f_theta)_k above the margin on
the planes x_k = 0.  Training uses Adam for 3000 epochs, with a learning rate
decaying geometrically per epoch from 1e-2 to 1e-4, and the fixed seed of
lvnet.SEED.  The model and the training loop are in lvnet.py.

It then integrates g, f_theta and the lifted system ydot = -y + tanh(WP y + b)
from x(0) = (0.89, 0.04, 0.07), y(0) = pinv(P) x(0), on [0, 500], estimates
the period of f_theta on an oriented section in the late tail, measures the
discrepancy between P y(t) and the independently integrated f_theta, and
draws the four figures of Section 5.  The diagnostics of the trained fields
(errors, equilibria, plane condition, initial conditions, Floquet
multipliers) are computed from the cached weights by verify_lv.py.

Files written:

  results/lv-training-symmetric.npz    weights, training history, trajectories
  results/lv-training-asymmetric.npz   of g and f_theta, y(t) on the
                                       displayed window
  results/lv-training.json             training settings, final losses,
                                       periods, projection errors
  figures/comparison_symmetric.png     figures/comparison_asymmetric.png
  figures/populations_symmetric.png    figures/populations_asymmetric.png

Training takes about three minutes per network.  With --replot the four
figures are redrawn from the two .npz caches in a few seconds, without
training or integrating.

Usage (from the repository root):

  python3 train_lv.py            train, integrate, cache, plot
  python3 train_lv.py --replot   plot from the cached .npz files only
"""

import argparse
import json
import pathlib
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import Normalize, TwoSlopeNorm  # noqa: E402

from lvnet import (  # noqa: E402
    CONDITIONS, DISPLAY_PERIODS, DT, INTEGRATION, NETWORKS, T_END,
    TrainingConfig, estimate_period, first_return_time, g_lv, integrate,
    model_from_weights, times, train_model, x_initial,
)

HERE = pathlib.Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
FIG_DIR = HERE / "figures"
RESULTS = RESULTS_DIR / "lv-training.json"


def cache_path(key):
    return RESULTS_DIR / f"lv-training-{key}.npz"


# ---------------------------------------------------------------------------
# Plotting style

sns.set_theme(style="whitegrid")

COLORS = ["tab:blue", "tab:red", "tab:green"]

TITLE_SIZE = 30
TEXT_SIZE = 25
plt.rcParams.update({
    "font.size": TEXT_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "figure.titlesize": TITLE_SIZE,
    "axes.labelsize": TEXT_SIZE,
    "xtick.labelsize": TEXT_SIZE,
    "ytick.labelsize": TEXT_SIZE,
    "legend.fontsize": TEXT_SIZE,
    "legend.title_fontsize": TEXT_SIZE,
    "axes.titlepad": 17,
    "axes.labelpad": 17,
})


# ---------------------------------------------------------------------------
# Full computation

def compute():
    config = TrainingConfig()
    summary = {
        "training": {
            "hidden_dim": config.hidden_dim, "train_size": config.train_size,
            "batch_size": config.batch_size, "epochs": config.epochs,
            "learning_rate": [config.learning_rate,
                              config.learning_rate_final],
            "learning_rate_schedule": "geometric per epoch",
            "lambda_jacobian": config.lambda_jacobian,
            "plane_weight": config.plane_weight,
            "plane_margin": config.plane_margin,
            "plane_batch": config.plane_batch, "seed": config.seed,
        },
        "integration": {"x0": x_initial.tolist(), "T_end": T_END, "dt": DT,
                        **INTEGRATION},
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    for key, (label, eigs) in CONDITIONS.items():
        print(f"\n{label}: lambda_u={eigs}", flush=True)
        start = time.time()
        model, history = train_model(eigs, config)
        train_seconds = time.time() - start
        print(f"  trained in {train_seconds:.0f} s", flush=True)

        x0 = x_initial.copy()
        y0 = np.linalg.pinv(model.P) @ x0
        g = integrate(lambda x: g_lv(x, eigs), x0)
        ftheta = integrate(model.field, x0)
        y = integrate(model.lifted_field, y0)
        projection_error = float(np.max(
            np.linalg.norm(y @ model.P.T - ftheta, axis=1)))

        T, t0, spread, recurrence, component, level = estimate_period(
            times, ftheta)
        T_het_first, _ = first_return_time(times, g, component, level)
        window = (times >= t0) & (times <= t0 + DISPLAY_PERIODS * T)

        np.savez_compressed(
            cache_path(key),
            W=model.W, b=model.b, p=model.p, lambda_u=eigs,
            history=history, times=times, g=g, ftheta=ftheta,
            window=np.flatnonzero(window), y_window=y[window],
            period=T, time_origin=t0, section_component=component,
            section_level=level, projection_error=projection_error,
        )
        print(f"  wrote {cache_path(key).relative_to(HERE)}", flush=True)

        summary[key] = {
            "lambda_u": eigs.tolist(),
            "training_seconds": round(train_seconds, 1),
            "final_loss": {"total": float(history[-1, 0]),
                           "value": float(history[-1, 1]),
                           "jacobian": float(history[-1, 2])},
            "period": {
                "T": T,
                "heteroclinic_first_return": T_het_first,
                "time_origin": t0,
                "section_component": NETWORKS[component],
                "section_level": float(level),
                "relative_period_spread": spread,
                "relative_return_error": recurrence,
                "projection_error": projection_error,
                "display_window": [float(times[window][0]),
                                   float(times[window][-1])],
            },
        }
        print(json.dumps({k: summary[key][k] for k in
                          ("final_loss", "period")}, indent=1), flush=True)

    with open(RESULTS, "w") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    print(f"wrote {RESULTS.relative_to(HERE)}")


# ---------------------------------------------------------------------------
# Plotting, from the caches only

models, results = {}, {}


def load_caches():
    for key in CONDITIONS:
        z = np.load(cache_path(key))
        models[key] = model_from_weights(z["W"], z["b"], z["p"])
        results[key] = {
            "g": z["g"], "ftheta": z["ftheta"],
            "t_window": z["times"][z["window"]], "y_window": z["y_window"],
            "period": float(z["period"]),
            "time_origin": float(z["time_origin"]),
        }


def label_curves(ax):
    for i, (name, color) in enumerate(zip(NETWORKS, COLORS)):
        ax.text(
            0.02 + i * 0.25, 1.0, f"{name}(t)",
            color=color, transform=ax.transAxes,
            fontsize=TEXT_SIZE, fontweight="bold", va="top"
        )


def draw_traces(ax, t, x, xlim_common):
    for i, color in enumerate(COLORS):
        ax.plot(t, x[:, i], color=color, lw=2.6)
    ax.set_ylim(*xlim_common)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Time")
    ax.set_ylabel("Activity")
    label_curves(ax)


def save_figure(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / (name + ".png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(HERE)}")


def draw_vector_field_3d(ax, field, n_grid=5, arrow_length=0.10):
    values = np.linspace(0.0, 1.0, n_grid)
    X, Y, Z = np.meshgrid(values, values, values, indexing="ij")
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    vectors = np.array([field(point) for point in points])
    norms = np.linalg.norm(vectors, axis=1)
    mask = np.all(np.isfinite(vectors), axis=1) & (norms > 1e-10)
    ax.quiver(
        *points[mask].T,
        *vectors[mask].T,
        length=arrow_length,
        normalize=True,
        color="gray",
        alpha=0.65,
        linewidth=0.6,
        arrow_length_ratio=0.30
    )


def plot_comparison(key, xlim_common):
    label, eigs = CONDITIONS[key]

    fig = plt.figure(figsize=(14, 10), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])

    names = [("g", r"$g$"), ("ftheta", r"$f_\theta$")]
    fields = {"g": lambda x: g_lv(x, eigs), "ftheta": models[key].field}

    for j, (name, title) in enumerate(names):
        x = results[key][name]

        # Top row: activities in time.
        ax = fig.add_subplot(grid[0, j])
        draw_traces(ax, times, x, xlim_common)
        ax.set_xlim(0, min(140.0, T_END))
        ax.set_title(title)

        # Bottom row: vector field and 3D trajectory.
        ax3 = fig.add_subplot(grid[1, j], projection="3d")
        draw_vector_field_3d(ax3, fields[name], n_grid=5, arrow_length=0.10)
        ax3.plot(*x.T, lw=1.8, color="#334c80")
        ax3.scatter(*x[0], color="black", s=18)
        ax3.set(
            xlabel="CN", ylabel="DMN", zlabel="SN",
            xlim=xlim_common, ylim=xlim_common, zlim=xlim_common
        )
        ax3.set_box_aspect((1, 1, 1))
        ax3.view_init(elev=24, azim=38)

    fig.suptitle(f"{label} — unstable eigenvalues: {eigs}",
                 fontsize=TITLE_SIZE)
    save_figure(fig, "comparison_" + key)


def population_row(fig, grid, key, p_norm, y_norm):
    model, result = models[key], results[key]
    n = model.hidden_dim
    T = result["period"]
    t0 = result["time_origin"]
    # Window [t0, t0 + DISPLAY_PERIODS T] clipped at T_END, cached.
    t_pop = result["t_window"]
    y_win = result["y_window"]

    axp = fig.add_subplot(grid[0])

    # Middle panel: brackets on the left, heatmap on the right.
    middle = grid[1].subgridspec(1, 2, width_ratios=[1.8, 5], wspace=0.02)
    ax_labels = fig.add_subplot(middle[0])
    axy = fig.add_subplot(middle[1])

    axx = fig.add_subplot(grid[2])

    im_p = axp.imshow(
        model.P.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        norm=p_norm,
        extent=(-0.5, 2.5, -0.5, n - 0.5)
    )

    centers = [(idx[0] + idx[-1]) / 2 for idx in model.population_indices]
    axp.set_yticks(centers, NETWORKS)
    axp.set_xticks(range(3), NETWORKS, rotation=45)
    axp.set_title(r"Matrix $P^T$")
    axp.grid(False)

    half_dt = (t_pop[1] - t_pop[0]) / 2
    im_y = axy.imshow(
        y_win.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        norm=y_norm,
        extent=(t_pop[0] - half_dt, t_pop[-1] + half_dt, -0.5, n - 0.5)
    )
    axy.set_yticks([])
    axy.set_title(r"$t \mapsto y(t)$")
    axy.set_xlabel("Time")
    axy.grid(False)

    for idx in model.population_indices[:-1]:
        axy.axhline(idx[-1] + 0.5, color="white", lw=5.5)

    ax_labels.set_xlim(0, 1)
    ax_labels.set_ylim(-0.5, n - 0.5)
    ax_labels.set_axis_off()

    for name, idx in zip(NETWORKS, model.population_indices):
        bottom = idx[0] - 0.35
        top = idx[-1] + 0.35
        center = (bottom + top) / 2
        ax_labels.plot([0.98, 0.88, 0.88, 0.98], [bottom, bottom, top, top],
                       color="black", lw=1.2)
        ax_labels.text(
            0.80, center,
            name + "\n" + rf"$y_{{{idx[0]+1}}},\ldots,y_{{{idx[-1]+1}}}$",
            ha="right", va="center", fontsize=TEXT_SIZE
        )

    # Three diverse time courses from each block.
    for block, (idx, color) in enumerate(zip(model.population_indices,
                                             COLORS)):
        curves = y_win[:, idx].T
        first = int(np.argmax(np.std(curves, axis=1)))
        chosen = [first]
        while len(chosen) < 3:
            distances = np.min([np.mean((curves - curves[j])**2, axis=1)
                                for j in chosen], axis=0)
            distances[chosen] = -np.inf
            chosen.append(int(np.argmax(distances)))
        for local, style in zip(chosen, ["-", "--", ":"]):
            j = int(idx[local])
            axx.plot(t_pop, y_win[:, j], color=color, ls=style, lw=1.8,
                     label=rf"{NETWORKS[block]} $y_{{{j+1}}}$")

    axx.set_title("Selected populations")
    axx.set_ylabel(r"$y_j(t)$")
    axx.legend(fontsize=10, ncol=3, loc="upper center",
               bbox_to_anchor=(.5, -.18))

    period_time = t0 + T
    for ax in [axy, axx]:
        ax.axvline(period_time, color="black", ls="--", lw=1.8)
        ax.text(period_time, 1.001, rf"$T\approx {T:.2f}$",
                transform=ax.get_xaxis_transform(), ha="center", fontsize=14)
        ax.set_xlim(t_pop[0], t_pop[-1])
        ax.set_xlabel(r"$t$")

    fig.colorbar(im_p, ax=axp, shrink=0.8, pad=0.04)
    fig.colorbar(im_y, ax=axy, shrink=0.8, pad=0.02)


def plot_population(key, p_norm, y_norm):
    label, eigs = CONDITIONS[key]
    fig = plt.figure(figsize=(21, 7), layout="constrained")
    grid = fig.add_gridspec(1, 3, width_ratios=[1.5, 5, 3.5])
    population_row(fig, [grid[0, j] for j in range(3)], key, p_norm, y_norm)
    fig.suptitle(f"{label} — unstable eigenvalues: {eigs}",
                 fontsize=TITLE_SIZE)
    save_figure(fig, "populations_" + key)


def plot_all():
    load_caches()
    all_x = np.concatenate([r[k] for r in results.values()
                            for k in ("g", "ftheta")])
    xlim_common = (min(-0.03, float(all_x.min()) - 0.03),
                   max(1.05, float(all_x.max()) + 0.03))
    y_norm = Normalize(vmin=-1, vmax=1)
    p_bound = max(float(np.abs(m.P).max()) for m in models.values())
    p_norm = TwoSlopeNorm(vmin=-max(p_bound, 1e-12), vcenter=0,
                          vmax=max(p_bound, 1e-12))
    for key in CONDITIONS:
        plot_comparison(key, xlim_common)
    for key in CONDITIONS:
        plot_population(key, p_norm, y_norm)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replot", action="store_true",
                        help="redraw the figures from the cached .npz files")
    args = parser.parse_args()
    if not args.replot:
        compute()
    plot_all()


if __name__ == "__main__":
    main()
