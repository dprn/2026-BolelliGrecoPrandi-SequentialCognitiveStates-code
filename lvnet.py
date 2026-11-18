"""Shared code for Section 5: the Lotka-Volterra target and its neural fields.

The target is the Lotka-Volterra field

    g_k(x) = x_k (1 - sum_j rho_kj x_j),   k = 1, 2, 3,

with rho built from the unstable eigenvalues lambda_u, the stable eigenvalue
lambda_s = 0.95 and a_j = 1, and the approximating field is the block neural
field

    f_theta(x) = -x + P tanh(W x + b),   P = M diag(p),

with N = 90 hidden units in three blocks of 30.  This module holds the target,
the network, the training loop (C^1 loss plus a penalty on the coordinate
planes, Adam), the integrator, the period estimate and the diagnostics.  It
does no plotting and writes no file: train_lv.py and verify_lv.py do that.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy.spatial import cKDTree

SEED = 34554354353

NETWORKS = ["CN", "DMN", "SN"]

a = np.array([1.0, 1.0, 1.0])
lambda_s = 0.95
lambda_equal = np.array([0.70, 0.70, 0.70])
lambda_slow_cn = np.array([0.30, 0.70, 0.70])

eta_K = 0.05
A = 1.10
K_MIN, K_MAX = -eta_K, A + eta_K

x_initial = np.array([0.89, 0.04, 0.07])
T_END = 500.0
DT = 0.05
PERIOD_TOLERANCE = 0.02
DISPLAY_PERIODS = 5.0
CONDITIONS = {
    "symmetric": ("Symmetric", lambda_equal),
    "asymmetric": ("Asymmetric", lambda_slow_cn),
}

times = np.linspace(0, T_END, int(round(T_END / DT)) + 1)

# Integration settings used for every trajectory of g, f_theta and y.
INTEGRATION = {"method": "DOP853", "rtol": 1e-9, "atol": 1e-11,
               "max_step": 0.1}

# Radius of the neighborhood of Gamma used for the restricted errors.
GAMMA_RADIUS = 0.1


# ---------------------------------------------------------------------------
# Target field

def rho_matrix(lambda_u):
    """Matrix rho in the paper; indices are interpreted modulo 3."""
    rho = np.empty((3, 3), dtype=float)
    for k in range(3):
        for j in range(3):
            if k == j:
                rho[k, j] = 1.0 / a[j]
            elif k == (j + 1) % 3:
                rho[k, j] = (1.0 - lambda_u[j]) / a[j]
            else:
                rho[k, j] = (1.0 + lambda_s) / a[j]
    return rho


def g_lv(x, lambda_u):
    x = np.asarray(x)
    rho = rho_matrix(lambda_u)
    return x * (1.0 - rho @ x)


def jac_lv(x, lambda_u):
    x = np.asarray(x)
    rho = rho_matrix(lambda_u)
    return np.diag(1.0 - rho @ x) - np.diag(x) @ rho


def target_batch(x, lambda_u):
    rho = rho_matrix(lambda_u)
    return x * (1.0 - x @ rho.T)


def target_jacobian_batch(x, lambda_u):
    rho = rho_matrix(lambda_u)
    v = 1.0 - x @ rho.T
    return (
        np.eye(3)[None, :, :] * v[:, :, None]
        - x[:, :, None] * rho[None, :, :]
    )


# ---------------------------------------------------------------------------
# Model and training

@dataclass
class TrainingConfig:
    hidden_dim: int = 90
    train_size: int = 8000
    test_size: int = 3000
    batch_size: int = 512
    epochs: int = 3000
    learning_rate: float = 1e-2        # at the first epoch
    learning_rate_final: float = 1e-4  # at the last epoch
    lambda_jacobian: float = 1.0
    weight_decay: float = 0.0
    plane_weight: float = 100.0
    plane_margin: float = 0.005
    plane_batch: int = 192
    seed: int = SEED

    def lr(self, epoch):
        """Geometric decay from learning_rate to learning_rate_final."""
        ratio = self.learning_rate_final / self.learning_rate
        return self.learning_rate * ratio ** (epoch / (self.epochs - 1))


class BlockNeuralField:
    def __init__(self, hidden_dim=30, seed=SEED):
        if hidden_dim % 3 != 0:
            raise ValueError("hidden_dim must be divisible by 3")
        self.hidden_dim = hidden_dim
        self.population_size = hidden_dim // 3
        self.population_indices = [
            np.arange(i * self.population_size, (i + 1) * self.population_size)
            for i in range(3)
        ]
        self.M = np.zeros((3, hidden_dim))
        for i, idx in enumerate(self.population_indices):
            self.M[i, idx] = 1.0

        local_rng = np.random.default_rng(seed)
        self.W = local_rng.normal(0.0, 0.70, size=(hidden_dim, 3))
        self.b = local_rng.normal(0.0, 0.10, size=hidden_dim)
        self.p = local_rng.normal(0.0, 0.50, size=hidden_dim)

    @staticmethod
    def activation(z):
        return np.tanh(z)

    @property
    def P(self):
        return self.M @ np.diag(self.p)

    @property
    def WP(self):
        return self.W @ self.P

    def hidden(self, x):
        x = np.asarray(x)
        return self.activation(x @ self.W.T + self.b)

    def field_batch(self, x):
        x = np.asarray(x)
        s = self.hidden(x)
        return -x + (self.p * s) @ self.M.T

    def field(self, x):
        return self.field_batch(np.asarray(x)[None, :])[0]

    def jacobian_batch(self, x):
        x = np.asarray(x)
        s = self.hidden(x)
        sp = 1.0 - s**2
        nonlinear = np.einsum(
            "ar,br,rj->baj", self.M, sp, self.p[:, None] * self.W
        )
        return -np.eye(3)[None, :, :] + nonlinear

    def jacobian(self, x):
        return self.jacobian_batch(np.asarray(x)[None, :])[0]

    def lifted_field(self, y):
        y = np.asarray(y)
        return -y + self.activation(self.WP @ y + self.b)


def c1_loss_and_gradients(model, x, lambda_u, lambda_j, weight_decay):
    """Exact gradient of the value-plus-Jacobian loss."""
    z = x @ model.W.T + model.b
    s = model.activation(z)
    sp = 1.0 - s**2
    spp = -2.0 * s * sp

    # Value term
    f = -x + (model.p * s) @ model.M.T
    target = target_batch(x, lambda_u)
    residual = f - target
    value_loss = np.mean(residual**2)

    d_f = 2.0 * residual / residual.size
    d_h = d_f @ model.M
    grad_p = np.sum(d_h * s, axis=0)
    d_z = d_h * model.p * sp
    grad_W = d_z.T @ x
    grad_b = np.sum(d_z, axis=0)

    # Jacobian term
    jac = -np.eye(3)[None, :, :] + np.einsum(
        "ar,br,rj->baj", model.M, sp, model.p[:, None] * model.W
    )
    target_jac = target_jacobian_batch(x, lambda_u)
    jac_residual = jac - target_jac
    jacobian_loss = np.mean(jac_residual**2)

    d_jac = 2.0 * lambda_j * jac_residual / jac_residual.size
    # Sum over the receiving population index a.
    transported = np.einsum("baj,ar->brj", d_jac, model.M)

    grad_p += np.einsum("brj,br,rj->r", transported, sp, model.W)
    grad_W += np.einsum("brj,br,r->rj", transported, sp, model.p)

    d_sp = np.einsum("brj,r,rj->br", transported, model.p, model.W)
    d_z_jac = d_sp * spp
    grad_W += d_z_jac.T @ x
    grad_b += np.sum(d_z_jac, axis=0)

    # Small regularization
    reg_loss = weight_decay * (
        np.mean(model.W**2) + np.mean(model.b**2) + np.mean(model.p**2)
    )
    grad_W += 2.0 * weight_decay * model.W / model.W.size
    grad_b += 2.0 * weight_decay * model.b / model.b.size
    grad_p += 2.0 * weight_decay * model.p / model.p.size

    total = value_loss + lambda_j * jacobian_loss + reg_loss
    return total, (grad_W, grad_b, grad_p), value_loss, jacobian_loss


def plane_loss_and_gradients(model, x, k, margin, weight):
    s = model.hidden(x)
    mask = model.M[k]
    normal = np.sum(s * model.p * mask, axis=1)  # x_k = 0
    deficit = np.maximum(margin - normal, 0.0)
    d = (-2 * weight / len(x)) * deficit
    dh = d[:, None] * mask
    dz = dh * model.p * (1 - s * s)
    return weight * np.mean(deficit**2), (dz.T @ x, dz.sum(axis=0),
                                          (dh * s).sum(axis=0))


def train_model(lambda_u, config=TrainingConfig(), verbose=True):
    local_rng = np.random.default_rng(config.seed)
    model = BlockNeuralField(config.hidden_dim, config.seed)
    x_train = local_rng.uniform(K_MIN, K_MAX, size=(config.train_size, 3))

    params = [model.W, model.b, model.p]
    first_moment = [np.zeros_like(q) for q in params]
    second_moment = [np.zeros_like(q) for q in params]
    beta1, beta2 = 0.9, 0.999
    step = 0
    history = []

    for epoch in range(config.epochs):
        lr = config.lr(epoch)
        order = local_rng.permutation(config.train_size)
        for start in range(0, config.train_size, config.batch_size):
            step += 1
            idx = order[start:start + config.batch_size]
            loss, gradients, value_loss, jac_loss = c1_loss_and_gradients(
                model, x_train[idx], lambda_u,
                config.lambda_jacobian, config.weight_decay
            )

            xb = local_rng.uniform(K_MIN, K_MAX, size=(config.plane_batch, 3))
            kb = np.arange(config.plane_batch) % 3
            xb[np.arange(config.plane_batch), kb] = 0.0
            plane_loss, plane_grads = plane_loss_and_gradients(
                model, xb, kb, config.plane_margin, config.plane_weight)
            gradients = tuple(u + v for u, v in zip(gradients, plane_grads))
            loss += plane_loss

            for parameter, grad, m, v in zip(
                params, gradients, first_moment, second_moment
            ):
                m *= beta1
                m += (1.0 - beta1) * grad
                v *= beta2
                v += (1.0 - beta2) * grad * grad
                m_hat = m / (1.0 - beta1**step)
                v_hat = v / (1.0 - beta2**step)
                parameter -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

        history.append((loss, value_loss, jac_loss))
        if verbose and (epoch % 250 == 0 or epoch == config.epochs - 1):
            print(
                f"epoch {epoch:4d} | lr={lr:.2e} | total={loss:.3e} "
                f"| value={value_loss:.3e} | jac={jac_loss:.3e}", flush=True
            )

    return model, np.asarray(history)


def model_from_weights(W, b, p):
    model = BlockNeuralField(len(b), SEED)
    model.W, model.b, model.p = W.copy(), b.copy(), p.copy()
    return model


def load_model(path):
    """Network and lambda_u stored in a cache written by train_lv.py."""
    z = np.load(path)
    return model_from_weights(z["W"], z["b"], z["p"]), z["lambda_u"]


# ---------------------------------------------------------------------------
# Integration and period

def integrate(field, initial, t_eval=times):
    t_end = float(t_eval[-1])
    sol = solve_ivp(
        lambda t, z: field(z),
        (0, t_end),
        initial,
        t_eval=t_eval,
        **INTEGRATION,
    )
    if not sol.success or len(sol.t) != len(t_eval):
        raise RuntimeError(sol.message)
    if not np.all(np.isfinite(sol.y)):
        raise RuntimeError("Trajectory not finite")
    return sol.y.T


def oriented_crossings(t, x, component, level, t_min=0.0):
    """Upward crossings x_component = level, with linear interpolation."""
    ids = np.flatnonzero(
        (x[:-1, component] < level)
        & (x[1:, component] >= level)
        & (t[:-1] >= t_min)
    )
    if len(ids) == 0:
        return np.array([]), np.empty((0, x.shape[1]))

    alpha = (
        (level - x[ids, component])
        / (x[ids + 1, component] - x[ids, component])
    )
    crossing_times = t[ids] + alpha * (t[ids + 1] - t[ids])
    crossing_states = x[ids] + alpha[:, None] * (x[ids + 1] - x[ids])
    return crossing_times, crossing_states


def estimate_period(t, x):
    """Median of the last five return times to an oriented section.

    The section is the upward crossing of the midpoint of the component with
    the largest peak-to-peak amplitude on the tail t >= 0.65 t_end.  Returns
    the period T, a reference crossing time t0, the relative spread of the
    return times, the relative difference between successive crossing
    states, and the section (component, level).
    """
    tail = t >= 0.65 * t[-1]
    component = int(np.argmax(np.ptp(x[tail], axis=0)))
    amplitude = np.ptp(x[tail, component])
    if amplitude < 1e-3:
        raise RuntimeError("No sustained oscillation for period estimation")

    level = 0.5 * (
        x[tail, component].min() + x[tail, component].max()
    )
    crossings, states = oriented_crossings(
        t, x, component, level, t_min=0.65 * t[-1]
    )
    if len(crossings) < 6:
        raise RuntimeError("Too few late section crossings; increase T_END")

    periods = np.diff(crossings[-6:])
    T = float(np.median(periods))
    spread = float(np.max(np.abs(periods - T)) / T)
    recurrence = float(
        np.max(np.linalg.norm(np.diff(states[-6:], axis=0), axis=1))
        / max(np.linalg.norm(np.ptp(x[tail], axis=0)), 1e-12)
    )
    if max(spread, recurrence) > PERIOD_TOLERANCE:
        raise RuntimeError(
            f"Period not settled: relative spread={spread}, "
            f"recurrence={recurrence}"
        )

    t0 = float(crossings[-4])
    return T, t0, spread, recurrence, component, level


def first_return_time(t, x, component, level):
    """First return time to the same oriented section used for f_theta."""
    crossings, _ = oriented_crossings(t, x, component, level, t_min=0.0)
    if len(crossings) < 2:
        return np.nan, crossings
    return float(crossings[1] - crossings[0]), crossings


# ---------------------------------------------------------------------------
# Diagnostics: errors on K and near Gamma, equilibria near the saddles

def error_summary(model, x, lambda_u):
    value_errors = np.linalg.norm(
        model.field_batch(x) - target_batch(x, lambda_u), axis=1
    )
    jac_errors = np.linalg.norm(
        model.jacobian_batch(x) - target_jacobian_batch(x, lambda_u),
        axis=(1, 2)
    )
    return {
        "field RMS": float(np.sqrt(np.mean(value_errors**2))),
        "field maximum": float(np.max(value_errors)),
        "Jacobian RMS": float(np.sqrt(np.mean(jac_errors**2))),
        "Jacobian maximum": float(np.max(jac_errors)),
    }


def empirical_c1_diagnostics(model, lambda_u, test_size, seed):
    local_rng = np.random.default_rng(seed)
    x = local_rng.uniform(K_MIN, K_MAX, size=(test_size, 3))
    return error_summary(model, x, lambda_u)


def heteroclinic_cycle(lambda_u):
    """Gamma: the three connections a_i e_i -> a_{i+1} e_{i+1} of g."""
    t = np.linspace(0, 150.0, int(round(150.0 / 0.01)) + 1)
    pieces = []
    for i in range(3):
        x0 = np.zeros(3)
        x0[i] = a[i] - 1e-6
        x0[(i + 1) % 3] = 1e-6
        sol = solve_ivp(lambda _, z: g_lv(z, lambda_u), (0, 150.0), x0,
                        t_eval=t, method="DOP853", max_step=0.1,
                        rtol=1e-9, atol=1e-11)
        pieces.append(sol.y.T)
    return np.vstack(pieces)


def near_gamma_diagnostics(model, lambda_u, n_points=400_000, tree=None):
    if tree is None:
        tree = cKDTree(heteroclinic_cycle(lambda_u))
    x = np.random.default_rng(1).uniform(K_MIN, K_MAX, size=(n_points, 3))
    near = tree.query(x)[0] <= GAMMA_RADIUS
    out = error_summary(model, x[near], lambda_u)
    out.update(radius=GAMMA_RADIUS, points_in_K=n_points,
               points_near_Gamma=int(near.sum()))
    return out


def equilibria(model, i, starts=300, radius=0.5):
    """Distinct zeros of f_theta within radius of e_i (Newton from random
    starts in a box of half-side 0.35 around e_i)."""
    r = np.random.default_rng(3)
    found = []
    for _ in range(starts):
        z0 = np.eye(3)[i] + r.uniform(-0.35, 0.35, 3)
        z, _, ok, _ = fsolve(model.field, z0, fprime=model.jacobian,
                             full_output=True, xtol=1e-13)
        if (ok == 1 and np.linalg.norm(model.field(z)) < 1e-10
                and np.linalg.norm(z - np.eye(3)[i]) < radius
                and not any(np.linalg.norm(z - w) < 1e-6 for w in found)):
            found.append(z)
    return found


def complex_list(values):
    values = np.sort_complex(np.asarray(values, dtype=complex))
    return [[float(v.real), float(v.imag)] for v in values]


def equilibria_report(model, lambda_u):
    rows = []
    for i in range(3):
        e_i = a[i] * np.eye(3)[i]
        rows.append({
            "near": f"e_{i + 1} ({NETWORKS[i]})",
            "g_eigenvalues_at_e_i": complex_list(
                np.linalg.eigvals(jac_lv(e_i, lambda_u))),
            "f_theta_equilibria": [
                {"x": z.tolist(),
                 "distance_to_e_i": float(np.linalg.norm(z - e_i)),
                 "residual": float(np.linalg.norm(model.field(z))),
                 "eigenvalues": complex_list(
                     np.linalg.eigvals(model.jacobian(z)))}
                for z in equilibria(model, i)
            ],
        })
    return rows
