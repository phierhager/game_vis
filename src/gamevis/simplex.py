"""First-order learning rules on probability simplices.

A player's mixed strategy is a probability vector ``x`` (the last axis indexes
the player's actions). Its *first-order signal* is the payoff vector ``g`` with
``g[i] = dU/dx[i]``: the marginal payoff of moving weight onto action ``i``.
For a player whose expected payoff is linear in its own strategy (every
normal-form game, every agent of a signaling game) that vector is exactly the
gradient of the player's own payoff.

Every rule below maps ``(x, g)`` to a velocity ``dx/dt`` tangent to the
simplex. Arrays may carry any number of leading batch axes; the rules act on
the last axis. The same rules exist as discrete-time learning algorithms with a
step size ``eta`` (:func:`discrete_step`); as ``eta -> 0`` they trace the ODEs.

=================  ==========================================  ======================
rule               continuous time                             discrete-time algorithm
=================  ==========================================  ======================
``replicator``     x_i (g_i - x.g)                             multiplicative weights
``projection``     g projected onto the simplex tangent cone   projected gradient ascent
``softmax_pg``     J(x) J(x) g with J = diag(x) - x x^T        policy gradient on logits
``logit``          softmax(g / tau) - x                        smoothed fictitious play
``best_response``  BR(g) - x                                   fictitious play
=================  ==========================================  ======================

Only the first three are *gradient* (first-order) dynamics in the strict
sense; ``logit`` and ``best_response`` use the same signal but jump toward the
(smoothed) best reply and are included for contrast.
"""

from __future__ import annotations

import numpy as np

RULES = ("replicator", "projection", "softmax_pg", "logit", "best_response")

GRADIENT_RULES = ("replicator", "projection", "softmax_pg")

RULE_LABELS = {
    "replicator": "Replicator (multiplicative weights)",
    "projection": "Projected gradient",
    "softmax_pg": "Softmax policy gradient",
    "logit": "Logit response",
    "best_response": "Best response",
}


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def replicator(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Replicator dynamics, the continuous-time limit of multiplicative weights."""
    return x * (g - np.sum(x * g, axis=-1, keepdims=True))


def projection(x: np.ndarray, g: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Euclidean projection dynamics.

    Projects ``g`` onto the tangent cone of the simplex at ``x``: the component
    of the gradient that keeps ``x`` feasible. In the interior this is just
    ``g - mean(g)``; on a face, coordinates at zero may only grow.
    """
    x = np.asarray(x, dtype=float)
    g = np.asarray(g, dtype=float)
    shape = np.broadcast_shapes(x.shape, g.shape)
    x = np.broadcast_to(x, shape)
    g = np.broadcast_to(g, shape)
    n = shape[-1]
    zero = x <= tol
    free_sum = np.where(zero, 0.0, g).sum(axis=-1)
    free_cnt = (~zero).sum(axis=-1)
    # Coordinates sitting at zero join the free set, largest signal first, as
    # long as their signal beats the running mean (active-set solution of the
    # KKT system; the admissible set is always a prefix of this order).
    gz = -np.sort(-np.where(zero, g, -np.inf), axis=-1)
    finite = np.isfinite(gz)
    csum = np.cumsum(np.where(finite, gz, 0.0), axis=-1)
    k = np.arange(1, n + 1)
    mu_k = (free_sum[..., None] + csum) / (free_cnt[..., None] + k)
    n_join = (finite & (gz > mu_k)).sum(axis=-1)
    mu0 = free_sum / np.maximum(free_cnt, 1)
    mu_join = np.take_along_axis(mu_k, np.maximum(n_join - 1, 0)[..., None], axis=-1)[..., 0]
    mu = np.where(n_join > 0, mu_join, mu0)
    v = g - mu[..., None]
    return np.where(zero & (v < 0), 0.0, v)


def softmax_pg(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Gradient ascent on softmax logits, pushed forward to strategy space.

    With ``x = softmax(theta)`` the logit gradient is ``J g`` where
    ``J = diag(x) - x x^T``; strategies then move by ``J (J g)``, which is the
    replicator field applied twice.
    """
    return replicator(x, replicator(x, g))


def logit(x: np.ndarray, g: np.ndarray, temperature: float = 0.05) -> np.ndarray:
    """Logit (smoothed best response) dynamics."""
    return softmax(np.asarray(g) / temperature) - x


def best_response_vector(g: np.ndarray, x: np.ndarray | None = None, tol: float = 1e-12) -> np.ndarray:
    """A best reply to ``g`` along the last axis.

    Among several maximisers the reply keeps the current relative weights of
    ``x`` (so exact ties, such as a message nobody sends, do not cause motion);
    without ``x``, or when ``x`` puts no weight on any maximiser, it mixes them
    uniformly.
    """
    g = np.asarray(g, dtype=float)
    best = (g >= g.max(axis=-1, keepdims=True) - tol).astype(float)
    uniform = best / best.sum(axis=-1, keepdims=True)
    if x is None:
        return uniform
    kept = np.asarray(x, dtype=float) * best
    total = kept.sum(axis=-1, keepdims=True)
    return np.where(total > tol, kept / np.where(total > tol, total, 1.0), uniform)


def best_response(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Best-response dynamics: move straight toward the current best reply."""
    return best_response_vector(g, x) - x


def field(rule: str, x: np.ndarray, g: np.ndarray, temperature: float = 0.05) -> np.ndarray:
    """Velocity of ``rule`` at strategy ``x`` given first-order signal ``g``."""
    if rule == "replicator":
        return replicator(x, g)
    if rule == "projection":
        return projection(x, g)
    if rule == "softmax_pg":
        return softmax_pg(x, g)
    if rule == "logit":
        return logit(x, g, temperature)
    if rule == "best_response":
        return best_response(x, g)
    raise ValueError(f"unknown rule {rule!r}; expected one of {RULES}")


def project_to_simplex(y: np.ndarray) -> np.ndarray:
    """Euclidean projection of each row of ``y`` onto the probability simplex."""
    y = np.asarray(y, dtype=float)
    n = y.shape[-1]
    u = -np.sort(-y, axis=-1)
    css = np.cumsum(u, axis=-1) - 1.0
    idx = np.arange(1, n + 1)
    cond = u - css / idx > 0
    rho = n - 1 - np.argmax(cond[..., ::-1], axis=-1)
    theta = np.take_along_axis(css, rho[..., None], axis=-1) / (rho[..., None] + 1.0)
    return np.maximum(y - theta, 0.0)


def discrete_step(rule: str, x: np.ndarray, g: np.ndarray, eta: float, temperature: float = 0.05) -> np.ndarray:
    """One step of the discrete-time learning algorithm behind ``rule``."""
    x = np.asarray(x, dtype=float)
    if rule == "replicator":
        w = x * np.exp(eta * (g - g.max(axis=-1, keepdims=True)))
        return w / w.sum(axis=-1, keepdims=True)
    if rule == "projection":
        return project_to_simplex(x + eta * g)
    if rule == "softmax_pg":
        theta = np.log(np.clip(x, 1e-300, None))
        return softmax(theta + eta * replicator(x, g))
    if rule == "logit":
        return (1 - eta) * x + eta * softmax(np.asarray(g) / temperature)
    if rule == "best_response":
        return (1 - eta) * x + eta * best_response_vector(g, x)
    raise ValueError(f"unknown rule {rule!r}; expected one of {RULES}")


# ---------------------------------------------------------------------------
# Two actions: the whole simplex is one number
# ---------------------------------------------------------------------------


def binary_field(rule: str, x: np.ndarray, d: np.ndarray, temperature: float = 0.05) -> np.ndarray:
    """Velocity of ``x = P(first action)`` for a two-action player.

    ``d = g[0] - g[1]`` is the payoff advantage of the first action. This is
    :func:`field` restricted to the segment ``[(1, 0), (0, 1)]``.
    """
    x = np.asarray(x, dtype=float)
    d = np.asarray(d, dtype=float)
    if rule == "replicator":
        return x * (1 - x) * d
    if rule == "projection":
        v = d / 2
        stuck = ((x <= 1e-12) & (v < 0)) | ((x >= 1 - 1e-12) & (v > 0))
        return np.where(stuck, 0.0, v)
    if rule == "softmax_pg":
        return 2 * (x * (1 - x)) ** 2 * d
    if rule == "logit":
        return 0.5 * (1 + np.tanh(d / (2 * temperature))) - x
    if rule == "best_response":
        return np.where(d > 1e-12, 1 - x, np.where(d < -1e-12, -x, 0.0))
    raise ValueError(f"unknown rule {rule!r}; expected one of {RULES}")


def binary_discrete_step(rule: str, x: np.ndarray, d: np.ndarray, eta: float, temperature: float = 0.05) -> np.ndarray:
    """:func:`discrete_step` for a two-action player (``x`` = P(first action))."""
    x = np.asarray(x, dtype=float)
    d = np.asarray(d, dtype=float)
    if rule == "replicator":
        # Multiplicative weights: odds are multiplied by exp(eta * d).
        z = np.log(np.clip(x, 1e-300, None)) - np.log(np.clip(1 - x, 1e-300, None)) + eta * d
        return 0.5 * (1 + np.tanh(z / 2))
    if rule == "projection":
        return np.clip(x + eta * d / 2, 0.0, 1.0)
    if rule == "softmax_pg":
        z = np.log(np.clip(x, 1e-300, None)) - np.log(np.clip(1 - x, 1e-300, None)) + 2 * eta * x * (1 - x) * d
        return 0.5 * (1 + np.tanh(z / 2))
    if rule == "logit":
        return (1 - eta) * x + eta * 0.5 * (1 + np.tanh(d / (2 * temperature)))
    if rule == "best_response":
        target = np.where(d > 1e-12, 1.0, np.where(d < -1e-12, 0.0, x))
        return (1 - eta) * x + eta * target
    raise ValueError(f"unknown rule {rule!r}; expected one of {RULES}")


def clean(x: np.ndarray) -> np.ndarray:
    """Clip tiny negative round-off and renormalise rows onto the simplex."""
    x = np.clip(x, 0.0, None)
    return x / x.sum(axis=-1, keepdims=True)


def random_simplex(rng: np.random.Generator, shape: tuple[int, ...], n: int) -> np.ndarray:
    """Uniform samples from the (n-1)-simplex (flat Dirichlet)."""
    return rng.dirichlet(np.ones(n), size=shape)
