"""Tiny fixed-step integrators for systems whose state is a tuple of simplices.

States are tuples of arrays; every array's last axis is a probability simplex
(one row per agent). Leading axes are free, so a whole ensemble of initial
conditions integrates in one vectorised call.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .simplex import clean, project_to_simplex

State = tuple[np.ndarray, ...]
Field = Callable[[State], State]

# Rules whose vector field is smooth: integrate with RK4. The others are
# discontinuous on faces or at switching surfaces: integrate with Euler.
SMOOTH_RULES = ("replicator", "softmax_pg", "logit")


def _axpy(a: float, xs: Sequence[np.ndarray], ys: Sequence[np.ndarray]) -> State:
    return tuple(y + a * x for x, y in zip(xs, ys))


def rk4_step(f: Field, y: State, dt: float) -> State:
    k1 = f(y)
    k2 = f(_axpy(dt / 2, k1, y))
    k3 = f(_axpy(dt / 2, k2, y))
    k4 = f(_axpy(dt, k3, y))
    return tuple(
        clean(yi + dt / 6 * (a + 2 * b + 2 * c + d))
        for yi, a, b, c, d in zip(y, k1, k2, k3, k4)
    )


def euler_step(f: Field, y: State, dt: float, project: bool = True) -> State:
    k = f(y)
    if project:
        return tuple(project_to_simplex(yi + dt * ki) for yi, ki in zip(y, k))
    return tuple(clean(yi + dt * ki) for yi, ki in zip(y, k))


def integrate(
    f: Field,
    y0: State,
    t_max: float,
    dt: float,
    smooth: bool = True,
    record_every: int = 1,
) -> tuple[np.ndarray, list[State]]:
    """Integrate ``dy/dt = f(y)`` and return ``(times, states)``.

    ``smooth=True`` uses RK4, otherwise projected Euler (the right scheme for
    projection and best-response dynamics, whose fields jump on faces).
    """
    n_steps = int(round(t_max / dt))
    y = tuple(np.array(a, dtype=float) for a in y0)
    times = [0.0]
    states = [y]
    step = rk4_step if smooth else euler_step
    for i in range(1, n_steps + 1):
        y = step(f, y, dt)
        if i % record_every == 0 or i == n_steps:
            times.append(i * dt)
            states.append(y)
    return np.array(times), states
