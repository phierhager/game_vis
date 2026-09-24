"""Duopoly markets as games with continuous strategies.

The first-order signal of a firm is its *marginal profit*: the derivative of
its own profit with respect to its own price (or quantity). Gradient play,
``dp_i/dt = rate_i * d profit_i / d p_i``, is the market version of every
player following its first-order signal. With a *sympathy* weight ``lam`` each
firm instead climbs ``profit_i + lam * profit_j``: ``lam = 0`` is pure
competition, ``lam = 1`` is a cartel maximising joint profit.

Models
------
* :class:`LogitBertrand` - differentiated products with logit demand, the
  workhorse of the algorithmic-collusion literature (Calvano et al., 2020).
* :class:`LinearBertrand` - linear differentiated demand with closed forms.
* :class:`Cournot` - quantity competition with linear inverse demand.
* :class:`Edgeworth` - homogeneous goods with capacity limits: best replies
  undercut until undercutting stops paying, then jump back up, so prices
  cycle forever instead of settling (Edgeworth, 1897).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class _Duopoly:
    """Shared machinery: profits, sympathy gradients, best replies, equilibria."""

    lo: float = 0.0
    hi: float = 1.0

    def profit(self, p1, p2):  # pragma: no cover - defined by subclasses
        raise NotImplementedError

    def marginal_profit(self, p1, p2):  # pragma: no cover
        raise NotImplementedError

    def cross_marginal(self, p1, p2):
        """``(d profit_2 / d p1, d profit_1 / d p2)`` by central differences."""
        h = 1e-6
        d21 = (self.profit(p1 + h, p2)[1] - self.profit(p1 - h, p2)[1]) / (2 * h)
        d12 = (self.profit(p1, p2 + h)[0] - self.profit(p1, p2 - h)[0]) / (2 * h)
        return d21, d12

    def velocity(self, p1, p2, rates=(1.0, 1.0), sympathy: float = 0.0):
        """Gradient play: each firm follows the slope of ``profit_i + lam * profit_j`` in its own price."""
        g1, g2 = self.marginal_profit(p1, p2)
        if sympathy:
            d21, d12 = self.cross_marginal(p1, p2)
            g1 = g1 + sympathy * d21
            g2 = g2 + sympathy * d12
        v1 = rates[0] * np.asarray(g1)
        v2 = rates[1] * np.asarray(g2)
        # Keep prices inside the plotted box.
        v1 = np.where(((np.asarray(p1) <= self.lo) & (v1 < 0)) | ((np.asarray(p1) >= self.hi) & (v1 > 0)), 0.0, v1)
        v2 = np.where(((np.asarray(p2) <= self.lo) & (v2 < 0)) | ((np.asarray(p2) >= self.hi) & (v2 > 0)), 0.0, v2)
        return v1, v2

    def simulate(self, p0, t_max=50.0, dt=0.01, rates=(1.0, 1.0), sympathy: float = 0.0):
        p = np.array(p0, dtype=float)
        n = int(round(t_max / dt))
        out = np.empty((n + 1,) + p.shape)
        out[0] = p
        f = lambda z: np.stack(self.velocity(z[..., 0], z[..., 1], rates, sympathy), -1)  # noqa: E731
        for i in range(1, n + 1):
            k1 = f(p)
            k2 = f(p + dt / 2 * k1)
            k3 = f(p + dt / 2 * k2)
            k4 = f(p + dt * k3)
            p = np.clip(p + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), self.lo, self.hi)
            out[i] = p
        return out

    def best_response(self, i: int, other, sympathy: float = 0.0, grid: int = 2001):
        """Best reply of firm ``i`` (0 or 1) to the other firm's price: grid search, then Newton polish."""
        other = np.atleast_1d(np.asarray(other, dtype=float))
        xs = np.linspace(self.lo, self.hi, grid)
        out = np.empty_like(other)
        for n, o in enumerate(other):
            if i == 0:
                pr = self.profit(xs, np.full_like(xs, o))
                obj = pr[0] + sympathy * pr[1]
            else:
                pr = self.profit(np.full_like(xs, o), xs)
                obj = pr[1] + sympathy * pr[0]
            x = xs[int(np.argmax(obj))]
            out[n] = self._polish(i, x, o, sympathy)
        return out

    def _polish(self, i, x, o, sympathy, steps=30):
        h = 1e-5

        def obj(z):
            pr = self.profit(z, o) if i == 0 else self.profit(o, z)
            return pr[i] + sympathy * pr[1 - i]

        for _ in range(steps):
            d1 = (obj(x + h) - obj(x - h)) / (2 * h)
            d2 = (obj(x + h) - 2 * obj(x) + obj(x - h)) / h**2
            if d2 >= 0:
                break
            step = d1 / d2
            x_new = min(max(x - step, self.lo), self.hi)
            if abs(x_new - x) < 1e-12:
                break
            x = x_new
        return float(x)

    def equilibrium(self, sympathy: float = 0.0, iters: int = 500):
        """Fixed point of the (sympathetic) best replies by damped iteration."""
        p = np.array([(self.lo + self.hi) / 2] * 2)
        for _ in range(iters):
            new = np.array([self.best_response(0, p[1], sympathy)[0], self.best_response(1, p[0], sympathy)[0]])
            if np.abs(new - p).max() < 1e-11:
                p = new
                break
            p = 0.5 * p + 0.5 * new
        return p

    def nash(self):
        return self.equilibrium(0.0)

    def collusive(self):
        """Joint-profit maximum (the cartel outcome)."""
        return self.equilibrium(1.0)


@dataclass
class LogitBertrand(_Duopoly):
    """Price competition with logit demand.

    ``q_i = exp((a_i - p_i)/mu) / (sum_j exp((a_j - p_j)/mu) + exp(a0/mu))``.
    Defaults are the baseline of Calvano, Calzolari, Denicolo and Pastorello
    (2020): costs 1, qualities 2, outside good 0, ``mu = 0.25``. Smaller ``mu``
    means closer substitutes and a fiercer price war.
    """

    quality: tuple[float, float] = (2.0, 2.0)
    cost: tuple[float, float] = (1.0, 1.0)
    mu: float = 0.25
    a0: float = 0.0
    lo: float = 1.0
    hi: float = 2.4

    def demand(self, p1, p2):
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        z1 = (self.quality[0] - p1) / self.mu
        z2 = (self.quality[1] - p2) / self.mu
        z0 = self.a0 / self.mu
        m = np.maximum(np.maximum(z1, z2), z0)
        e1, e2, e0 = np.exp(z1 - m), np.exp(z2 - m), np.exp(z0 - m)
        tot = e1 + e2 + e0
        return e1 / tot, e2 / tot

    def profit(self, p1, p2):
        q1, q2 = self.demand(p1, p2)
        return (np.asarray(p1) - self.cost[0]) * q1, (np.asarray(p2) - self.cost[1]) * q2

    def marginal_profit(self, p1, p2):
        q1, q2 = self.demand(p1, p2)
        g1 = q1 * (1 - (np.asarray(p1) - self.cost[0]) * (1 - q1) / self.mu)
        g2 = q2 * (1 - (np.asarray(p2) - self.cost[1]) * (1 - q2) / self.mu)
        return g1, g2

    def cross_marginal(self, p1, p2):
        q1, q2 = self.demand(p1, p2)
        # d q_j / d p_i = q_i q_j / mu
        d21 = (np.asarray(p2) - self.cost[1]) * q1 * q2 / self.mu
        d12 = (np.asarray(p1) - self.cost[0]) * q1 * q2 / self.mu
        return d21, d12


@dataclass
class LinearBertrand(_Duopoly):
    """Differentiated price competition with ``q_i = alpha - beta p_i + gamma p_j``."""

    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 0.5
    cost: float = 0.0
    lo: float = 0.0
    hi: float = 1.5

    def demand(self, p1, p2):
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        q1 = np.maximum(self.alpha - self.beta * p1 + self.gamma * p2, 0.0)
        q2 = np.maximum(self.alpha - self.beta * p2 + self.gamma * p1, 0.0)
        return q1, q2

    def profit(self, p1, p2):
        q1, q2 = self.demand(p1, p2)
        return (np.asarray(p1) - self.cost) * q1, (np.asarray(p2) - self.cost) * q2

    def marginal_profit(self, p1, p2):
        q1, q2 = self.demand(p1, p2)
        g1 = np.where(q1 > 0, q1 - self.beta * (np.asarray(p1) - self.cost), 0.0)
        g2 = np.where(q2 > 0, q2 - self.beta * (np.asarray(p2) - self.cost), 0.0)
        return g1, g2

    def nash_closed_form(self) -> float:
        return (self.alpha + self.beta * self.cost) / (2 * self.beta - self.gamma)

    def collusive_closed_form(self) -> float:
        b = self.beta - self.gamma
        return (self.alpha + b * self.cost) / (2 * b)


@dataclass
class Cournot(_Duopoly):
    """Quantity competition with inverse demand ``P = a - b (q1 + q2)`` and unit cost ``c``."""

    a: float = 1.0
    b: float = 1.0
    c: float = 0.0
    lo: float = 0.0
    hi: float = 0.6

    def price(self, q1, q2):
        return np.maximum(self.a - self.b * (np.asarray(q1) + np.asarray(q2)), 0.0)

    def profit(self, q1, q2):
        P = self.price(q1, q2)
        return (P - self.c) * np.asarray(q1), (P - self.c) * np.asarray(q2)

    def marginal_profit(self, q1, q2):
        P = self.price(q1, q2)
        return P - self.c - self.b * np.asarray(q1), P - self.c - self.b * np.asarray(q2)

    def nash_closed_form(self) -> float:
        return (self.a - self.c) / (3 * self.b)

    def collusive_closed_form(self) -> float:
        return (self.a - self.c) / (4 * self.b)


@dataclass
class Edgeworth:
    """Homogeneous-good price competition with capacity limits (efficient rationing).

    Demand is ``D(p) = 1 - p``; each firm can sell at most ``capacity`` at unit
    cost ``cost``. With ``capacity >= D(cost)`` this is the textbook Bertrand
    game and undercutting drives prices to cost. With
    ``D(cost) / 2 < capacity < D(cost)`` the undercut firm is better off
    serving the leftover demand at a high price, so best replies undercut
    down to a floor and then jump back up: an Edgeworth cycle.
    """

    capacity: float = 0.5
    cost: float = 0.1
    tick: float = 0.01

    def demand(self, p):
        return np.clip(1.0 - np.asarray(p, dtype=float), 0.0, None)

    def sales(self, p_own, p_other):
        """Quantity sold by a firm charging ``p_own`` against ``p_other``."""
        p_own = np.asarray(p_own, dtype=float)
        p_other = np.asarray(p_other, dtype=float)
        k = self.capacity
        low = np.minimum(k, self.demand(p_own))
        high = np.minimum(k, np.clip(self.demand(p_own) - k, 0.0, None))
        tie = np.minimum(k, self.demand(p_own) / 2)
        return np.where(p_own < p_other, low, np.where(p_own > p_other, high, tie))

    def profit(self, p_own, p_other):
        return (np.asarray(p_own) - self.cost) * self.sales(p_own, p_other)

    def monopoly_price(self) -> float:
        return (1.0 + self.cost) / 2

    def residual_price(self) -> float:
        """Best price when the rival undercuts and serves ``capacity`` units first."""
        return max((1.0 - self.capacity + self.cost) / 2, self.cost)

    def best_response(self, p_other: float) -> float:
        """Myopic best reply on the price grid (ties go to the higher price)."""
        grid = np.round(np.arange(self.cost, 1.0 + 1e-9, self.tick), 10)
        prof = self.profit(grid, p_other)
        best = np.flatnonzero(prof >= prof.max() - 1e-12)
        return float(grid[best[-1]])

    def simulate(self, p0=(0.9, 0.9), periods: int = 120) -> np.ndarray:
        """Alternating moves: in period t firm ``t % 2`` best-responds to the other's price."""
        p = np.array(p0, dtype=float)
        out = [p.copy()]
        for t in range(periods):
            i = t % 2
            p[i] = self.best_response(p[1 - i])
            out.append(p.copy())
        return np.array(out)
