"""Normal-form games and their first-order learning dynamics.

Two families are covered:

* :class:`Bimatrix` - two players with two actions each. A mixed profile is a
  point ``(p, q)`` of the unit square (``p`` = probability the row player uses
  its first action, ``q`` the same for the column player), so the learning
  dynamics is a planar vector field: the classic phase portrait.
* :class:`Symmetric` - one population playing an n-strategy symmetric game.
  For n = 3 the state lives on a triangle (the 2-simplex).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from . import simplex
from .dynamics import SMOOTH_RULES

# ---------------------------------------------------------------------------
# Fixed points and their stability
# ---------------------------------------------------------------------------


def jacobian(f, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Central-difference Jacobian of a field ``f: R^d -> R^d`` at ``x``."""
    x = np.asarray(x, dtype=float)
    d = x.shape[0]
    J = np.empty((d, d))
    for j in range(d):
        e = np.zeros(d)
        e[j] = h
        J[:, j] = (np.asarray(f(x + e)) - np.asarray(f(x - e))) / (2 * h)
    return J


def stability(J: np.ndarray, tol: float = 1e-7) -> str:
    """Name the linear behaviour: sink, source, saddle, center or degenerate."""
    ev = np.linalg.eigvals(J)
    re = ev.real
    if np.all(re < -tol):
        return "sink"
    if np.all(re > tol):
        return "source"
    if np.any(re < -tol) and np.any(re > tol):
        return "saddle"
    if np.all(np.abs(re) <= tol) and np.any(np.abs(ev.imag) > tol):
        return "center"
    return "degenerate"


# ---------------------------------------------------------------------------
# Two players, two actions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bimatrix:
    A: np.ndarray  # row player's payoffs, A[i, j] for row action i vs column action j
    B: np.ndarray  # column player's payoffs, same indexing
    name: str = "2x2 game"
    row_actions: tuple[str, str] = ("A", "B")
    col_actions: tuple[str, str] = ("A", "B")
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "A", np.asarray(self.A, dtype=float))
        object.__setattr__(self, "B", np.asarray(self.B, dtype=float))

    def advantages(self, p, q) -> tuple[np.ndarray, np.ndarray]:
        """Payoff advantage of each player's first action (their first-order signal)."""
        A, B = self.A, self.B
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)
        d_row = (A[0, 0] - A[1, 0]) * q + (A[0, 1] - A[1, 1]) * (1 - q)
        d_col = (B[0, 0] - B[0, 1]) * p + (B[1, 0] - B[1, 1]) * (1 - p)
        return d_row, d_col

    def payoffs(self, p, q) -> tuple[np.ndarray, np.ndarray]:
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)
        w = np.stack([p * q, p * (1 - q), (1 - p) * q, (1 - p) * (1 - q)])
        return np.tensordot(self.A.ravel(), w, 1), np.tensordot(self.B.ravel(), w, 1)

    def velocity(self, p, q, rule: str = "replicator", rates=(1.0, 1.0), temperature: float = 0.05):
        d_row, d_col = self.advantages(p, q)
        return (
            rates[0] * simplex.binary_field(rule, p, d_row, temperature),
            rates[1] * simplex.binary_field(rule, q, d_col, temperature),
        )

    def discrete_step(self, p, q, rule: str = "replicator", eta: float = 0.1, rates=(1.0, 1.0), temperature: float = 0.05):
        """One simultaneous step of both players' discrete-time learning algorithms."""
        d_row, d_col = self.advantages(p, q)
        return (
            simplex.binary_discrete_step(rule, p, d_row, eta * rates[0], temperature),
            simplex.binary_discrete_step(rule, q, d_col, eta * rates[1], temperature),
        )

    def simulate(self, p0, q0, rule="replicator", t_max=20.0, dt=0.01, rates=(1.0, 1.0), temperature=0.05):
        """Continuous-time trajectory (RK4 for smooth rules, Euler otherwise)."""
        P = np.array(p0, dtype=float)
        Q = np.array(q0, dtype=float)
        n = int(round(t_max / dt))
        out = np.empty((n + 1,) + np.broadcast(P, Q).shape + (2,))
        out[0, ..., 0], out[0, ..., 1] = P, Q
        f = lambda a, b: self.velocity(a, b, rule, rates, temperature)  # noqa: E731
        for i in range(1, n + 1):
            if rule in SMOOTH_RULES:
                a1, b1 = f(P, Q)
                a2, b2 = f(P + dt / 2 * a1, Q + dt / 2 * b1)
                a3, b3 = f(P + dt / 2 * a2, Q + dt / 2 * b2)
                a4, b4 = f(P + dt * a3, Q + dt * b3)
                P = P + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
                Q = Q + dt / 6 * (b1 + 2 * b2 + 2 * b3 + b4)
            else:
                a1, b1 = f(P, Q)
                P, Q = P + dt * a1, Q + dt * b1
            P, Q = np.clip(P, 0, 1), np.clip(Q, 0, 1)
            out[i, ..., 0], out[i, ..., 1] = P, Q
        return out

    def nash_equilibria(self, tol: float = 1e-12) -> list[tuple[float, float]]:
        """All isolated Nash equilibria: pure profiles and the interior mixed one."""
        eqs = []
        for p in (1.0, 0.0):
            for q in (1.0, 0.0):
                d_row, d_col = self.advantages(p, q)
                row_ok = d_row >= -tol if p == 1.0 else d_row <= tol
                col_ok = d_col >= -tol if q == 1.0 else d_col <= tol
                if row_ok and col_ok:
                    eqs.append((p, q))
        A, B = self.A, self.B
        den_q = (A[0, 0] - A[1, 0]) - (A[0, 1] - A[1, 1])
        den_p = (B[0, 0] - B[0, 1]) - (B[1, 0] - B[1, 1])
        if abs(den_q) > tol and abs(den_p) > tol:
            q = -(A[0, 1] - A[1, 1]) / den_q
            p = -(B[1, 0] - B[1, 1]) / den_p
            if 0 < p < 1 and 0 < q < 1:
                eqs.append((float(p), float(q)))
        return eqs

    def equilibrium_stability(self, point, rule: str = "replicator") -> str:
        """Linear stability of a rest point under ``rule`` (replicator by default)."""
        f = lambda z: np.array(self.velocity(z[0], z[1], rule)).ravel()  # noqa: E731
        return stability(jacobian(f, np.array(point, dtype=float)))


def symmetric_2x2(R: float = 1.0, S: float = 0.0, T: float = 0.0, P: float = 0.0, name: str | None = None) -> Bimatrix:
    """Symmetric game with Reward, Sucker, Temptation and Punishment payoffs.

    Actions are Cooperate and Defect. Fixing ``R = 1`` and ``P = 0`` leaves the
    (T, S) plane in which the four classic social dilemmas are quadrants.
    """
    A = np.array([[R, S], [T, P]])
    return Bimatrix(A, A.T, name or ts_region(T, S, R, P), ("Cooperate", "Defect"), ("Cooperate", "Defect"))


def ts_region(T: float, S: float, R: float = 1.0, P: float = 0.0) -> str:
    """Name of the (T, S) quadrant for a symmetric 2x2 game with R > P."""
    greedy = T > R  # temptation to defect on a cooperator
    fear = S < P  # fear of being exploited by a defector
    if greedy and fear:
        return "Prisoner's Dilemma"
    if greedy:
        return "Snowdrift (Chicken)"
    if fear:
        return "Stag Hunt"
    return "Harmony"


BIMATRIX_PRESETS = {
    "prisoners_dilemma": Bimatrix(
        [[3, 0], [5, 1]], [[3, 5], [0, 1]], "Prisoner's Dilemma", ("Cooperate", "Defect"), ("Cooperate", "Defect"),
        "Defecting is better whatever the other does; both end up worse off.",
    ),
    "stag_hunt": Bimatrix(
        [[4, 0], [3, 3]], [[4, 3], [0, 3]], "Stag Hunt", ("Stag", "Hare"), ("Stag", "Hare"),
        "Two stable conventions; the safe one has the larger basin.",
    ),
    "chicken": Bimatrix(
        [[0, -1], [1, -10]], [[0, 1], [-1, -10]], "Chicken (Hawk-Dove)", ("Swerve", "Straight"), ("Swerve", "Straight"),
        "Anti-coordination: one player yields. The mixed equilibrium is a saddle between two populations.",
    ),
    "matching_pennies": Bimatrix(
        [[1, -1], [-1, 1]], [[-1, 1], [1, -1]], "Matching Pennies", ("Heads", "Tails"), ("Heads", "Tails"),
        "Zero-sum and purely rotational: replicator orbits close, discrete learning spirals out.",
    ),
    "battle_of_sexes": Bimatrix(
        [[3, 0], [0, 2]], [[2, 0], [0, 3]], "Battle of the Sexes", ("Opera", "Football"), ("Opera", "Football"),
        "Coordination with a conflict over which convention.",
    ),
    "coordination": Bimatrix(
        [[1, 0], [0, 1]], [[1, 0], [0, 1]], "Pure Coordination", ("Left", "Right"), ("Left", "Right"),
        "Drive on the left or on the right: it only matters to agree.",
    ),
}


# ---------------------------------------------------------------------------
# One population, n strategies
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Symmetric:
    A: np.ndarray  # A[i, j]: payoff of strategy i against strategy j
    name: str = "Symmetric game"
    labels: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "A", np.asarray(self.A, dtype=float))
        if not self.labels:
            object.__setattr__(self, "labels", tuple(f"s{i + 1}" for i in range(self.A.shape[0])))

    @property
    def n(self) -> int:
        return self.A.shape[0]

    def signal(self, x: np.ndarray) -> np.ndarray:
        """Payoff of every pure strategy against population state ``x``."""
        return np.einsum("ij,...j->...i", self.A, x)

    def velocity(self, x: np.ndarray, rule: str = "replicator", temperature: float = 0.05) -> np.ndarray:
        return simplex.field(rule, x, self.signal(x), temperature)

    def simulate(self, x0, rule="replicator", t_max=30.0, dt=0.01, temperature=0.05, record_every=1):
        from .dynamics import integrate

        times, states = integrate(
            lambda y: (self.velocity(y[0], rule, temperature),),
            (np.asarray(x0, dtype=float),),
            t_max,
            dt,
            smooth=rule in SMOOTH_RULES,
            record_every=record_every,
        )
        return times, np.stack([s[0] for s in states])

    def rest_points(self, tol: float = 1e-9) -> list[dict]:
        """Rest points of the replicator dynamics, one per support.

        Each entry has the point ``x``, whether it is a Nash equilibrium, and
        its linear stability (computed in barycentric coordinates).
        """
        n = self.n
        found = []
        for size in range(1, n + 1):
            for support in combinations(range(n), size):
                idx = list(support)
                M = np.zeros((size + 1, size + 1))
                M[:size, :size] = self.A[np.ix_(idx, idx)]
                M[:size, size] = -1.0
                M[size, :size] = 1.0
                rhs = np.zeros(size + 1)
                rhs[size] = 1.0
                try:
                    sol = np.linalg.solve(M, rhs)
                except np.linalg.LinAlgError:
                    continue
                if np.any(sol[:size] <= tol):
                    continue
                x = np.zeros(n)
                x[idx] = sol[:size]
                f = self.signal(x)
                nash = bool(np.all(f <= sol[size] + 1e-9))
                found.append(dict(x=x, support=support, nash=nash, stability=self._stability(x)))
        return found

    def _stability(self, x: np.ndarray, rule: str = "replicator") -> str:
        # Coordinates: the first n-1 shares, the last one is 1 - sum.
        def f(z):
            full = np.append(z, 1 - z.sum())
            return self.velocity(full, rule)[:-1]

        return stability(jacobian(f, x[:-1]))


def rock_paper_scissors(win: float = 1.0, lose: float = 1.0) -> Symmetric:
    """Rock-paper-scissors where a win pays ``win`` and a loss costs ``lose``.

    ``win == lose`` is zero-sum and the replicator orbits are closed;
    ``win > lose`` makes the mixed equilibrium attract; ``win < lose`` makes it
    repel, and play spirals out toward the boundary cycle R -> P -> S -> R.
    """
    A = np.array([[0, -lose, win], [win, 0, -lose], [-lose, win, 0]], dtype=float)
    if win == lose:
        kind = "zero-sum"
    elif win > lose:
        kind = "wins pay more"
    else:
        kind = "losses cost more"
    return Symmetric(A, f"Rock-Paper-Scissors ({kind})", ("Rock", "Paper", "Scissors"))


SYMMETRIC_PRESETS = {
    "rps_zero_sum": rock_paper_scissors(1.0, 1.0),
    "rps_stable": rock_paper_scissors(1.0, 0.5),
    "rps_unstable": rock_paper_scissors(0.5, 1.0),
    "coordination3": Symmetric(
        np.diag([1.0, 2.0, 3.0]), "Three conventions", ("Left", "Middle", "Right"),
        "Any shared convention is stable; the better ones have larger basins.",
    ),
    "hawk_dove_bourgeois": Symmetric(
        [[-1.0, 2.0, 0.5], [0.0, 1.0, 0.5], [-0.5, 1.5, 1.0]], "Hawk-Dove-Bourgeois", ("Hawk", "Dove", "Bourgeois"),
        "Maynard Smith's owner-fights, intruder-yields convention (V = 2, C = 4).",
    ),
    "repeated_pd": Symmetric(
        [[30.0, 0.0, 30.0], [50.0, 10.0, 14.0], [30.0, 9.0, 30.0]], "Repeated Prisoner's Dilemma (10 rounds)",
        ("Always cooperate", "Always defect", "Tit for tat"),
        "Tit for tat resists defectors but drifts neutrally against unconditional cooperators.",
    ),
}
