"""Tabular Q-learning pricing agents in a logit Bertrand duopoly.

This is the contrast case for the rest of the package. First-order learners
follow the slope of their own profit and end at the Nash prices. Q-learning
agents with one period of memory (Calvano, Calzolari, Denicolo and Pastorello,
2020) condition their price on last period's prices, can learn to punish
undercutting, and often settle well above Nash.

Setup (the paper's baseline): 15 prices spread from 10% below the Nash price
to 10% above the monopoly price, state = last period's price pair (225 states),
learning rate 0.15, discount 0.95, epsilon-greedy exploration decaying as
exp(-beta t), Q initialised at the discounted payoff against a uniformly random
rival.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .markets import LogitBertrand


@dataclass
class QPricing:
    market: LogitBertrand = field(default_factory=LogitBertrand)
    m: int = 15
    xi: float = 0.1
    alpha: float = 0.15
    delta: float = 0.95
    beta: float = 4e-6

    def __post_init__(self) -> None:
        self.p_nash = float(self.market.nash()[0])
        self.p_monopoly = float(self.market.collusive()[0])
        span = self.p_monopoly - self.p_nash
        self.prices = np.linspace(self.p_nash - self.xi * span, self.p_monopoly + self.xi * span, self.m)
        P1, P2 = np.meshgrid(self.prices, self.prices, indexing="ij")
        pr1, pr2 = self.market.profit(P1, P2)
        self.profit = np.stack([pr1, pr2])  # profit[i, a1, a2]
        self.profit_nash = float(self.market.profit(self.p_nash, self.p_nash)[0])
        self.profit_monopoly = float(self.market.profit(self.p_monopoly, self.p_monopoly)[0])

    @property
    def n_states(self) -> int:
        return self.m * self.m

    def initial_q(self) -> np.ndarray:
        """Q[i, s, a]: payoff of a against a uniform rival, discounted forever (same for every state)."""
        q1 = self.profit[0].mean(axis=1) / (1 - self.delta)  # firm 1's action a1 vs uniform a2
        q2 = self.profit[1].mean(axis=0) / (1 - self.delta)
        Q = np.empty((2, self.n_states, self.m))
        Q[0] = q1[None, :]
        Q[1] = q2[None, :]
        return Q

    def profit_gain(self, profits: np.ndarray) -> np.ndarray:
        """Delta = (profit - Nash profit) / (monopoly profit - Nash profit)."""
        return (profits - self.profit_nash) / (self.profit_monopoly - self.profit_nash)

    def run(self, sessions: int = 8, periods: int = 1_000_000, seed: int = 0, record_every: int = 1000) -> dict:
        """Train ``sessions`` independent pairs of agents in lockstep."""
        rng = np.random.default_rng(seed)
        K, m = sessions, self.m
        Q = np.broadcast_to(self.initial_q(), (K, 2, self.n_states, m)).copy()
        s = rng.integers(self.n_states, size=K)
        ks = np.arange(K)
        rec_t, rec_p = [], []
        acc = np.zeros((K, 2))
        for t in range(periods):
            eps = np.exp(-self.beta * t)
            greedy = Q[ks[:, None], np.arange(2)[None, :], s[:, None]].argmax(axis=-1)  # (K, 2)
            explore = rng.random((K, 2)) < eps
            a = np.where(explore, rng.integers(m, size=(K, 2)), greedy)
            s_next = a[:, 0] * m + a[:, 1]
            pay = np.stack([self.profit[0, a[:, 0], a[:, 1]], self.profit[1, a[:, 0], a[:, 1]]], -1)
            best_next = Q[ks[:, None], np.arange(2)[None, :], s_next[:, None]].max(axis=-1)
            idx = (ks[:, None], np.arange(2)[None, :], s[:, None], a)
            Q[idx] += self.alpha * (pay + self.delta * best_next - Q[idx])
            s = s_next
            acc += self.prices[a]
            if (t + 1) % record_every == 0:
                rec_t.append(t + 1)
                rec_p.append(acc / record_every)
                acc[:] = 0
        return dict(Q=Q, state=s, t=np.array(rec_t), prices=np.array(rec_p))

    def greedy(self, Q: np.ndarray) -> np.ndarray:
        """Greedy action of each firm in each state: shape (..., 2, n_states)."""
        return Q.argmax(axis=-1)

    def play(self, Q: np.ndarray, s0: int, periods: int = 60, deviate_at: int | None = None) -> np.ndarray:
        """Greedy play of one trained pair from state ``s0``; optionally firm 1 deviates once.

        The deviation is firm 1's static best reply to firm 2's greedy price in
        that period. Returns the price indices, shape (periods, 2).
        """
        G = self.greedy(Q)
        s = s0
        out = []
        for t in range(periods):
            a1, a2 = G[0, s], G[1, s]
            if deviate_at is not None and t == deviate_at:
                a1 = int(np.argmax(self.profit[0, :, a2]))
            out.append((a1, a2))
            s = a1 * self.m + a2
        return np.array(out)

    def limit_cycle(self, Q: np.ndarray, s0: int) -> list[int]:
        """States of the cycle that greedy play falls into from ``s0``."""
        G = self.greedy(Q)
        seen: dict[int, int] = {}
        path = []
        s = s0
        while s not in seen:
            seen[s] = len(path)
            path.append(s)
            s = int(G[0, s]) * self.m + int(G[1, s])
        return path[seen[s]:]

    def cycle_profit_gain(self, Q: np.ndarray, s0: int) -> float:
        cyc = self.limit_cycle(Q, s0)
        a1 = np.array([c // self.m for c in cyc])
        a2 = np.array([c % self.m for c in cyc])
        pay = (self.profit[0, a1, a2] + self.profit[1, a1, a2]) / 2
        return float(self.profit_gain(pay.mean()))
