"""Sender-receiver signaling games and their first-order learning dynamics.

Nature draws a state ``t`` with prior ``pi``. The sender sees ``t`` and sends a
message ``m`` with probability ``S[t, m]``; the receiver sees only ``m`` and
takes action ``a`` with probability ``R[m, a]``. Payoffs depend on the state
and the action, and the sender may pay a cost ``C[t, m]`` for the message.

Each row of ``S`` (one per state) and of ``R`` (one per message) is a separate
probability simplex, so the joint strategy space is a product of simplices:
for 2 states x 2 messages x 2 actions it is the 4-cube
``(S[t1,m1], S[t2,m1], R[m1,a1], R[m2,a1])``. That is why the 2D phase
portraits of 2x2 normal-form games stop working: the flow is 4-dimensional.

"Everyone follows the first-order signal" means every row moves along its own
payoff gradient (the ex-ante marginal payoff of each option):

    dU_S/dS[t, m] = pi[t] * (sum_a R[m, a] u_S[t, a] - C[t, m])
    dU_R/dR[m, a] = sum_t pi[t] * S[t, m] * u_R[t, a]
                  = P(m) * E[u_R(t, a) | m]

Two consequences shape all the pictures. A rare state learns slowly (its
signal is scaled by ``pi[t]``), and a message nobody sends gives the receiver
no signal at all (``P(m) = 0``), which is what freezes pooling outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import simplex
from .dynamics import SMOOTH_RULES, integrate

# ---------------------------------------------------------------------------
# The game
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalingGame:
    prior: np.ndarray
    sender_utility: np.ndarray  # (n_states, n_actions)
    receiver_utility: np.ndarray  # (n_states, n_actions)
    message_cost: np.ndarray  # (n_states, n_messages)
    name: str = "Signaling game"
    state_labels: tuple[str, ...] = ()
    message_labels: tuple[str, ...] = ()
    action_labels: tuple[str, ...] = ()
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        prior = np.asarray(self.prior, dtype=float)
        object.__setattr__(self, "prior", prior / prior.sum())
        for name in ("sender_utility", "receiver_utility", "message_cost"):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=float))
        n, k, l = self.n_states, self.n_messages, self.n_actions
        if self.sender_utility.shape != (n, l) or self.receiver_utility.shape != (n, l):
            raise ValueError("utilities must have shape (n_states, n_actions)")
        if not self.state_labels:
            object.__setattr__(self, "state_labels", tuple(f"t{i + 1}" for i in range(n)))
        if not self.message_labels:
            object.__setattr__(self, "message_labels", tuple(f"m{i + 1}" for i in range(k)))
        if not self.action_labels:
            object.__setattr__(self, "action_labels", tuple(f"a{i + 1}" for i in range(l)))

    @property
    def n_states(self) -> int:
        return self.prior.shape[0]

    @property
    def n_messages(self) -> int:
        return self.message_cost.shape[1]

    @property
    def n_actions(self) -> int:
        return self.sender_utility.shape[1]

    @property
    def common_interest(self) -> bool:
        return bool(
            np.allclose(self.sender_utility, self.receiver_utility)
            and np.allclose(self.message_cost, 0.0)
        )

    # -- first-order signals ------------------------------------------------

    def sender_signal(self, R: np.ndarray) -> np.ndarray:
        """Gradient of the sender's ex-ante payoff w.r.t. ``S``; shape (..., n, k)."""
        value = np.einsum("ta,...ma->...tm", self.sender_utility, R)
        return self.prior[:, None] * (value - self.message_cost)

    def receiver_signal(self, S: np.ndarray) -> np.ndarray:
        """Gradient of the receiver's ex-ante payoff w.r.t. ``R``; shape (..., k, l)."""
        weighted = self.prior[:, None] * self.receiver_utility
        return np.einsum("...tm,ta->...ma", S, weighted)

    def payoffs(self, S: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Ex-ante expected payoffs ``(U_sender, U_receiver)``."""
        u_s = np.sum(S * self.sender_signal(R), axis=(-2, -1))
        u_r = np.sum(R * self.receiver_signal(S), axis=(-2, -1))
        return u_s, u_r

    def velocity(
        self,
        S: np.ndarray,
        R: np.ndarray,
        rule: str = "replicator",
        rates: tuple[float, float] = (1.0, 1.0),
        temperature: float = 0.05,
        exploration: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        dS = rates[0] * simplex.field(rule, S, self.sender_signal(R), temperature)
        dR = rates[1] * simplex.field(rule, R, self.receiver_signal(S), temperature)
        if exploration:
            dS = dS + exploration * (1.0 / self.n_messages - S)
            dR = dR + exploration * (1.0 / self.n_actions - R)
        return dS, dR

    # -- information --------------------------------------------------------

    def channel(self, S: np.ndarray, R: np.ndarray) -> np.ndarray:
        """``P(a | t)``: the state-to-action channel ``S @ R``."""
        return np.einsum("...tm,...ma->...ta", S, R)

    def mutual_information(self, P: np.ndarray) -> np.ndarray:
        """``I(T; Y)`` in bits for a channel ``P[t, y] = P(y | t)``."""
        joint = self.prior[:, None] * P
        marginal = joint.sum(axis=-2, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(joint > 0, P / np.where(marginal > 0, marginal, 1.0), 1.0)
            terms = np.where(joint > 0, joint * np.log2(ratio), 0.0)
        return terms.sum(axis=(-2, -1))

    def entropy(self) -> float:
        p = self.prior[self.prior > 0]
        return float(-(p * np.log2(p)).sum())

    # -- 2 states x 2 messages x 2 actions: four numbers ---------------------

    @property
    def is_binary(self) -> bool:
        return self.n_states == 2 and self.n_messages == 2 and self.n_actions == 2

    def binary_velocity(
        self,
        X: np.ndarray,
        rule: str = "replicator",
        rates: tuple[float, float] = (1.0, 1.0),
        temperature: float = 0.05,
        exploration: float = 0.0,
    ) -> np.ndarray:
        """Velocity in the coordinates ``X = (S[t1,m1], S[t2,m1], R[m1,a1], R[m2,a1])``.

        Same flow as :meth:`velocity`, written out for the 2x2x2 case where
        every agent's strategy is one probability.
        """
        x1, x2, y1, y2 = np.moveaxis(np.asarray(X, dtype=float), -1, 0)
        pi = self.prior
        du_s = self.sender_utility[:, 0] - self.sender_utility[:, 1]
        du_r = self.receiver_utility[:, 0] - self.receiver_utility[:, 1]
        dc = self.message_cost[:, 0] - self.message_cost[:, 1]
        sep = y1 - y2
        d1 = pi[0] * (sep * du_s[0] - dc[0])  # advantage of m1 for the t1 sender
        d2 = pi[1] * (sep * du_s[1] - dc[1])
        e1 = pi[0] * x1 * du_r[0] + pi[1] * x2 * du_r[1]  # advantage of a1 after m1
        e2 = pi[0] * (1 - x1) * du_r[0] + pi[1] * (1 - x2) * du_r[1]
        f = simplex.binary_field
        v = np.stack(
            [
                rates[0] * f(rule, x1, d1, temperature),
                rates[0] * f(rule, x2, d2, temperature),
                rates[1] * f(rule, y1, e1, temperature),
                rates[1] * f(rule, y2, e2, temperature),
            ],
            -1,
        )
        if exploration:
            v = v + exploration * (0.5 - np.asarray(X))
        return v


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


def lewis(n: int = 2, prior: float | np.ndarray | None = None, n_messages: int | None = None) -> SignalingGame:
    """Lewis's game of pure common interest: both get 1 when the act fits the state.

    ``prior`` may be the probability of the first state (for ``n = 2``) or a
    full vector. With ``n_messages < n`` there are not enough words to go
    round and some states must share one.
    """
    k = n if n_messages is None else n_messages
    if prior is None:
        p = np.full(n, 1.0 / n)
    elif np.isscalar(prior):
        if n != 2:
            raise ValueError("a scalar prior is only meaningful for two states")
        p = np.array([float(prior), 1.0 - float(prior)])
    else:
        p = np.asarray(prior, dtype=float)
    u = np.eye(n)
    return SignalingGame(
        prior=p,
        sender_utility=u,
        receiver_utility=u,
        message_cost=np.zeros((n, k)),
        name=f"Lewis signaling game ({n}x{k}x{n})",
        action_labels=tuple(f"a{i + 1}" for i in range(n)),
        notes="Pure common interest: sender and receiver both want the act to fit the state.",
    )


def costly_signaling(
    prior_high: float = 0.4,
    cost_high: float = 0.3,
    cost_low: float = 0.6,
    benefit: float = 1.0,
) -> SignalingGame:
    """A handicap (Spence/Zahavi) game with a conflict of interest.

    States: a *high* and a *low* type. Messages: a costly *display* or *no
    display*. Actions: *reward* or *withhold*. Both sender types want the
    reward (worth ``benefit``); the receiver only wants to reward the high
    type. The display costs ``cost_high`` for the high type and ``cost_low``
    for the low type.

    * ``cost_low > benefit``: faking does not pay, honest separation is stable.
    * ``cost_high < cost_low < benefit`` (and a minority of high types):
      a hybrid equilibrium where low types fake part of the time. First-order
      learners cycle around it: mimicry erodes trust, lost trust stops
      mimicry, restored trust invites mimicry again.
    """
    return SignalingGame(
        prior=np.array([prior_high, 1.0 - prior_high]),
        sender_utility=np.array([[benefit, 0.0], [benefit, 0.0]]),
        receiver_utility=np.array([[1.0, 0.0], [0.0, 1.0]]),
        message_cost=np.array([[cost_high, 0.0], [cost_low, 0.0]]),
        name="Costly signaling (handicap principle)",
        state_labels=("high", "low"),
        message_labels=("display", "no display"),
        action_labels=("reward", "withhold"),
        notes="Both sender types want the reward; only the high type deserves it.",
        extra=dict(prior_high=prior_high, cost_high=cost_high, cost_low=cost_low, benefit=benefit),
    )


def hybrid_equilibrium(game: SignalingGame) -> tuple[np.ndarray, np.ndarray] | None:
    """The partially honest equilibrium of :func:`costly_signaling`, if it exists.

    The high type always displays; the low type displays with the probability
    that makes the receiver indifferent after a display; the receiver rewards a
    display with the probability that makes the low type indifferent.
    """
    p = game.extra
    if not p:
        return None
    ph, ch, cl, b = p["prior_high"], p["cost_high"], p["cost_low"], p["benefit"]
    mimic = ph / (1.0 - ph)  # makes the posterior after a display exactly 1/2
    trust = cl / b  # makes the low type indifferent about displaying
    if not (0.0 < mimic < 1.0 and 0.0 < trust < 1.0 and ch <= cl):
        return None
    S = np.array([[1.0, 0.0], [mimic, 1.0 - mimic]])
    R = np.array([[trust, 1.0 - trust], [0.0, 1.0]])
    return S, R


# ---------------------------------------------------------------------------
# Coordinates for pictures
# ---------------------------------------------------------------------------


def square_coordinates(S: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For 2x2x2 games: sender point ``(S[t1,m1], S[t2,m1])`` and receiver point ``(R[m1,a1], R[m2,a1])``."""
    return np.stack([S[..., 0, 0], S[..., 1, 0]], -1), np.stack([R[..., 0, 0], R[..., 1, 0]], -1)


def meaning_coordinates(S: np.ndarray, R: np.ndarray) -> np.ndarray:
    """For 2x2x2 games: (sender separation, receiver separation).

    Sender separation ``S[t1,m1] - S[t2,m1]`` is +1 when m1 means t1, -1 when
    m1 means t2 and 0 when the message carries no information. Receiver
    separation ``R[m1,a1] - R[m2,a1]`` says how differently the two messages
    are read. With equal priors the common payoff is exactly
    ``(1 + dx * dy) / 2``: a saddle.
    """
    dx = S[..., 0, 0] - S[..., 1, 0]
    dy = R[..., 0, 0] - R[..., 1, 0]
    return np.stack([dx, dy], -1)


def from_square_coordinates(sender_xy, receiver_xy) -> tuple[np.ndarray, np.ndarray]:
    sx = np.asarray(sender_xy, dtype=float)
    rx = np.asarray(receiver_xy, dtype=float)
    S = np.stack([np.stack([sx[..., 0], 1 - sx[..., 0]], -1), np.stack([sx[..., 1], 1 - sx[..., 1]], -1)], -2)
    R = np.stack([np.stack([rx[..., 0], 1 - rx[..., 0]], -1), np.stack([rx[..., 1], 1 - rx[..., 1]], -1)], -2)
    return S, R


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


@dataclass
class Trajectory:
    game: SignalingGame
    t: np.ndarray  # (T,)
    S: np.ndarray  # (T, ..., n, k)
    R: np.ndarray  # (T, ..., k, l)

    def payoffs(self) -> tuple[np.ndarray, np.ndarray]:
        return self.game.payoffs(self.S, self.R)

    def information(self) -> np.ndarray:
        """``I(T; A)`` in bits along the trajectory."""
        return self.game.mutual_information(self.game.channel(self.S, self.R))

    def sender_information(self) -> np.ndarray:
        """``I(T; M)`` in bits along the trajectory."""
        return self.game.mutual_information(self.S)


def random_start(game: SignalingGame, rng: np.random.Generator, size: int | tuple = ()) -> tuple[np.ndarray, np.ndarray]:
    size = (size,) if isinstance(size, int) else tuple(size)
    S = simplex.random_simplex(rng, size + (game.n_states,), game.n_messages)
    R = simplex.random_simplex(rng, size + (game.n_messages,), game.n_actions)
    return S, R


def simulate(
    game: SignalingGame,
    S0: np.ndarray,
    R0: np.ndarray,
    rule: str = "replicator",
    t_max: float = 100.0,
    dt: float = 0.05,
    rates: tuple[float, float] = (1.0, 1.0),
    temperature: float = 0.05,
    exploration: float = 0.0,
    record_every: int = 10,
) -> Trajectory:
    """Integrate the learning dynamics from ``(S0, R0)`` (batched over leading axes)."""

    def f(y):
        return game.velocity(y[0], y[1], rule, rates, temperature, exploration)

    times, states = integrate(f, (S0, R0), t_max, dt, smooth=rule in SMOOTH_RULES, record_every=record_every)
    S = np.stack([s[0] for s in states])
    R = np.stack([s[1] for s in states])
    return Trajectory(game, times, S, R)


def simulate_binary(
    game: SignalingGame,
    X0: np.ndarray,
    rule: str = "replicator",
    t_max: float = 100.0,
    dt: float = 0.05,
    rates: tuple[float, float] = (1.0, 1.0),
    temperature: float = 0.05,
    exploration: float = 0.0,
    record_every: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast path for 2x2x2 games in the four coordinates of :meth:`SignalingGame.binary_velocity`.

    Returns ``(times, X)`` with ``X`` of shape ``(T, ..., 4)``.
    """
    if not game.is_binary:
        raise ValueError("simulate_binary needs a 2x2x2 game")

    def f(x):
        return game.binary_velocity(x, rule, rates, temperature, exploration)

    X = np.array(X0, dtype=float)
    n_steps = int(round(t_max / dt))
    times, out = [0.0], [X.copy()]
    smooth = rule in SMOOTH_RULES
    for i in range(1, n_steps + 1):
        if smooth:
            k1 = f(X)
            k2 = f(X + dt / 2 * k1)
            k3 = f(X + dt / 2 * k2)
            k4 = f(X + dt * k3)
            X = X + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            X = X + dt * f(X)
        np.clip(X, 0.0, 1.0, out=X)
        if i % record_every == 0 or i == n_steps:
            times.append(i * dt)
            out.append(X.copy())
    return np.array(times), np.stack(out)


def binary_to_matrices(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    return from_square_coordinates(X[..., :2], X[..., 2:])


def matrices_to_binary(S: np.ndarray, R: np.ndarray) -> np.ndarray:
    return np.stack([S[..., 0, 0], S[..., 1, 0], R[..., 0, 0], R[..., 1, 0]], -1)


# ---------------------------------------------------------------------------
# Where do learning runs end up?
# ---------------------------------------------------------------------------

OUTCOME_CODES = ("m1 means t1", "m1 means t2", "no information", "cycling or partial")


def outcome_labels(game: SignalingGame) -> tuple[str, ...]:
    """Outcome names in the game's own words (e.g. "display means high")."""
    t, m = game.state_labels, game.message_labels
    return (f"{m[0]} means {t[0]}", f"{m[0]} means {t[1]}", "no information", "cycling or partial")


def classify_binary(X_tail: np.ndarray, separated: float = 0.9, silent: float = 0.1, swing: float = 0.05) -> np.ndarray:
    """Label 2x2x2 runs from the tail of their trajectories, shape ``(T_tail, ..., 4)``.

    Uses the meaning coordinates (sender separation ``dx``, receiver separation
    ``dy``): a run that still swings by more than ``swing`` is cycling; one
    where either side stopped discriminating (``|dx|`` or ``|dy|`` below
    ``silent``) carries no information; ``dx, dy`` both near +1 means m1 has
    come to mean t1, both near -1 that it means t2. Codes index
    :data:`OUTCOME_CODES`.
    """
    X_tail = np.asarray(X_tail, dtype=float)
    dx = X_tail[..., 0] - X_tail[..., 1]
    dy = X_tail[..., 2] - X_tail[..., 3]
    spread = np.maximum(dx.max(0) - dx.min(0), dy.max(0) - dy.min(0))
    fx, fy = dx[-1], dy[-1]
    codes = np.full(fx.shape, 3, dtype=int)
    still = spread < swing
    codes[still & (np.minimum(np.abs(fx), np.abs(fy)) < silent)] = 2
    codes[still & (fx > separated) & (fy > separated)] = 0
    codes[still & (fx < -separated) & (fy < -separated)] = 1
    return codes


def classify(traj: Trajectory, tail: float = 0.2, **kwargs) -> np.ndarray:
    """Label every run of a (batched) trajectory; see :func:`classify_binary`.

    For games larger than 2x2x2 the codes are 0 = fully informative
    (``I(T;A) = H(T)``), 2 = no information and 3 = partial.
    """
    start = int(len(traj.t) * (1 - tail))
    if traj.game.is_binary:
        return classify_binary(matrices_to_binary(traj.S[start:], traj.R[start:]), **kwargs)
    info = traj.information()[start:] / max(traj.game.entropy(), 1e-12)
    mean = info.mean(axis=0)
    spread = info.max(axis=0) - info.min(axis=0)
    codes = np.full(mean.shape, 3, dtype=int)
    codes[(spread < 0.05) & (mean > 0.95)] = 0
    codes[(spread < 0.05) & (mean < 0.05)] = 2
    return codes


def distinguished_states(game: SignalingGame, S: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Communicative success times the number of states, for Lewis games.

    With a uniform prior this is the number of states that get their own
    meaning: n for a signaling system, fewer for partial pooling, 1 for total
    pooling.
    """
    u = game.payoffs(S, R)[1]
    return u * game.n_states


def outcome_fractions(codes: np.ndarray, n_codes: int = 4) -> np.ndarray:
    counts = np.bincount(np.asarray(codes).ravel(), minlength=n_codes).astype(float)
    return counts / counts.sum()


def basin_fractions(
    game: SignalingGame,
    rule: str = "replicator",
    n_samples: int = 2000,
    t_max: float = 400.0,
    dt: float = 0.1,
    rates: tuple[float, float] = (1.0, 1.0),
    temperature: float = 0.05,
    exploration: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Share of uniformly random starting points ending in each outcome (2x2x2 games)."""
    X0 = np.random.default_rng(seed).random((n_samples, 4))
    _, X = simulate_binary(game, X0, rule, t_max, dt, rates, temperature, exploration, record_every=max(1, int(1 / dt)))
    tail = X[int(len(X) * 0.8):]
    return outcome_fractions(classify_binary(tail))
