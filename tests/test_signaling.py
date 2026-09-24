import numpy as np
import pytest

from gamevis import signaling as sg
from gamevis import simplex


def random_game(rng, n=3, k=2, l=4):
    return sg.SignalingGame(
        prior=rng.random(n) + 0.1,
        sender_utility=rng.normal(size=(n, l)),
        receiver_utility=rng.normal(size=(n, l)),
        message_cost=rng.random((n, k)) * 0.3,
    )


def test_signals_are_payoff_gradients():
    rng = np.random.default_rng(1)
    game = random_game(rng)
    S, R = sg.random_start(game, rng)
    us, ur = game.payoffs(S, R)
    gs, gr = game.sender_signal(R), game.receiver_signal(S)
    h = 1e-7
    for t in range(game.n_states):
        for m in range(game.n_messages):
            S2 = S.copy()
            S2[t, m] += h
            assert np.isclose((game.payoffs(S2, R)[0] - us) / h, gs[t, m], atol=1e-5)
    for m in range(game.n_messages):
        for a in range(game.n_actions):
            R2 = R.copy()
            R2[m, a] += h
            assert np.isclose((game.payoffs(S, R2)[1] - ur) / h, gr[m, a], atol=1e-5)


@pytest.mark.parametrize("rule", simplex.RULES)
def test_binary_fast_path_matches_general_velocity(rule):
    rng = np.random.default_rng(2)
    for game in (sg.lewis(2, 0.7), sg.costly_signaling(), random_game(rng, 2, 2, 2)):
        X = rng.random((30, 4))
        S, R = sg.binary_to_matrices(X)
        fast = game.binary_velocity(X, rule, rates=(1.5, 0.5), exploration=0.02)
        slow = sg.matrices_to_binary(*game.velocity(S, R, rule, rates=(1.5, 0.5), exploration=0.02))
        assert np.allclose(fast, slow, atol=1e-12)


def test_equal_prior_lewis_payoff_is_a_saddle():
    rng = np.random.default_rng(3)
    game = sg.lewis(2)
    X = rng.random((100, 4))
    S, R = sg.binary_to_matrices(X)
    dx, dy = np.moveaxis(sg.meaning_coordinates(S, R), -1, 0)
    assert np.allclose(game.payoffs(S, R)[1], (1 + dx * dy) / 2)


def test_signaling_systems_are_rest_points_with_full_information():
    game = sg.lewis(3)
    S = np.eye(3)[[2, 0, 1]]
    R = S.T.copy()
    for rule in ("replicator", "projection", "softmax_pg"):
        dS, dR = game.velocity(S, R, rule)
        assert np.allclose(dS, 0) and np.allclose(dR, 0)
    assert np.isclose(game.payoffs(S, R)[1], 1.0)
    assert np.isclose(game.mutual_information(game.channel(S, R)), np.log2(3))
    pooled = np.tile([1.0, 0.0, 0.0], (3, 1))
    assert np.isclose(game.mutual_information(pooled), 0.0)


def test_common_interest_payoff_never_decreases():
    rng = np.random.default_rng(4)
    game = sg.lewis(3)
    S0, R0 = sg.random_start(game, rng, 20)
    for rule in ("replicator", "projection", "softmax_pg"):
        tr = sg.simulate(game, S0, R0, rule, t_max=30, dt=0.05, record_every=5)
        u = tr.payoffs()[1]
        assert np.all(np.diff(u, axis=0) > -1e-9)


def test_pooling_basin_grows_with_prior_skew():
    even = sg.basin_fractions(sg.lewis(2, 0.5), n_samples=600, t_max=300)
    skewed = sg.basin_fractions(sg.lewis(2, 0.9), n_samples=600, t_max=300)
    assert even[2] < 0.02
    assert skewed[2] > 0.1
    assert np.isclose(even[0], even[1], atol=0.08)


def test_fast_receivers_trap_language_in_pooling():
    game = sg.lewis(2, 0.8)
    slow_ears = sg.basin_fractions(game, n_samples=600, t_max=300, rates=(4.0, 1.0))
    fast_ears = sg.basin_fractions(game, n_samples=600, t_max=300, rates=(1.0, 4.0))
    assert fast_ears[2] > slow_ears[2] + 0.15


def test_costly_signaling_hybrid_is_a_rest_point_circled_by_orbits():
    game = sg.costly_signaling(prior_high=0.4, cost_high=0.3, cost_low=0.6)
    S, R = sg.hybrid_equilibrium(game)
    dS, dR = game.velocity(S, R, "replicator")
    assert np.allclose(dS, 0) and np.allclose(dR, 0)
    X0 = sg.matrices_to_binary(S, R) + np.array([-0.03, -0.1, -0.1, 0.03])
    _, X = sg.simulate_binary(game, X0, t_max=400, dt=0.05, record_every=10)
    mimic = X[len(X) // 2:, 1]
    # Mimicry keeps swinging instead of settling.
    assert mimic.max() - mimic.min() > 0.2
    assert sg.classify_binary(X[-len(X) // 5:])[()] == 3
    # Expensive faking makes honest signaling the outcome.
    honest = sg.costly_signaling(prior_high=0.4, cost_high=0.3, cost_low=1.3)
    assert sg.hybrid_equilibrium(honest) is None
    assert sg.basin_fractions(honest, n_samples=400, t_max=300)[0] > 0.7


def test_classify_binary_labels():
    tail = np.array(
        [
            [[1, 0, 1, 0], [0, 1, 0, 1], [0.6, 0.6, 1, 1], [1, 0, 1, 1]],
            [[1, 0, 1, 0], [0, 1, 0, 1], [0.6, 0.6, 1, 1], [1, 0, 1, 1]],
        ],
        dtype=float,
    )
    assert list(sg.classify_binary(tail)) == [0, 1, 2, 2]
    swinging = np.array([[[0.9, 0.2, 0.7, 0.0]], [[0.9, 0.5, 0.4, 0.0]]])
    assert list(sg.classify_binary(swinging)) == [3]


def test_three_state_lewis_game_mostly_finds_signaling_systems():
    rng = np.random.default_rng(5)
    game = sg.lewis(3)
    S0, R0 = sg.random_start(game, rng, 150)
    tr = sg.simulate(game, S0, R0, "replicator", t_max=400, dt=0.1, record_every=100)
    k = sg.distinguished_states(game, tr.S[-1], tr.R[-1])
    full = np.mean(k > 2.95)
    partial = np.mean(np.abs(k - 2) < 0.05)
    assert full > 0.85 and full + partial > 0.99
