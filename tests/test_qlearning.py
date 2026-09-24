import numpy as np

from gamevis.qlearning import QPricing


def test_grid_profits_and_initial_q():
    qp = QPricing()
    span = qp.p_monopoly - qp.p_nash
    assert np.isclose(qp.prices[0], qp.p_nash - 0.1 * span)
    assert np.isclose(qp.prices[-1], qp.p_monopoly + 0.1 * span)
    a1, a2 = 3, 11
    assert np.isclose(qp.profit[0, a1, a2], qp.market.profit(qp.prices[a1], qp.prices[a2])[0])
    assert np.isclose(qp.profit[1, a1, a2], qp.market.profit(qp.prices[a1], qp.prices[a2])[1])
    Q = qp.initial_q()
    assert np.allclose(Q[0, 0], qp.profit[0].mean(axis=1) / (1 - qp.delta))
    assert np.allclose(Q[0, 7], Q[0, 0])  # the same in every state


def test_q_learners_price_above_nash():
    qp = QPricing(beta=1e-4)
    res = qp.run(sessions=4, periods=60_000, seed=3, record_every=5000)
    gains = [qp.cycle_profit_gain(res["Q"][k], int(res["state"][k])) for k in range(4)]
    assert np.mean(gains) > 0.3
    assert res["prices"][-1].mean() > qp.p_nash + 0.05


def test_deviation_is_a_static_best_reply():
    qp = QPricing(beta=1e-4)
    res = qp.run(sessions=1, periods=20_000, seed=0, record_every=5000)
    Q = res["Q"][0]
    s0 = qp.limit_cycle(Q, int(res["state"][0]))[0]
    path = qp.play(Q, s0, periods=6, deviate_at=2)
    a2 = path[2, 1]
    assert path[2, 0] == np.argmax(qp.profit[0, :, a2])
