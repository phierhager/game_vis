import numpy as np
import pytest

from gamevis import markets as mk


def test_calvano_baseline_prices():
    m = mk.LogitBertrand()
    assert np.allclose(m.nash(), 1.4729, atol=1e-4)
    assert np.allclose(m.collusive(), 1.9250, atol=1e-4)


@pytest.mark.parametrize("model", [mk.LogitBertrand(), mk.LinearBertrand(), mk.Cournot()])
def test_marginal_profit_is_the_own_price_derivative(model):
    rng = np.random.default_rng(0)
    p = model.lo + (model.hi - model.lo) * (0.2 + 0.6 * rng.random((20, 2)))
    g1, g2 = model.marginal_profit(p[:, 0], p[:, 1])
    h = 1e-6
    n1 = (model.profit(p[:, 0] + h, p[:, 1])[0] - model.profit(p[:, 0] - h, p[:, 1])[0]) / (2 * h)
    n2 = (model.profit(p[:, 0], p[:, 1] + h)[1] - model.profit(p[:, 0], p[:, 1] - h)[1]) / (2 * h)
    assert np.allclose(g1, n1, atol=1e-6) and np.allclose(g2, n2, atol=1e-6)


def test_closed_forms():
    lin = mk.LinearBertrand(alpha=1.0, beta=1.0, gamma=0.5, cost=0.1)
    assert np.allclose(lin.nash(), lin.nash_closed_form(), atol=1e-6)
    assert np.allclose(lin.collusive(), lin.collusive_closed_form(), atol=1e-6)
    c = mk.Cournot(a=1.0, b=1.0, c=0.1)
    assert np.allclose(c.nash(), c.nash_closed_form(), atol=1e-6)
    assert np.allclose(c.collusive(), c.collusive_closed_form(), atol=1e-6)


def test_gradient_play_finds_nash_not_the_cartel():
    m = mk.LogitBertrand()
    starts = np.array([[1.05, 2.35], [2.3, 2.3], [1.925, 1.925]])
    end = m.simulate(starts, t_max=300, dt=0.05)[-1]
    assert np.allclose(end, m.nash()[0], atol=1e-4)


def test_sympathy_moves_the_equilibrium_to_the_cartel():
    m = mk.LogitBertrand()
    prices = [m.equilibrium(lam)[0] for lam in (0.0, 0.5, 1.0)]
    assert prices[0] < prices[1] < prices[2]
    assert np.isclose(prices[2], m.collusive()[0])


def test_edgeworth_cycles_and_bertrand_paradox():
    e = mk.Edgeworth(capacity=0.5, cost=0.1, tick=0.01)
    path = e.simulate((0.9, 0.9), 200)[100:, 0]
    assert path.max() - path.min() > 0.05  # never settles
    assert path.max() <= e.residual_price() + 1e-9
    paradox = mk.Edgeworth(capacity=1.0, cost=0.1, tick=0.01).simulate((0.9, 0.9), 200)
    assert np.allclose(paradox[-1], 0.11)
