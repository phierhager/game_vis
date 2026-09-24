import numpy as np
import pytest

from gamevis import simplex


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.mark.parametrize("rule", simplex.RULES)
def test_fields_are_tangent_to_the_simplex(rule, rng):
    x = simplex.random_simplex(rng, (40,), 4)
    g = rng.normal(size=(40, 4))
    v = simplex.field(rule, x, g)
    assert np.allclose(v.sum(-1), 0.0, atol=1e-12)


@pytest.mark.parametrize("rule", simplex.RULES)
def test_binary_field_matches_general_field(rule, rng):
    x = rng.random(200)
    g = rng.normal(size=(200, 2))
    general = simplex.field(rule, np.stack([x, 1 - x], -1), g)[:, 0]
    binary = simplex.binary_field(rule, x, g[:, 0] - g[:, 1])
    assert np.allclose(general, binary, atol=1e-12)


@pytest.mark.parametrize("rule", ["replicator", "projection", "softmax_pg", "logit"])
def test_discrete_steps_trace_the_ode(rule, rng):
    x = simplex.random_simplex(rng, (30,), 3) * 0.8 + 0.2 / 3  # stay off the faces
    g = rng.normal(size=(30, 3))
    eta = 1e-6
    step = (simplex.discrete_step(rule, x, g, eta) - x) / eta
    assert np.allclose(step, simplex.field(rule, x, g), atol=1e-4)


@pytest.mark.parametrize("rule", ["replicator", "projection", "softmax_pg", "logit", "best_response"])
def test_binary_discrete_step_matches_general(rule, rng):
    x = rng.random(50) * 0.9 + 0.05
    g = rng.normal(size=(50, 2))
    general = simplex.discrete_step(rule, np.stack([x, 1 - x], -1), g, 0.3)[:, 0]
    binary = simplex.binary_discrete_step(rule, x, g[:, 0] - g[:, 1], 0.3)
    assert np.allclose(general, binary, atol=1e-10)


def test_softmax_pg_is_replicator_twice(rng):
    x = simplex.random_simplex(rng, (10,), 3)
    g = rng.normal(size=(10, 3))
    # Push the logit gradient J g through the softmax Jacobian J.
    expected = []
    for xi, gi in zip(x, g):
        J = np.diag(xi) - np.outer(xi, xi)
        expected.append(J @ (J @ gi))
    assert np.allclose(simplex.softmax_pg(x, g), np.array(expected))


def test_projection_onto_simplex(rng):
    y = rng.normal(size=(100, 5)) * 2
    p = simplex.project_to_simplex(y)
    assert np.all(p >= 0)
    assert np.allclose(p.sum(-1), 1)
    assert np.allclose(simplex.project_to_simplex(p), p)
    # It is the closest point: no random simplex point is closer.
    for yi, pi in zip(y[:10], p[:10]):
        others = simplex.random_simplex(rng, (2000,), 5)
        assert np.linalg.norm(yi - pi) <= np.linalg.norm(others - yi, axis=1).min() + 1e-12


def test_projection_dynamics_on_faces():
    # At a vertex, only the coordinates whose signal beats the mean may grow.
    x = np.array([1.0, 0.0, 0.0])
    v = simplex.projection(x, np.array([0.0, 1.0, -5.0]))
    assert np.allclose(v, [-0.5, 0.5, 0.0])
    # Interior: plain mean-centred gradient.
    x = np.array([0.2, 0.3, 0.5])
    g = np.array([1.0, 2.0, 6.0])
    assert np.allclose(simplex.projection(x, g), g - g.mean())
    # A coordinate at zero with a bad signal stays put.
    x = np.array([0.5, 0.5, 0.0])
    v = simplex.projection(x, np.array([1.0, 0.0, -1.0]))
    assert v[2] == 0.0 and np.isclose(v.sum(), 0.0)


def test_best_response_keeps_ties_still():
    x = np.array([0.3, 0.7])
    assert np.allclose(simplex.best_response(x, np.array([1.0, 1.0])), 0.0)
    assert np.allclose(simplex.binary_field("best_response", 0.3, 0.0), 0.0)
