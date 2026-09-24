import numpy as np
import pytest

from gamevis import normal_form as nf


def test_classic_equilibria():
    P = nf.BIMATRIX_PRESETS
    assert P["prisoners_dilemma"].nash_equilibria() == [(0.0, 0.0)]
    stag = P["stag_hunt"].nash_equilibria()
    assert (1.0, 1.0) in stag and (0.0, 0.0) in stag
    assert np.allclose(stag[-1], (0.75, 0.75))
    assert np.allclose(P["chicken"].nash_equilibria()[-1], (0.9, 0.9))
    assert np.allclose(P["matching_pennies"].nash_equilibria(), [(0.5, 0.5)])
    assert np.allclose(P["battle_of_sexes"].nash_equilibria()[-1], (0.6, 0.4))


@pytest.mark.parametrize("key,expected", [
    ("stag_hunt", ["sink", "sink", "saddle"]),
    ("chicken", ["sink", "sink", "saddle"]),
    ("matching_pennies", ["center"]),
])
def test_equilibrium_stability(key, expected):
    g = nf.BIMATRIX_PRESETS[key]
    assert [g.equilibrium_stability(e) for e in g.nash_equilibria()] == expected


def test_matching_pennies_continuous_orbits_close_discrete_spirals_out():
    g = nf.BIMATRIX_PRESETS["matching_pennies"]

    def kl_energy(p, q):  # conserved by the continuous replicator dynamics here
        return -(0.5 * np.log(p) + 0.5 * np.log(1 - p) + 0.5 * np.log(q) + 0.5 * np.log(1 - q))

    traj = g.simulate(0.7, 0.5, "replicator", t_max=20, dt=0.001)
    e = kl_energy(traj[:, 0], traj[:, 1])
    assert np.ptp(e) < 1e-6
    p, q = 0.7, 0.5
    start = kl_energy(p, q)
    for _ in range(300):
        p, q = g.discrete_step(p, q, "replicator", eta=0.1)
    assert kl_energy(p, q) > start + 0.1


def test_symmetric_2x2_regions():
    assert nf.ts_region(T=1.5, S=-0.5) == "Prisoner's Dilemma"
    assert nf.ts_region(T=0.5, S=-0.5) == "Stag Hunt"
    assert nf.ts_region(T=1.5, S=0.5) == "Snowdrift (Chicken)"
    assert nf.ts_region(T=0.5, S=0.5) == "Harmony"
    g = nf.symmetric_2x2(1, -0.5, 1.5, 0)
    assert g.nash_equilibria() == [(0.0, 0.0)]


@pytest.mark.parametrize("key", list(nf.SYMMETRIC_PRESETS))
def test_rest_points_do_not_move(key):
    game = nf.SYMMETRIC_PRESETS[key]
    for rp in game.rest_points():
        assert np.allclose(game.velocity(rp["x"]), 0, atol=1e-10)


def test_rps_interior_stability_depends_on_win_and_loss():
    def interior(game):
        return [r for r in game.rest_points() if len(r["support"]) == 3][0]["stability"]

    assert interior(nf.rock_paper_scissors(1, 1)) == "center"
    assert interior(nf.rock_paper_scissors(1, 0.5)) == "sink"
    assert interior(nf.rock_paper_scissors(0.5, 1)) == "source"
