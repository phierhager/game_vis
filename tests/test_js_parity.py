"""The browser library (docs/js/lib) must compute the same numbers as the Python package."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from gamevis import markets as mk
from gamevis import normal_form as nf
from gamevis import signaling as sg
from gamevis import simplex

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


@pytest.fixture(scope="module")
def js():
    res = subprocess.run(["node", str(ROOT / "tests/js/parity.mjs")], capture_output=True, text=True, check=True, cwd=ROOT)
    return json.loads(res.stdout)


def test_fields(js):
    for rule in simplex.RULES:
        py = simplex.field(rule, np.array([0.2, 0.5, 0.3]), np.array([0.4, -0.1, 0.9]))
        assert np.allclose(py, js["fields"][rule], atol=1e-12), rule


def test_signaling(js):
    X = np.array([0.73, 0.21, 0.64, 0.38])
    for name, game in [("lewis07", sg.lewis(2, 0.7)), ("costly", sg.costly_signaling())]:
        for rule in simplex.RULES:
            py = game.binary_velocity(X, rule, rates=(1.3, 0.8), exploration=0.01)
            assert np.allclose(py, js["binary"][name][rule], atol=1e-12), (name, rule)
    g3 = sg.lewis(3)
    y3 = np.array([0.2, 0.5, 0.3, 0.6, 0.1, 0.3, 0.3, 0.3, 0.4, 0.5, 0.25, 0.25, 0.1, 0.8, 0.1, 0.2, 0.2, 0.6])
    S, R = y3[:9].reshape(3, 3), y3[9:].reshape(3, 3)
    dS, dR = g3.velocity(S, R, "replicator")
    assert np.allclose(np.concatenate([dS.ravel(), dR.ravel()]), js["lewis3_velocity"], atol=1e-12)
    _, Xs = sg.simulate_binary(sg.lewis(2, 0.7), X, "replicator", t_max=20, dt=0.05, record_every=400)
    assert np.allclose(Xs[-1], js["lewis07_run"], atol=1e-10)
    tr = sg.simulate(g3, S, R, "replicator", t_max=10, dt=0.05, record_every=200)
    assert np.allclose(np.concatenate([tr.S[-1].ravel(), tr.R[-1].ravel()]), js["lewis3_run"], atol=1e-10)
    assert np.isclose(g3.mutual_information(S), js["info"]["message"])
    assert np.isclose(g3.mutual_information(g3.channel(S, R)), js["info"]["action"])


def test_normal_form(js):
    for key, g in nf.BIMATRIX_PRESETS.items():
        py = [(p, q, g.equilibrium_stability((p, q))) for p, q in g.nash_equilibria()]
        assert len(py) == len(js["nash"][key])
        for (p, q, k), (jp, jq, jk) in zip(py, js["nash"][key]):
            assert np.isclose(p, jp) and np.isclose(q, jq) and k == jk, key
    for key, g in nf.SYMMETRIC_PRESETS.items():
        py = g.rest_points()
        assert len(py) == len(js["rest"][key]), key
        for r, (jx, jnash, jstab) in zip(py, js["rest"][key]):
            assert np.allclose(r["x"], jx) and r["nash"] == jnash and r["stability"] == jstab, key


def test_markets(js):
    m = mk.LogitBertrand()
    assert np.allclose(m.nash(), js["bertrand"]["nash"], atol=1e-8)
    assert np.allclose(m.collusive(), js["bertrand"]["collusive"], atol=1e-8)
    assert np.isclose(m.best_response(0, 1.8)[0], js["bertrand"]["br"], atol=1e-8)
    assert np.allclose(m.equilibrium(0.5), js["bertrand"]["half"], atol=1e-8)
    assert np.allclose(mk.Edgeworth().simulate((0.9, 0.9), 60), js["edgeworth"], atol=1e-12)
