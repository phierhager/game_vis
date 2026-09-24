import test from 'node:test';
import assert from 'node:assert/strict';

import * as sx from '../../docs/js/lib/simplex.js';
import * as sg from '../../docs/js/lib/signaling.js';
import * as nf from '../../docs/js/lib/normalform.js';
import * as mk from '../../docs/js/lib/markets.js';
import { mulberry32 } from '../../docs/js/lib/rng.js';

const close = (a, b, tol = 1e-9) => assert.ok(Math.abs(a - b) <= tol, `${a} != ${b}`);

test('every rule is tangent to the simplex and matches its two-action form', () => {
  const rand = mulberry32(3);
  for (const rule of sx.RULES) {
    for (let i = 0; i < 50; i++) {
      const x = rand();
      const g = [rand() * 2 - 1, rand() * 2 - 1];
      const v = sx.field(rule, [x, 1 - x], g);
      close(v[0] + v[1], 0, 1e-12);
      close(v[0], sx.binaryField(rule, x, g[0] - g[1]), 1e-12);
    }
  }
});

test('projection onto the simplex', () => {
  const p = sx.projectToSimplex([0.5, 2, -1]);
  close(p.reduce((a, b) => a + b, 0), 1);
  assert.deepEqual(p.map((v) => +v.toFixed(12)), [0, 1, 0]);
  const v = sx.projection([1, 0, 0], [0, 1, -5]);
  assert.deepEqual(v, [-0.5, 0.5, 0]);
});

test('signaling gradients equal finite differences', () => {
  const game = sg.costlySignaling();
  const rand = mulberry32(5);
  const y = sg.randomStart(game, rand);
  const S = y.subarray(0, 4);
  const R = y.subarray(4);
  const [us, ur] = game.payoffs(S, R);
  const gs = game.senderSignal(R);
  const gr = game.receiverSignal(S);
  const h = 1e-7;
  for (let i = 0; i < 4; i++) {
    const S2 = Float64Array.from(S); S2[i] += h;
    close((game.payoffs(S2, R)[0] - us) / h, gs[i], 1e-5);
    const R2 = Float64Array.from(R); R2[i] += h;
    close((game.payoffs(S, R2)[1] - ur) / h, gr[i], 1e-5);
  }
});

test('binary fast path matches the general velocity', () => {
  const rand = mulberry32(9);
  for (const game of [sg.lewis(2, 0.7), sg.costlySignaling()]) {
    for (const rule of sx.RULES) {
      const X = [rand(), rand(), rand(), rand()];
      const opts = { rule, rates: [1.5, 0.5], exploration: 0.02 };
      const fast = game.binaryVelocity(X, opts);
      const { S, R } = sg.binaryToMatrices(X);
      const y = new Float64Array(8); y.set(S); y.set(R, 4);
      const v = game.velocity(y, opts);
      const slow = [v[0], v[2], v[4], v[6]];
      for (let i = 0; i < 4; i++) close(fast[i], slow[i], 1e-12);
    }
  }
});

test('equal-prior Lewis runs find a signaling system; skewed priors can pool', () => {
  const rand = mulberry32(1);
  let pooled = 0;
  for (let i = 0; i < 200; i++) {
    const X0 = [rand(), rand(), rand(), rand()];
    const even = sg.runBinary(sg.lewis(2, 0.5), X0, { tMax: 300 });
    assert.ok(even.code === 0 || even.code === 1 || Math.abs(X0[0] - X0[1]) < 0.02);
    if (sg.runBinary(sg.lewis(2, 0.9), X0, { tMax: 300 }).code === 2) pooled += 1;
  }
  assert.ok(pooled > 20, `pooled ${pooled}`);
});

test('costly signaling: the hybrid equilibrium is a rest point', () => {
  const game = sg.costlySignaling({ priorHigh: 0.4, costHigh: 0.3, costLow: 0.6 });
  const X = sg.hybridEquilibrium(game);
  const v = game.binaryVelocity(X, { rule: 'replicator' });
  for (const vi of v) close(vi, 0, 1e-12);
});

test('2x2 equilibria and stability', () => {
  const g = nf.BIMATRIX_PRESETS.stag_hunt;
  const eqs = g.nashEquilibria();
  assert.equal(eqs.length, 3);
  assert.deepEqual(eqs.map((e) => g.stability(e)), ['sink', 'sink', 'saddle']);
  const mp = nf.BIMATRIX_PRESETS.matching_pennies;
  assert.equal(mp.stability(mp.nashEquilibria()[0]), 'center');
});

test('3-strategy rest points do not move', () => {
  for (const g of Object.values(nf.SYMMETRIC_PRESETS)) {
    for (const rp of g.restPoints()) {
      for (const v of g.velocity(rp.x)) close(v, 0, 1e-9);
    }
  }
  const interior = (g) => g.restPoints().find((r) => r.support.length === 3).stability;
  assert.equal(interior(nf.rockPaperScissors(1, 1)), 'center');
  assert.equal(interior(nf.rockPaperScissors(1, 0.5)), 'sink');
  assert.equal(interior(nf.rockPaperScissors(0.5, 1)), 'source');
});

test('Bertrand: Calvano baseline, gradient play reaches Nash', () => {
  const m = new mk.LogitBertrand();
  close(m.nash()[0], 1.4729, 1e-4);
  close(m.collusive()[0], 1.925, 1e-4);
  let p = [1.1, 2.3];
  for (let i = 0; i < 6000; i++) p = m.step(p[0], p[1], 0.05);
  close(p[0], m.nash()[0], 1e-4);
  const lin = new mk.LinearBertrand();
  close(lin.nash()[0], lin.nashClosedForm(), 1e-6);
});

test('Edgeworth cycles never settle', () => {
  const e = new mk.Edgeworth();
  const path = e.simulate([0.9, 0.9], 200).slice(100).map((p) => p[0]);
  assert.ok(Math.max(...path) - Math.min(...path) > 0.05);
});
