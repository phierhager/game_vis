// Prints numbers computed by the browser library so tests/test_js_parity.py
// can compare them with the Python package.
import * as sx from '../../docs/js/lib/simplex.js';
import * as sg from '../../docs/js/lib/signaling.js';
import * as nf from '../../docs/js/lib/normalform.js';
import * as mk from '../../docs/js/lib/markets.js';
import { QPricing } from '../../docs/js/lib/qlearning.js';

const out = {};
const X = [0.73, 0.21, 0.64, 0.38];
out.fields = {};
for (const rule of sx.RULES) {
  out.fields[rule] = sx.field(rule, [0.2, 0.5, 0.3], [0.4, -0.1, 0.9]);
}
out.binary = {};
for (const [name, game] of [['lewis07', sg.lewis(2, 0.7)], ['costly', sg.costlySignaling()]]) {
  out.binary[name] = {};
  for (const rule of sx.RULES) out.binary[name][rule] = Array.from(game.binaryVelocity(X, { rule, rates: [1.3, 0.8], exploration: 0.01 }));
}
const g3 = sg.lewis(3);
const y3 = Float64Array.from([0.2, 0.5, 0.3, 0.6, 0.1, 0.3, 0.3, 0.3, 0.4, 0.5, 0.25, 0.25, 0.1, 0.8, 0.1, 0.2, 0.2, 0.6]);
out.lewis3_velocity = Array.from(g3.velocity(y3, { rule: 'replicator' }));
const run = Float64Array.from(X);
for (let i = 0; i < 400; i++) sg.stepBinary(sg.lewis(2, 0.7), run, 0.05, { rule: 'replicator' });
out.lewis07_run = Array.from(run);
const y = Float64Array.from(y3);
for (let i = 0; i < 200; i++) sg.step(g3, y, 0.05, { rule: 'replicator' });
out.lewis3_run = Array.from(y);
out.info = g3.information(y3.subarray(0, 9), y3.subarray(9));
out.nash = {};
for (const [k, g] of Object.entries(nf.BIMATRIX_PRESETS)) out.nash[k] = g.nashEquilibria().map((e) => [...e, g.stability(e)]);
out.rest = {};
for (const [k, g] of Object.entries(nf.SYMMETRIC_PRESETS)) out.rest[k] = g.restPoints().map((r) => [r.x, r.nash, r.stability]);
const m = new mk.LogitBertrand();
out.bertrand = { nash: m.nash(), collusive: m.collusive(), br: m.bestResponse(0, 1.8), half: m.equilibrium(0.5) };
out.edgeworth = new mk.Edgeworth().simulate([0.9, 0.9], 60);
const qp = new QPricing();
const Q0 = qp.initialQ();
out.qpricing = { prices: qp.prices, profit1: Array.from(qp.profit[0]), q0: Array.from(Q0[0].subarray(0, 15)), q1: Array.from(Q0[1].subarray(0, 15)) };
console.log(JSON.stringify(out));
