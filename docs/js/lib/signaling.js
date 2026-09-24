// Sender-receiver signaling games under first-order learning.
// Mirrors src/gamevis/signaling.py.
//
// Strategies are flat row-major arrays: S[t*k + m] = P(message m | state t),
// R[m*l + a] = P(action a | message m). Each row is its own simplex.

import { SMOOTH_RULES, binaryField, field, clean, randomSimplex } from './simplex.js';

export class SignalingGame {
  constructor({ prior, senderUtility, receiverUtility, messageCost, name = 'Signaling game', stateLabels, messageLabels, actionLabels, notes = '', extra = {} }) {
    const tot = prior.reduce((a, b) => a + b, 0);
    this.prior = prior.map((p) => p / tot);
    this.n = prior.length;
    this.l = senderUtility[0].length;
    this.k = messageCost[0].length;
    this.uS = senderUtility;
    this.uR = receiverUtility;
    this.cost = messageCost;
    this.name = name;
    this.stateLabels = stateLabels || Array.from({ length: this.n }, (_, i) => `t${i + 1}`);
    this.messageLabels = messageLabels || Array.from({ length: this.k }, (_, i) => `m${i + 1}`);
    this.actionLabels = actionLabels || Array.from({ length: this.l }, (_, i) => `a${i + 1}`);
    this.notes = notes;
    this.extra = extra;
  }

  get isBinary() {
    return this.n === 2 && this.k === 2 && this.l === 2;
  }

  get commonInterest() {
    for (let t = 0; t < this.n; t++) {
      for (let a = 0; a < this.l; a++) if (Math.abs(this.uS[t][a] - this.uR[t][a]) > 1e-12) return false;
      for (let m = 0; m < this.k; m++) if (Math.abs(this.cost[t][m]) > 1e-12) return false;
    }
    return true;
  }

  // dU_S / dS[t, m] = pi_t (sum_a R[m,a] uS[t,a] - C[t,m])
  senderSignal(R, out = new Float64Array(this.n * this.k)) {
    const { n, k, l } = this;
    for (let t = 0; t < n; t++) {
      for (let m = 0; m < k; m++) {
        let v = 0;
        for (let a = 0; a < l; a++) v += R[m * l + a] * this.uS[t][a];
        out[t * k + m] = this.prior[t] * (v - this.cost[t][m]);
      }
    }
    return out;
  }

  // dU_R / dR[m, a] = sum_t pi_t S[t, m] uR[t, a]
  receiverSignal(S, out = new Float64Array(this.k * this.l)) {
    const { n, k, l } = this;
    out.fill(0);
    for (let t = 0; t < n; t++) {
      for (let m = 0; m < k; m++) {
        const w = this.prior[t] * S[t * k + m];
        if (w === 0) continue;
        for (let a = 0; a < l; a++) out[m * l + a] += w * this.uR[t][a];
      }
    }
    return out;
  }

  payoffs(S, R) {
    const gs = this.senderSignal(R);
    const gr = this.receiverSignal(S);
    let us = 0;
    let ur = 0;
    for (let i = 0; i < gs.length; i++) us += S[i] * gs[i];
    for (let i = 0; i < gr.length; i++) ur += R[i] * gr[i];
    return [us, ur];
  }

  // Joint velocity of the flat state y = [S..., R...].
  velocity(y, { rule = 'replicator', rates = [1, 1], temperature = 0.05, exploration = 0 } = {}, out = new Float64Array(y.length)) {
    const { n, k, l } = this;
    const nk = n * k;
    const S = y.subarray(0, nk);
    const R = y.subarray(nk);
    const gs = this.senderSignal(R, scratch('gs', nk));
    const gr = this.receiverSignal(S, scratch('gr', k * l));
    for (let t = 0; t < n; t++) rowVelocity(rule, y, gs, t * k, t * k, k, out, t * k, rates[0], exploration, temperature);
    for (let m = 0; m < k; m++) rowVelocity(rule, y, gr, nk + m * l, m * l, l, out, nk + m * l, rates[1], exploration, temperature);
    return out;
  }

  // Velocity in X = (S[t1,m1], S[t2,m1], R[m1,a1], R[m2,a1]) for 2x2x2 games.
  binaryVelocity(X, { rule = 'replicator', rates = [1, 1], temperature = 0.05, exploration = 0 } = {}, out = new Float64Array(4)) {
    const [x1, x2, y1, y2] = X;
    const pi = this.prior;
    const duS0 = this.uS[0][0] - this.uS[0][1];
    const duS1 = this.uS[1][0] - this.uS[1][1];
    const duR0 = this.uR[0][0] - this.uR[0][1];
    const duR1 = this.uR[1][0] - this.uR[1][1];
    const dc0 = this.cost[0][0] - this.cost[0][1];
    const dc1 = this.cost[1][0] - this.cost[1][1];
    const sep = y1 - y2;
    const d1 = pi[0] * (sep * duS0 - dc0);
    const d2 = pi[1] * (sep * duS1 - dc1);
    const e1 = pi[0] * x1 * duR0 + pi[1] * x2 * duR1;
    const e2 = pi[0] * (1 - x1) * duR0 + pi[1] * (1 - x2) * duR1;
    out[0] = rates[0] * binaryField(rule, x1, d1, temperature) + exploration * (0.5 - x1);
    out[1] = rates[0] * binaryField(rule, x2, d2, temperature) + exploration * (0.5 - x2);
    out[2] = rates[1] * binaryField(rule, y1, e1, temperature) + exploration * (0.5 - y1);
    out[3] = rates[1] * binaryField(rule, y2, e2, temperature) + exploration * (0.5 - y2);
    return out;
  }

  // Payoff advantages that drive each binary agent (useful for drawing fields).
  binaryAdvantages(X) {
    const [x1, x2, y1, y2] = X;
    const pi = this.prior;
    const sep = y1 - y2;
    const d1 = pi[0] * (sep * (this.uS[0][0] - this.uS[0][1]) - (this.cost[0][0] - this.cost[0][1]));
    const d2 = pi[1] * (sep * (this.uS[1][0] - this.uS[1][1]) - (this.cost[1][0] - this.cost[1][1]));
    const duR0 = this.uR[0][0] - this.uR[0][1];
    const duR1 = this.uR[1][0] - this.uR[1][1];
    const e1 = pi[0] * x1 * duR0 + pi[1] * x2 * duR1;
    const e2 = pi[0] * (1 - x1) * duR0 + pi[1] * (1 - x2) * duR1;
    return [d1, d2, e1, e2];
  }

  channel(S, R) {
    const { n, k, l } = this;
    const P = new Float64Array(n * l);
    for (let t = 0; t < n; t++) {
      for (let m = 0; m < k; m++) {
        const s = S[t * k + m];
        if (s === 0) continue;
        for (let a = 0; a < l; a++) P[t * l + a] += s * R[m * l + a];
      }
    }
    return P;
  }

  // I(T; Y) in bits for a channel P[t * cols + y] = P(y | t).
  mutualInformation(P, cols) {
    const { n } = this;
    const marg = new Float64Array(cols);
    for (let t = 0; t < n; t++) for (let y = 0; y < cols; y++) marg[y] += this.prior[t] * P[t * cols + y];
    let I = 0;
    for (let t = 0; t < n; t++) {
      for (let y = 0; y < cols; y++) {
        const j = this.prior[t] * P[t * cols + y];
        if (j > 0 && marg[y] > 0) I += j * Math.log2(P[t * cols + y] / marg[y]);
      }
    }
    return Math.max(I, 0);
  }

  entropy() {
    let h = 0;
    for (const p of this.prior) if (p > 0) h -= p * Math.log2(p);
    return h;
  }

  information(S, R) {
    return {
      message: this.mutualInformation(S, this.k),
      action: this.mutualInformation(this.channel(S, R), this.l),
    };
  }
}

// ---------------------------------------------------------------------------
// Allocation-free helpers for the general dynamics
// ---------------------------------------------------------------------------

const SCRATCH = new Map();

function scratch(name, len) {
  const key = `${name}:${len}`;
  let a = SCRATCH.get(key);
  if (!a) {
    a = new Float64Array(len);
    SCRATCH.set(key, a);
  }
  return a;
}

// Velocity of one simplex row y[yo .. yo+len) given its signal g[go .. go+len), written to out[oo ..].
function rowVelocity(rule, y, g, yo, go, len, out, oo, rate, eps, temperature) {
  if (rule === 'replicator' || rule === 'softmax_pg') {
    let m = 0;
    for (let i = 0; i < len; i++) m += y[yo + i] * g[go + i];
    if (rule === 'replicator') {
      for (let i = 0; i < len; i++) out[oo + i] = rate * y[yo + i] * (g[go + i] - m);
    } else {
      // replicator applied twice: J (J g)
      let m2 = 0;
      for (let i = 0; i < len; i++) m2 += y[yo + i] * y[yo + i] * (g[go + i] - m);
      for (let i = 0; i < len; i++) {
        const r = y[yo + i] * (g[go + i] - m);
        out[oo + i] = rate * y[yo + i] * (r - m2);
      }
    }
  } else if (rule === 'logit') {
    let mx = -Infinity;
    for (let i = 0; i < len; i++) mx = Math.max(mx, g[go + i] / temperature);
    let z = 0;
    for (let i = 0; i < len; i++) z += Math.exp(g[go + i] / temperature - mx);
    for (let i = 0; i < len; i++) out[oo + i] = rate * (Math.exp(g[go + i] / temperature - mx) / z - y[yo + i]);
  } else {
    const x = Array.from(y.subarray(yo, yo + len));
    const v = field(rule, x, Array.from(g.subarray(go, go + len)), temperature);
    for (let i = 0; i < len; i++) out[oo + i] = rate * v[i];
  }
  if (eps) for (let i = 0; i < len; i++) out[oo + i] += eps * (1 / len - y[yo + i]);
}

// ---------------------------------------------------------------------------
// Presets
// ---------------------------------------------------------------------------

export function lewis(n = 2, prior = null, nMessages = null) {
  const k = nMessages ?? n;
  let p;
  if (prior === null || prior === undefined) p = Array(n).fill(1 / n);
  else if (typeof prior === 'number') p = [prior, 1 - prior];
  else p = prior;
  const u = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  return new SignalingGame({
    prior: p,
    senderUtility: u,
    receiverUtility: u,
    messageCost: Array.from({ length: n }, () => Array(k).fill(0)),
    name: `Lewis signaling game (${n}x${k}x${n})`,
    notes: 'Pure common interest: both want the act to fit the state.',
  });
}

export function costlySignaling({ priorHigh = 0.4, costHigh = 0.3, costLow = 0.6, benefit = 1 } = {}) {
  return new SignalingGame({
    prior: [priorHigh, 1 - priorHigh],
    senderUtility: [[benefit, 0], [benefit, 0]],
    receiverUtility: [[1, 0], [0, 1]],
    messageCost: [[costHigh, 0], [costLow, 0]],
    name: 'Costly signaling (handicap principle)',
    stateLabels: ['high', 'low'],
    messageLabels: ['display', 'no display'],
    actionLabels: ['reward', 'withhold'],
    notes: 'Both sender types want the reward; only the high type deserves it.',
    extra: { priorHigh, costHigh, costLow, benefit },
  });
}

export function hybridEquilibrium(game) {
  const p = game.extra;
  if (!p || p.priorHigh === undefined) return null;
  const mimic = p.priorHigh / (1 - p.priorHigh);
  const trust = p.costLow / p.benefit;
  if (!(mimic > 0 && mimic < 1 && trust > 0 && trust < 1 && p.costHigh <= p.costLow)) return null;
  return [1, mimic, trust, 0];
}

// ---------------------------------------------------------------------------
// Integration
// ---------------------------------------------------------------------------

function cleanRows(y, rows) {
  // rows: array of [offset, length]
  for (const [o, len] of rows) {
    let s = 0;
    for (let i = o; i < o + len; i++) {
      if (!(y[i] > 0)) y[i] = 0;
      s += y[i];
    }
    if (s > 0) for (let i = o; i < o + len; i++) y[i] /= s;
    else for (let i = o; i < o + len; i++) y[i] = 1 / len;
  }
}

export function rowLayout(game) {
  const rows = [];
  for (let t = 0; t < game.n; t++) rows.push([t * game.k, game.k]);
  for (let m = 0; m < game.k; m++) rows.push([game.n * game.k + m * game.l, game.l]);
  return rows;
}

// One integration step of the general dynamics, in place on the flat state y.
export function step(game, y, dt, opts = {}) {
  const rule = opts.rule || 'replicator';
  const N = y.length;
  const rows = opts.rows || rowLayout(game);
  const k1 = scratch('k1', N);
  if (SMOOTH_RULES.has(rule)) {
    const k2 = scratch('k2', N);
    const k3 = scratch('k3', N);
    const k4 = scratch('k4', N);
    const tmp = scratch('tmp', N);
    game.velocity(y, opts, k1);
    for (let i = 0; i < N; i++) tmp[i] = y[i] + (dt / 2) * k1[i];
    game.velocity(tmp, opts, k2);
    for (let i = 0; i < N; i++) tmp[i] = y[i] + (dt / 2) * k2[i];
    game.velocity(tmp, opts, k3);
    for (let i = 0; i < N; i++) tmp[i] = y[i] + dt * k3[i];
    game.velocity(tmp, opts, k4);
    for (let i = 0; i < N; i++) y[i] += (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
    cleanRows(y, rows);
  } else {
    game.velocity(y, opts, k1);
    for (let i = 0; i < N; i++) y[i] += dt * k1[i];
    // projected Euler: put every row back on its simplex
    for (const [o, len] of rows) {
      const row = Array.from(y.subarray(o, o + len));
      const proj = projectRow(row);
      for (let i = 0; i < len; i++) y[o + i] = proj[i];
    }
  }
  return y;
}

function projectRow(y) {
  const u = [...y].sort((a, b) => b - a);
  let css = 0;
  let theta = 0;
  for (let i = 0; i < u.length; i++) {
    css += u[i];
    const t = (css - 1) / (i + 1);
    if (u[i] - t > 0) theta = t;
  }
  return y.map((v) => Math.max(v - theta, 0));
}

// One step of the 2x2x2 fast path, in place on X (length 4).
const k1b = new Float64Array(4);
const k2b = new Float64Array(4);
const k3b = new Float64Array(4);
const k4b = new Float64Array(4);
const tmpb = new Float64Array(4);

export function stepBinary(game, X, dt, opts = {}) {
  const rule = opts.rule || 'replicator';
  if (SMOOTH_RULES.has(rule)) {
    game.binaryVelocity(X, opts, k1b);
    for (let i = 0; i < 4; i++) tmpb[i] = X[i] + (dt / 2) * k1b[i];
    game.binaryVelocity(tmpb, opts, k2b);
    for (let i = 0; i < 4; i++) tmpb[i] = X[i] + (dt / 2) * k2b[i];
    game.binaryVelocity(tmpb, opts, k3b);
    for (let i = 0; i < 4; i++) tmpb[i] = X[i] + dt * k3b[i];
    game.binaryVelocity(tmpb, opts, k4b);
    for (let i = 0; i < 4; i++) X[i] += (dt / 6) * (k1b[i] + 2 * k2b[i] + 2 * k3b[i] + k4b[i]);
  } else {
    game.binaryVelocity(X, opts, k1b);
    for (let i = 0; i < 4; i++) X[i] += dt * k1b[i];
  }
  for (let i = 0; i < 4; i++) X[i] = X[i] < 0 ? 0 : X[i] > 1 ? 1 : X[i];
  return X;
}

export function binaryToMatrices(X) {
  return {
    S: Float64Array.from([X[0], 1 - X[0], X[1], 1 - X[1]]),
    R: Float64Array.from([X[2], 1 - X[2], X[3], 1 - X[3]]),
  };
}

export function matricesToBinary(S, R) {
  return Float64Array.from([S[0], S[2], R[0], R[2]]);
}

export function randomStart(game, rand) {
  const y = new Float64Array(game.n * game.k + game.k * game.l);
  for (let t = 0; t < game.n; t++) y.set(randomSimplex(rand, game.k), t * game.k);
  for (let m = 0; m < game.k; m++) y.set(randomSimplex(rand, game.l), game.n * game.k + m * game.l);
  return y;
}

// ---------------------------------------------------------------------------
// Outcomes
// ---------------------------------------------------------------------------

export const OUTCOME_CODES = ['m1 means t1', 'm1 means t2', 'no information', 'cycling or partial'];

export function outcomeLabels(game) {
  const t = game.stateLabels;
  const m = game.messageLabels;
  return [`${m[0]} means ${t[0]}`, `${m[0]} means ${t[1]}`, 'no information', 'cycling or partial'];
}

// Classify a 2x2x2 run from the min/max/final of its meaning coordinates over a tail window.
export function classifyBinary({ dxMin, dxMax, dyMin, dyMax, dx, dy }, { separated = 0.9, silent = 0.1, swing = 0.05 } = {}) {
  const spread = Math.max(dxMax - dxMin, dyMax - dyMin);
  if (spread >= swing) return 3;
  if (Math.min(Math.abs(dx), Math.abs(dy)) < silent) return 2;
  if (dx > separated && dy > separated) return 0;
  if (dx < -separated && dy < -separated) return 1;
  return 3;
}

// Run a 2x2x2 start forward and classify it (the tail is the last 20% of the run).
export function runBinary(game, X0, { tMax = 400, dt = 0.1, ...opts } = {}) {
  const X = Float64Array.from(X0);
  const steps = Math.round(tMax / dt);
  const tailStart = Math.floor(steps * 0.8);
  const acc = { dxMin: Infinity, dxMax: -Infinity, dyMin: Infinity, dyMax: -Infinity, dx: 0, dy: 0 };
  for (let i = 1; i <= steps; i++) {
    stepBinary(game, X, dt, opts);
    if (i >= tailStart) {
      const dx = X[0] - X[1];
      const dy = X[2] - X[3];
      if (dx < acc.dxMin) acc.dxMin = dx;
      if (dx > acc.dxMax) acc.dxMax = dx;
      if (dy < acc.dyMin) acc.dyMin = dy;
      if (dy > acc.dyMax) acc.dyMax = dy;
      acc.dx = dx;
      acc.dy = dy;
    }
  }
  return { code: classifyBinary(acc), X };
}

// Classify a general run by its information level over the tail.
export function classifyGeneral(game, infoTail) {
  const H = Math.max(game.entropy(), 1e-12);
  let lo = Infinity;
  let hi = -Infinity;
  let mean = 0;
  for (const v of infoTail) {
    const x = v / H;
    lo = Math.min(lo, x);
    hi = Math.max(hi, x);
    mean += x;
  }
  mean /= infoTail.length;
  if (hi - lo < 0.05 && mean > 0.95) return 0;
  if (hi - lo < 0.05 && mean < 0.05) return 2;
  return 3;
}

// For Lewis games: communicative success times the number of states.
export function distinguishedStates(game, S, R) {
  return game.payoffs(S, R)[1] * game.n;
}

export { clean };
