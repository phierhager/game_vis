// Tabular Q-learning pricing agents in a logit Bertrand duopoly.
// Mirrors src/gamevis/qlearning.py (Calvano, Calzolari, Denicolo, Pastorello 2020 baseline).

import { LogitBertrand } from './markets.js';
import { mulberry32 } from './rng.js';

export class QPricing {
  constructor({ market = new LogitBertrand(), m = 15, xi = 0.1, alpha = 0.15, delta = 0.95, beta = 4e-6 } = {}) {
    Object.assign(this, { market, m, xi, alpha, delta, beta });
    this.pNash = market.nash()[0];
    this.pMonopoly = market.collusive()[0];
    const span = this.pMonopoly - this.pNash;
    const lo = this.pNash - xi * span;
    const hi = this.pMonopoly + xi * span;
    this.prices = Array.from({ length: m }, (_, i) => lo + ((hi - lo) * i) / (m - 1));
    this.S = m * m;
    // profit[i][a1 * m + a2]
    this.profit = [new Float64Array(this.S), new Float64Array(this.S)];
    for (let a1 = 0; a1 < m; a1++) {
      for (let a2 = 0; a2 < m; a2++) {
        const [u, v] = market.profit(this.prices[a1], this.prices[a2]);
        this.profit[0][a1 * m + a2] = u;
        this.profit[1][a1 * m + a2] = v;
      }
    }
    this.profitNash = market.profit(this.pNash, this.pNash)[0];
    this.profitMonopoly = market.profit(this.pMonopoly, this.pMonopoly)[0];
  }

  // Q[i][s * m + a]: payoff of a against a uniformly random rival, discounted forever.
  initialQ() {
    const { m, S, delta } = this;
    const Q = [new Float64Array(S * m), new Float64Array(S * m)];
    for (let a = 0; a < m; a++) {
      let q1 = 0;
      let q2 = 0;
      for (let b = 0; b < m; b++) {
        q1 += this.profit[0][a * m + b];
        q2 += this.profit[1][b * m + a];
      }
      q1 /= m * (1 - delta);
      q2 /= m * (1 - delta);
      for (let s = 0; s < S; s++) {
        Q[0][s * m + a] = q1;
        Q[1][s * m + a] = q2;
      }
    }
    return Q;
  }

  profitGain(p) {
    return (p - this.profitNash) / (this.profitMonopoly - this.profitNash);
  }

  newSession(seed = 1) {
    const rand = mulberry32(seed);
    return { Q: this.initialQ(), s: Math.floor(rand() * this.S), t: 0, rand };
  }

  argmax(Q, s) {
    const { m } = this;
    let best = 0;
    let bv = -Infinity;
    const o = s * m;
    for (let a = 0; a < m; a++) {
      if (Q[o + a] > bv) {
        bv = Q[o + a];
        best = a;
      }
    }
    return best;
  }

  maxQ(Q, s) {
    const o = s * this.m;
    let bv = -Infinity;
    for (let a = 0; a < this.m; a++) if (Q[o + a] > bv) bv = Q[o + a];
    return bv;
  }

  // Train for `periods` periods; returns the average price of each firm over them.
  train(session, periods) {
    const { m, alpha, delta, beta } = this;
    const [Q1, Q2] = session.Q;
    const rand = session.rand;
    let s = session.s;
    let sum1 = 0;
    let sum2 = 0;
    for (let k = 0; k < periods; k++) {
      const eps = Math.exp(-beta * session.t);
      const a1 = rand() < eps ? Math.floor(rand() * m) : this.argmax(Q1, s);
      const a2 = rand() < eps ? Math.floor(rand() * m) : this.argmax(Q2, s);
      const sn = a1 * m + a2;
      const i1 = s * m + a1;
      const i2 = s * m + a2;
      Q1[i1] += alpha * (this.profit[0][sn] + delta * this.maxQ(Q1, sn) - Q1[i1]);
      Q2[i2] += alpha * (this.profit[1][sn] + delta * this.maxQ(Q2, sn) - Q2[i2]);
      s = sn;
      session.t += 1;
      sum1 += this.prices[a1];
      sum2 += this.prices[a2];
    }
    session.s = s;
    return [sum1 / periods, sum2 / periods];
  }

  greedyPair(session, s) {
    return [this.argmax(session.Q[0], s), this.argmax(session.Q[1], s)];
  }

  limitCycle(session, s0) {
    const seen = new Map();
    const path = [];
    let s = s0;
    while (!seen.has(s)) {
      seen.set(s, path.length);
      path.push(s);
      const [a1, a2] = this.greedyPair(session, s);
      s = a1 * this.m + a2;
    }
    return path.slice(seen.get(s));
  }

  cycleProfitGain(session, s0) {
    const cyc = this.limitCycle(session, s0);
    let tot = 0;
    for (const s of cyc) tot += (this.profit[0][s] + this.profit[1][s]) / 2;
    return this.profitGain(tot / cyc.length);
  }

  // Greedy play from s0; firm 1 plays its static best reply once, at `deviateAt`.
  play(session, s0, periods = 20, deviateAt = null) {
    const out = [];
    let s = s0;
    for (let t = 0; t < periods; t++) {
      let [a1, a2] = this.greedyPair(session, s);
      if (deviateAt !== null && t === deviateAt) {
        let best = 0;
        for (let a = 1; a < this.m; a++) if (this.profit[0][a * this.m + a2] > this.profit[0][best * this.m + a2]) best = a;
        a1 = best;
      }
      out.push([a1, a2]);
      s = a1 * this.m + a2;
    }
    return out;
  }
}
