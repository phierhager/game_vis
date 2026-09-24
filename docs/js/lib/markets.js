// Duopoly markets: firms follow their marginal profit (the first-order signal).
// Mirrors src/gamevis/markets.py.

class Duopoly {
  // subclasses define lo, hi, profit(p1, p2) -> [pi1, pi2], marginalProfit(p1, p2)

  crossMarginal(p1, p2) {
    const h = 1e-6;
    const d21 = (this.profit(p1 + h, p2)[1] - this.profit(p1 - h, p2)[1]) / (2 * h);
    const d12 = (this.profit(p1, p2 + h)[0] - this.profit(p1, p2 - h)[0]) / (2 * h);
    return [d21, d12];
  }

  velocity(p1, p2, rates = [1, 1], sympathy = 0) {
    let [g1, g2] = this.marginalProfit(p1, p2);
    if (sympathy) {
      const [d21, d12] = this.crossMarginal(p1, p2);
      g1 += sympathy * d21;
      g2 += sympathy * d12;
    }
    let v1 = rates[0] * g1;
    let v2 = rates[1] * g2;
    if ((p1 <= this.lo && v1 < 0) || (p1 >= this.hi && v1 > 0)) v1 = 0;
    if ((p2 <= this.lo && v2 < 0) || (p2 >= this.hi && v2 > 0)) v2 = 0;
    return [v1, v2];
  }

  step(p1, p2, dt, rates = [1, 1], sympathy = 0) {
    const f = (a, b) => this.velocity(a, b, rates, sympathy);
    const [a1, b1] = f(p1, p2);
    const [a2, b2] = f(p1 + (dt / 2) * a1, p2 + (dt / 2) * b1);
    const [a3, b3] = f(p1 + (dt / 2) * a2, p2 + (dt / 2) * b2);
    const [a4, b4] = f(p1 + dt * a3, p2 + dt * b3);
    const clip = (v) => Math.min(this.hi, Math.max(this.lo, v));
    return [clip(p1 + (dt / 6) * (a1 + 2 * a2 + 2 * a3 + a4)), clip(p2 + (dt / 6) * (b1 + 2 * b2 + 2 * b3 + b4))];
  }

  objective(i, own, other, sympathy) {
    const pr = i === 0 ? this.profit(own, other) : this.profit(other, own);
    return pr[i] + sympathy * pr[1 - i];
  }

  bestResponse(i, other, sympathy = 0, grid = 401) {
    let best = this.lo;
    let bestVal = -Infinity;
    for (let g = 0; g < grid; g++) {
      const x = this.lo + ((this.hi - this.lo) * g) / (grid - 1);
      const v = this.objective(i, x, other, sympathy);
      if (v > bestVal) {
        bestVal = v;
        best = x;
      }
    }
    // Newton polish on the smooth objective
    const h = 1e-5;
    let x = best;
    for (let it = 0; it < 30; it++) {
      const f0 = this.objective(i, x, other, sympathy);
      const fp = this.objective(i, x + h, other, sympathy);
      const fm = this.objective(i, x - h, other, sympathy);
      const d1 = (fp - fm) / (2 * h);
      const d2 = (fp - 2 * f0 + fm) / (h * h);
      if (d2 >= 0) break;
      const nx = Math.min(this.hi, Math.max(this.lo, x - d1 / d2));
      if (Math.abs(nx - x) < 1e-12) break;
      x = nx;
    }
    return x;
  }

  equilibrium(sympathy = 0, iters = 500) {
    let p = [(this.lo + this.hi) / 2, (this.lo + this.hi) / 2];
    for (let it = 0; it < iters; it++) {
      const nw = [this.bestResponse(0, p[1], sympathy), this.bestResponse(1, p[0], sympathy)];
      if (Math.max(Math.abs(nw[0] - p[0]), Math.abs(nw[1] - p[1])) < 1e-11) return nw;
      p = [0.5 * p[0] + 0.5 * nw[0], 0.5 * p[1] + 0.5 * nw[1]];
    }
    return p;
  }

  nash() { return this.equilibrium(0); }

  collusive() { return this.equilibrium(1); }
}

export class LogitBertrand extends Duopoly {
  constructor({ quality = [2, 2], cost = [1, 1], mu = 0.25, a0 = 0, lo = 1.0, hi = 2.4 } = {}) {
    super();
    Object.assign(this, { quality, cost, mu, a0, lo, hi });
    this.kind = 'price';
    this.name = 'Logit demand (differentiated products)';
  }

  demand(p1, p2) {
    const z1 = (this.quality[0] - p1) / this.mu;
    const z2 = (this.quality[1] - p2) / this.mu;
    const z0 = this.a0 / this.mu;
    const m = Math.max(z1, z2, z0);
    const e1 = Math.exp(z1 - m);
    const e2 = Math.exp(z2 - m);
    const e0 = Math.exp(z0 - m);
    const tot = e1 + e2 + e0;
    return [e1 / tot, e2 / tot];
  }

  profit(p1, p2) {
    const [q1, q2] = this.demand(p1, p2);
    return [(p1 - this.cost[0]) * q1, (p2 - this.cost[1]) * q2];
  }

  marginalProfit(p1, p2) {
    const [q1, q2] = this.demand(p1, p2);
    return [q1 * (1 - ((p1 - this.cost[0]) * (1 - q1)) / this.mu), q2 * (1 - ((p2 - this.cost[1]) * (1 - q2)) / this.mu)];
  }

  crossMarginal(p1, p2) {
    const [q1, q2] = this.demand(p1, p2);
    return [((p2 - this.cost[1]) * q1 * q2) / this.mu, ((p1 - this.cost[0]) * q1 * q2) / this.mu];
  }
}

export class LinearBertrand extends Duopoly {
  constructor({ alpha = 1, beta = 1, gamma = 0.5, cost = 0, lo = 0, hi = 1.5 } = {}) {
    super();
    Object.assign(this, { alpha, beta, gamma, cost, lo, hi });
    this.kind = 'price';
    this.name = 'Linear demand (differentiated products)';
  }

  demand(p1, p2) {
    return [Math.max(this.alpha - this.beta * p1 + this.gamma * p2, 0), Math.max(this.alpha - this.beta * p2 + this.gamma * p1, 0)];
  }

  profit(p1, p2) {
    const [q1, q2] = this.demand(p1, p2);
    return [(p1 - this.cost) * q1, (p2 - this.cost) * q2];
  }

  marginalProfit(p1, p2) {
    const [q1, q2] = this.demand(p1, p2);
    return [q1 > 0 ? q1 - this.beta * (p1 - this.cost) : 0, q2 > 0 ? q2 - this.beta * (p2 - this.cost) : 0];
  }

  nashClosedForm() { return (this.alpha + this.beta * this.cost) / (2 * this.beta - this.gamma); }

  collusiveClosedForm() {
    const b = this.beta - this.gamma;
    return (this.alpha + b * this.cost) / (2 * b);
  }
}

export class Cournot extends Duopoly {
  constructor({ a = 1, b = 1, c = 0, lo = 0, hi = 0.6 } = {}) {
    super();
    Object.assign(this, { a, b, c, lo, hi });
    this.kind = 'quantity';
    this.name = 'Cournot (quantity competition)';
  }

  price(q1, q2) { return Math.max(this.a - this.b * (q1 + q2), 0); }

  profit(q1, q2) {
    const P = this.price(q1, q2);
    return [(P - this.c) * q1, (P - this.c) * q2];
  }

  marginalProfit(q1, q2) {
    const P = this.price(q1, q2);
    return [P - this.c - this.b * q1, P - this.c - this.b * q2];
  }

  nashClosedForm() { return (this.a - this.c) / (3 * this.b); }

  collusiveClosedForm() { return (this.a - this.c) / (4 * this.b); }
}

export class Edgeworth {
  constructor({ capacity = 0.5, cost = 0.1, tick = 0.01 } = {}) {
    Object.assign(this, { capacity, cost, tick });
  }

  demand(p) { return Math.max(1 - p, 0); }

  sales(own, other) {
    const k = this.capacity;
    if (own < other) return Math.min(k, this.demand(own));
    if (own > other) return Math.min(k, Math.max(this.demand(own) - k, 0));
    return Math.min(k, this.demand(own) / 2);
  }

  profit(own, other) { return (own - this.cost) * this.sales(own, other); }

  monopolyPrice() { return (1 + this.cost) / 2; }

  residualPrice() { return Math.max((1 - this.capacity + this.cost) / 2, this.cost); }

  grid() {
    const out = [];
    const n = Math.floor((1 - this.cost) / this.tick + 1e-9);
    for (let i = 0; i <= n; i++) out.push(Math.round((this.cost + i * this.tick) * 1e10) / 1e10);
    return out;
  }

  bestResponse(other) {
    const grid = this.grid();
    let best = grid[0];
    let bestVal = -Infinity;
    for (const p of grid) {
      const v = this.profit(p, other);
      if (v >= bestVal - 1e-12) {
        // ties go to the higher price
        bestVal = Math.max(v, bestVal);
        best = p;
      }
    }
    return best;
  }

  simulate(p0 = [0.9, 0.9], periods = 120) {
    const p = [...p0];
    const out = [[...p]];
    for (let t = 0; t < periods; t++) {
      const i = t % 2;
      p[i] = this.bestResponse(p[1 - i]);
      out.push([...p]);
    }
    return out;
  }
}
