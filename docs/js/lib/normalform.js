// Normal-form games: 2x2 bimatrix games and symmetric n-strategy games.
// Mirrors src/gamevis/normal_form.py.

import { SMOOTH_RULES, binaryField, binaryDiscreteStep, field, clean, projectToSimplex } from './simplex.js';

// ---------------------------------------------------------------------------
// Linear stability of planar rest points
// ---------------------------------------------------------------------------

export function jacobian2(f, x, y, h = 1e-6) {
  const [a1, b1] = f(x + h, y);
  const [a0, b0] = f(x - h, y);
  const [c1, d1] = f(x, y + h);
  const [c0, d0] = f(x, y - h);
  return [
    [(a1 - a0) / (2 * h), (c1 - c0) / (2 * h)],
    [(b1 - b0) / (2 * h), (d1 - d0) / (2 * h)],
  ];
}

export function classify2(J, tol = 1e-7) {
  const tr = J[0][0] + J[1][1];
  const det = J[0][0] * J[1][1] - J[0][1] * J[1][0];
  const disc = tr * tr - 4 * det;
  if (disc < 0) {
    // complex pair with real part tr / 2
    if (Math.abs(tr) <= tol) return 'center';
    return tr < 0 ? 'sink' : 'source';
  }
  const r = Math.sqrt(disc);
  const l1 = (tr - r) / 2;
  const l2 = (tr + r) / 2;
  if (l1 < -tol && l2 < -tol) return 'sink';
  if (l1 > tol && l2 > tol) return 'source';
  if (l1 < -tol && l2 > tol) return 'saddle';
  return 'degenerate';
}

// ---------------------------------------------------------------------------
// Two players, two actions
// ---------------------------------------------------------------------------

export class Bimatrix {
  // A[i][j]: row player's payoff when row plays i and column plays j; B likewise for the column player.
  constructor(A, B, { name = '2x2 game', rowActions = ['A', 'B'], colActions = ['A', 'B'], notes = '' } = {}) {
    this.A = A;
    this.B = B;
    this.name = name;
    this.rowActions = rowActions;
    this.colActions = colActions;
    this.notes = notes;
  }

  advantages(p, q) {
    const { A, B } = this;
    const dRow = (A[0][0] - A[1][0]) * q + (A[0][1] - A[1][1]) * (1 - q);
    const dCol = (B[0][0] - B[0][1]) * p + (B[1][0] - B[1][1]) * (1 - p);
    return [dRow, dCol];
  }

  payoffs(p, q) {
    const w = [p * q, p * (1 - q), (1 - p) * q, (1 - p) * (1 - q)];
    const a = [this.A[0][0], this.A[0][1], this.A[1][0], this.A[1][1]];
    const b = [this.B[0][0], this.B[0][1], this.B[1][0], this.B[1][1]];
    return [w.reduce((s, wi, i) => s + wi * a[i], 0), w.reduce((s, wi, i) => s + wi * b[i], 0)];
  }

  velocity(p, q, rule = 'replicator', rates = [1, 1], temperature = 0.05) {
    const [dr, dc] = this.advantages(p, q);
    return [rates[0] * binaryField(rule, p, dr, temperature), rates[1] * binaryField(rule, q, dc, temperature)];
  }

  discreteStep(p, q, rule, eta, rates = [1, 1], temperature = 0.05, signal = null) {
    const [dr, dc] = signal || this.advantages(p, q);
    return [binaryDiscreteStep(rule, p, dr, eta * rates[0], temperature), binaryDiscreteStep(rule, q, dc, eta * rates[1], temperature)];
  }

  // Continuous-time step (RK4 for smooth rules, Euler otherwise), clipped to the square.
  step(p, q, dt, rule = 'replicator', rates = [1, 1], temperature = 0.05) {
    const f = (a, b) => this.velocity(a, b, rule, rates, temperature);
    let np;
    let nq;
    if (SMOOTH_RULES.has(rule)) {
      const [a1, b1] = f(p, q);
      const [a2, b2] = f(p + (dt / 2) * a1, q + (dt / 2) * b1);
      const [a3, b3] = f(p + (dt / 2) * a2, q + (dt / 2) * b2);
      const [a4, b4] = f(p + dt * a3, q + dt * b3);
      np = p + (dt / 6) * (a1 + 2 * a2 + 2 * a3 + a4);
      nq = q + (dt / 6) * (b1 + 2 * b2 + 2 * b3 + b4);
    } else {
      const [a1, b1] = f(p, q);
      np = p + dt * a1;
      nq = q + dt * b1;
    }
    return [Math.min(1, Math.max(0, np)), Math.min(1, Math.max(0, nq))];
  }

  nashEquilibria(tol = 1e-12) {
    const eqs = [];
    for (const p of [1, 0]) {
      for (const q of [1, 0]) {
        const [dr, dc] = this.advantages(p, q);
        const rowOk = p === 1 ? dr >= -tol : dr <= tol;
        const colOk = q === 1 ? dc >= -tol : dc <= tol;
        if (rowOk && colOk) eqs.push([p, q]);
      }
    }
    const { A, B } = this;
    const denQ = A[0][0] - A[1][0] - (A[0][1] - A[1][1]);
    const denP = B[0][0] - B[0][1] - (B[1][0] - B[1][1]);
    if (Math.abs(denQ) > tol && Math.abs(denP) > tol) {
      const q = -(A[0][1] - A[1][1]) / denQ;
      const p = -(B[1][0] - B[1][1]) / denP;
      if (p > 0 && p < 1 && q > 0 && q < 1) eqs.push([p, q]);
    }
    return eqs;
  }

  stability([p, q], rule = 'replicator') {
    return classify2(jacobian2((a, b) => this.velocity(a, b, rule), p, q));
  }

  // Null-clines of the replicator-type dynamics: where each player is indifferent.
  indifference() {
    const { A, B } = this;
    const denQ = A[0][0] - A[1][0] - (A[0][1] - A[1][1]);
    const denP = B[0][0] - B[0][1] - (B[1][0] - B[1][1]);
    const q = Math.abs(denQ) > 1e-12 ? -(A[0][1] - A[1][1]) / denQ : null;
    const p = Math.abs(denP) > 1e-12 ? -(B[1][0] - B[1][1]) / denP : null;
    return { rowIndifferentAtQ: q !== null && q > 0 && q < 1 ? q : null, colIndifferentAtP: p !== null && p > 0 && p < 1 ? p : null };
  }

  get isSymmetric() {
    const { A, B } = this;
    return A[0][0] === B[0][0] && A[0][1] === B[1][0] && A[1][0] === B[0][1] && A[1][1] === B[1][1];
  }
}

export function symmetric2x2(R = 1, S = 0, T = 0, P = 0, name = null) {
  const A = [[R, S], [T, P]];
  const B = [[R, T], [S, P]];
  return new Bimatrix(A, B, { name: name || tsRegion(T, S, R, P), rowActions: ['Cooperate', 'Defect'], colActions: ['Cooperate', 'Defect'] });
}

export function tsRegion(T, S, R = 1, P = 0) {
  const greedy = T > R;
  const fear = S < P;
  if (greedy && fear) return "Prisoner's Dilemma";
  if (greedy) return 'Snowdrift (Chicken)';
  if (fear) return 'Stag Hunt';
  return 'Harmony';
}

export const BIMATRIX_PRESETS = {
  prisoners_dilemma: new Bimatrix([[3, 0], [5, 1]], [[3, 5], [0, 1]], {
    name: "Prisoner's Dilemma", rowActions: ['Cooperate', 'Defect'], colActions: ['Cooperate', 'Defect'],
    notes: 'Defecting is better whatever the other does; both end up worse off than if both cooperated.',
  }),
  stag_hunt: new Bimatrix([[4, 0], [3, 3]], [[4, 3], [0, 3]], {
    name: 'Stag Hunt', rowActions: ['Stag', 'Hare'], colActions: ['Stag', 'Hare'],
    notes: 'Two stable conventions. The safe one (Hare) has the larger basin, the better one (Stag) needs trust.',
  }),
  chicken: new Bimatrix([[0, -1], [1, -10]], [[0, 1], [-1, -10]], {
    name: 'Chicken (Hawk-Dove)', rowActions: ['Swerve', 'Straight'], colActions: ['Swerve', 'Straight'],
    notes: 'Anti-coordination: someone has to yield. With two populations the mixed equilibrium is a saddle; along the diagonal (one population) it is stable.',
  }),
  matching_pennies: new Bimatrix([[1, -1], [-1, 1]], [[-1, 1], [1, -1]], {
    name: 'Matching Pennies', rowActions: ['Heads', 'Tails'], colActions: ['Heads', 'Tails'],
    notes: 'Zero-sum and purely rotational. Continuous replicator orbits close; discrete learning with a fixed step spirals outward.',
  }),
  battle_of_sexes: new Bimatrix([[3, 0], [0, 2]], [[2, 0], [0, 3]], {
    name: 'Battle of the Sexes', rowActions: ['Opera', 'Football'], colActions: ['Opera', 'Football'],
    notes: 'Both want to coordinate but disagree on where.',
  }),
  coordination: new Bimatrix([[1, 0], [0, 1]], [[1, 0], [0, 1]], {
    name: 'Pure Coordination', rowActions: ['Left', 'Right'], colActions: ['Left', 'Right'],
    notes: 'Drive on the left or on the right: it only matters to agree.',
  }),
};

// ---------------------------------------------------------------------------
// One population, n strategies
// ---------------------------------------------------------------------------

export class Symmetric {
  constructor(A, { name = 'Symmetric game', labels = null, notes = '' } = {}) {
    this.A = A;
    this.n = A.length;
    this.name = name;
    this.labels = labels || Array.from({ length: this.n }, (_, i) => `s${i + 1}`);
    this.notes = notes;
  }

  signal(x) {
    return this.A.map((row) => row.reduce((s, a, j) => s + a * x[j], 0));
  }

  velocity(x, rule = 'replicator', temperature = 0.05) {
    return field(rule, x, this.signal(x), temperature);
  }

  meanPayoff(x) {
    const f = this.signal(x);
    return f.reduce((s, v, i) => s + v * x[i], 0);
  }

  step(x, dt, rule = 'replicator', temperature = 0.05) {
    const f = (z) => this.velocity(z, rule, temperature);
    if (SMOOTH_RULES.has(rule)) {
      const k1 = f(x);
      const k2 = f(x.map((v, i) => v + (dt / 2) * k1[i]));
      const k3 = f(x.map((v, i) => v + (dt / 2) * k2[i]));
      const k4 = f(x.map((v, i) => v + dt * k3[i]));
      return clean(x.map((v, i) => v + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])));
    }
    const k1 = f(x);
    return projectToSimplex(x.map((v, i) => v + dt * k1[i]));
  }

  restPoints(tol = 1e-9) {
    const n = this.n;
    const out = [];
    for (const idx of supports(n)) {
      const s = idx.length;
      // [A_II  -1] [x]   [0]
      // [1^T    0] [v] = [1]
      const M = Array.from({ length: s + 1 }, () => Array(s + 1).fill(0));
      const rhs = Array(s + 1).fill(0);
      for (let a = 0; a < s; a++) {
        for (let b = 0; b < s; b++) M[a][b] = this.A[idx[a]][idx[b]];
        M[a][s] = -1;
        M[s][a] = 1;
      }
      rhs[s] = 1;
      const sol = solve(M, rhs);
      if (!sol) continue;
      if (sol.slice(0, s).some((v) => v <= tol)) continue;
      const x = Array(n).fill(0);
      idx.forEach((i, a) => { x[i] = sol[a]; });
      const f = this.signal(x);
      const nash = f.every((v) => v <= sol[s] + 1e-9);
      out.push({ x, support: idx, nash, stability: this.stability(x) });
    }
    return out;
  }

  stability(x, rule = 'replicator') {
    if (this.n !== 3) return 'degenerate';
    const f = (a, b) => {
      const v = this.velocity([a, b, 1 - a - b], rule);
      return [v[0], v[1]];
    };
    return classify2(jacobian2(f, x[0], x[1]));
  }
}

// Non-empty index subsets, smallest first, each size in lexicographic order.
function supports(n) {
  const out = [];
  const rec = (start, size, acc) => {
    if (acc.length === size) {
      out.push([...acc]);
      return;
    }
    for (let i = start; i < n; i++) rec(i + 1, size, [...acc, i]);
  };
  for (let size = 1; size <= n; size++) rec(0, size, []);
  return out;
}

// Gaussian elimination with partial pivoting; returns null when singular.
export function solve(M, b) {
  const n = b.length;
  const a = M.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(a[r][c]) > Math.abs(a[piv][c])) piv = r;
    if (Math.abs(a[piv][c]) < 1e-12) return null;
    [a[c], a[piv]] = [a[piv], a[c]];
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = a[r][c] / a[c][c];
      for (let k = c; k <= n; k++) a[r][k] -= f * a[c][k];
    }
  }
  return a.map((row, i) => row[n] / row[i]);
}

export function rockPaperScissors(win = 1, lose = 1) {
  const A = [[0, -lose, win], [win, 0, -lose], [-lose, win, 0]];
  const kind = win === lose ? 'zero-sum' : win > lose ? 'wins pay more' : 'losses cost more';
  return new Symmetric(A, { name: `Rock-Paper-Scissors (${kind})`, labels: ['Rock', 'Paper', 'Scissors'] });
}

export const SYMMETRIC_PRESETS = {
  rps_zero_sum: Object.assign(rockPaperScissors(1, 1), { notes: 'Zero-sum: the replicator dynamics conserves a quantity, so orbits close around the mixed equilibrium.' }),
  rps_stable: Object.assign(rockPaperScissors(1, 0.5), { notes: 'A win pays more than a loss costs: the population spirals into the mixed equilibrium.' }),
  rps_unstable: Object.assign(rockPaperScissors(0.5, 1), { notes: 'A loss costs more than a win pays: the population spirals out toward the boundary cycle Rock → Paper → Scissors.' }),
  coordination3: new Symmetric([[1, 0, 0], [0, 2, 0], [0, 0, 3]], {
    name: 'Three conventions', labels: ['Left', 'Middle', 'Right'],
    notes: 'Any shared convention is stable; the better-paying ones have larger basins.',
  }),
  hawk_dove_bourgeois: new Symmetric([[-1, 2, 0.5], [0, 1, 0.5], [-0.5, 1.5, 1]], {
    name: 'Hawk-Dove-Bourgeois', labels: ['Hawk', 'Dove', 'Bourgeois'],
    notes: "Maynard Smith's owner-fights, intruder-yields convention (V = 2, C = 4) takes over.",
  }),
  repeated_pd: new Symmetric([[30, 0, 30], [50, 10, 14], [30, 9, 30]], {
    name: "Repeated Prisoner's Dilemma (10 rounds)", labels: ['AllC', 'AllD', 'TFT'],
    notes: 'Always cooperate, always defect, tit for tat. TFT resists defectors but drifts neutrally against unconditional cooperators, who then let defectors back in.',
  }),
};
