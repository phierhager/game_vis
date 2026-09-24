// First-order learning rules on probability simplices.
// Mirrors src/gamevis/simplex.py; see there for the maths.
//
// A strategy x is a probability vector; its first-order signal g is the payoff
// vector (the gradient of the player's own expected payoff). Every rule maps
// (x, g) to a velocity tangent to the simplex.

export const RULES = ['replicator', 'projection', 'softmax_pg', 'logit', 'best_response'];

export const RULE_LABELS = {
  replicator: 'Replicator (multiplicative weights)',
  projection: 'Projected gradient',
  softmax_pg: 'Softmax policy gradient',
  logit: 'Logit response',
  best_response: 'Best response',
};

export const SMOOTH_RULES = new Set(['replicator', 'softmax_pg', 'logit']);

const EPS = 1e-12;

export function softmax(z) {
  let m = -Infinity;
  for (const v of z) m = Math.max(m, v);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

function dot(x, g) {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i] * g[i];
  return s;
}

export function replicator(x, g) {
  const m = dot(x, g);
  return x.map((xi, i) => xi * (g[i] - m));
}

// Projection of g onto the tangent cone of the simplex at x.
export function projection(x, g, tol = EPS) {
  const n = x.length;
  const zero = x.map((v) => v <= tol);
  let freeSum = 0;
  let freeCnt = 0;
  for (let i = 0; i < n; i++) {
    if (!zero[i]) {
      freeSum += g[i];
      freeCnt += 1;
    }
  }
  // zero coordinates join the free set, largest signal first, while they beat the mean
  const zs = [];
  for (let i = 0; i < n; i++) if (zero[i]) zs.push(g[i]);
  zs.sort((a, b) => b - a);
  let mu = freeSum / Math.max(freeCnt, 1);
  let sum = freeSum;
  let cnt = freeCnt;
  for (const v of zs) {
    const next = (sum + v) / (cnt + 1);
    if (v > next) {
      sum += v;
      cnt += 1;
      mu = next;
    } else break;
  }
  return g.map((gi, i) => {
    const v = gi - mu;
    return zero[i] && v < 0 ? 0 : v;
  });
}

export function softmaxPG(x, g) {
  return replicator(x, replicator(x, g));
}

export function logit(x, g, temperature = 0.05) {
  const s = softmax(g.map((v) => v / temperature));
  return s.map((si, i) => si - x[i]);
}

export function bestResponseVector(g, x = null, tol = EPS) {
  let m = -Infinity;
  for (const v of g) m = Math.max(m, v);
  const best = g.map((v) => (v >= m - tol ? 1 : 0));
  const nb = best.reduce((a, b) => a + b, 0);
  if (x) {
    const kept = x.map((xi, i) => xi * best[i]);
    const tot = kept.reduce((a, b) => a + b, 0);
    if (tot > tol) return kept.map((v) => v / tot);
  }
  return best.map((v) => v / nb);
}

export function bestResponse(x, g) {
  const br = bestResponseVector(g, x);
  return br.map((b, i) => b - x[i]);
}

export function field(rule, x, g, temperature = 0.05) {
  switch (rule) {
    case 'replicator': return replicator(x, g);
    case 'projection': return projection(x, g);
    case 'softmax_pg': return softmaxPG(x, g);
    case 'logit': return logit(x, g, temperature);
    case 'best_response': return bestResponse(x, g);
    default: throw new Error(`unknown rule ${rule}`);
  }
}

export function projectToSimplex(y) {
  const n = y.length;
  const u = [...y].sort((a, b) => b - a);
  let css = 0;
  let theta = 0;
  for (let i = 0; i < n; i++) {
    css += u[i];
    const t = (css - 1) / (i + 1);
    if (u[i] - t > 0) theta = t;
  }
  return y.map((v) => Math.max(v - theta, 0));
}

export function discreteStep(rule, x, g, eta, temperature = 0.05) {
  switch (rule) {
    case 'replicator': {
      let m = -Infinity;
      for (const v of g) m = Math.max(m, v);
      const w = x.map((xi, i) => xi * Math.exp(eta * (g[i] - m)));
      const s = w.reduce((a, b) => a + b, 0);
      return w.map((v) => v / s);
    }
    case 'projection':
      return projectToSimplex(x.map((xi, i) => xi + eta * g[i]));
    case 'softmax_pg': {
      const r = replicator(x, g);
      return softmax(x.map((xi, i) => Math.log(Math.max(xi, 1e-300)) + eta * r[i]));
    }
    case 'logit': {
      const s = softmax(g.map((v) => v / temperature));
      return x.map((xi, i) => (1 - eta) * xi + eta * s[i]);
    }
    case 'best_response': {
      const br = bestResponseVector(g, x);
      return x.map((xi, i) => (1 - eta) * xi + eta * br[i]);
    }
    default: throw new Error(`unknown rule ${rule}`);
  }
}

export function clean(x) {
  const c = x.map((v) => Math.max(v, 0));
  const s = c.reduce((a, b) => a + b, 0);
  return c.map((v) => v / s);
}

// ---------------------------------------------------------------------------
// Two actions: x = P(first action), d = payoff advantage of the first action.
// ---------------------------------------------------------------------------

export function binaryField(rule, x, d, temperature = 0.05) {
  switch (rule) {
    case 'replicator': return x * (1 - x) * d;
    case 'projection': {
      const v = d / 2;
      if ((x <= EPS && v < 0) || (x >= 1 - EPS && v > 0)) return 0;
      return v;
    }
    case 'softmax_pg': { const w = x * (1 - x); return 2 * w * w * d; }
    case 'logit': return 0.5 * (1 + Math.tanh(d / (2 * temperature))) - x;
    case 'best_response': return d > EPS ? 1 - x : d < -EPS ? -x : 0;
    default: throw new Error(`unknown rule ${rule}`);
  }
}

function logOdds(x) {
  return Math.log(Math.max(x, 1e-300)) - Math.log(Math.max(1 - x, 1e-300));
}

export function binaryDiscreteStep(rule, x, d, eta, temperature = 0.05) {
  switch (rule) {
    case 'replicator': return 0.5 * (1 + Math.tanh((logOdds(x) + eta * d) / 2));
    case 'projection': return Math.min(1, Math.max(0, x + (eta * d) / 2));
    case 'softmax_pg': return 0.5 * (1 + Math.tanh((logOdds(x) + 2 * eta * x * (1 - x) * d) / 2));
    case 'logit': return (1 - eta) * x + eta * 0.5 * (1 + Math.tanh(d / (2 * temperature)));
    case 'best_response': {
      const target = d > EPS ? 1 : d < -EPS ? 0 : x;
      return (1 - eta) * x + eta * target;
    }
    default: throw new Error(`unknown rule ${rule}`);
  }
}

// Uniform sample on the simplex with n vertices.
export function randomSimplex(rand, n) {
  const e = Array.from({ length: n }, () => -Math.log(Math.max(rand(), 1e-300)));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}
