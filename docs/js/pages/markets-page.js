// Duopoly price flows: gradient play, best replies, sympathy, Edgeworth cycles.

import { tokens, onThemeChange, alpha, mixRgb } from '../ui/theme.js';
import { Plot, frame, ticks, axisLabels, polyline, dot, ring, marker, text, raster, gridLines } from '../ui/plot.js';
import { FlowField } from '../ui/flow.js';
import { LineChart, niceStep } from '../ui/chart.js';
import { initPage, bindRange, fixed } from '../ui/common.js';
import * as mk from '../lib/markets.js';
import { QPricing } from '../lib/qlearning.js';
import { mulberry32 } from '../lib/rng.js';

initPage();

const state = {
  modelKey: 'logit',
  mu: 0.25,
  gamma: 0.5,
  lambda: 0,
  adjust: 'gradient',
  shade: 'joint',
  playing: true,
  runs: [],
  hover: null,
  seed: 11,
  model: null,
  view: [1, 2.4],
  nash: null,
  cartel: null,
  eq: null,
  br1: [],
  br2: [],
};

const ui = {
  model: document.getElementById('model'),
  adjust: document.getElementById('adjust'),
  shade: document.getElementById('shade'),
  play: document.getElementById('play'),
  random: document.getElementById('random'),
  clear: document.getElementById('clear'),
  muControl: document.getElementById('mu-control'),
  gammaControl: document.getElementById('gamma-control'),
  legend: document.getElementById('plane-legend'),
  readout: document.getElementById('readout'),
  bench: document.getElementById('bench-rows'),
  notes: document.getElementById('model-notes'),
  title: document.getElementById('plane-title'),
};

bindRange('mu', (v) => v.toFixed(2), (v) => { state.mu = v; rebuild(); });
bindRange('gamma', (v) => v.toFixed(2), (v) => { state.gamma = v; rebuild(); });
bindRange('lambda', (v) => v.toFixed(2), (v) => { state.lambda = v; recomputeEquilibrium(); });

ui.model.addEventListener('change', () => {
  state.modelKey = ui.model.value;
  ui.muControl.hidden = state.modelKey !== 'logit';
  ui.gammaControl.hidden = state.modelKey !== 'linear';
  rebuild();
});
ui.adjust.addEventListener('change', () => { state.adjust = ui.adjust.value; restartRuns(); });
ui.shade.addEventListener('change', () => { state.shade = ui.shade.value; drawPlaneBg(); });
ui.play.addEventListener('click', () => {
  state.playing = !state.playing;
  ui.play.textContent = state.playing ? 'Pause' : 'Play';
});
ui.random.addEventListener('click', () => {
  const rand = mulberry32(state.seed++);
  const [lo, hi] = state.view;
  state.runs = [];
  for (let i = 0; i < 8; i++) addRun(lo + (hi - lo) * (0.05 + 0.9 * rand()), lo + (hi - lo) * (0.05 + 0.9 * rand()), false);
});
ui.clear.addEventListener('click', () => { state.runs = []; drawChart(); });

const isQuantity = () => state.modelKey === 'cournot';
const timeRate = () => ({ logit: 14, linear: 3, cournot: 3 })[state.modelKey];

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

function rebuild() {
  let m;
  if (state.modelKey === 'logit') {
    m = new mk.LogitBertrand({ mu: state.mu, lo: 1, hi: 5 });
    const cartel = m.collusive()[0];
    const nash = m.nash()[0];
    const top = Math.max(cartel, nash) + Math.max(0.25, 0.3 * (cartel - 1));
    m.lo = 1;
    m.hi = Math.ceil(top * 10) / 10;
  } else if (state.modelKey === 'linear') {
    m = new mk.LinearBertrand({ gamma: state.gamma, lo: 0, hi: 10 });
    const top = Math.max(m.collusiveClosedForm(), m.nashClosedForm()) * 1.3;
    m.hi = Math.ceil(top * 10) / 10;
  } else {
    m = new mk.Cournot({ lo: 0, hi: 0.6 });
  }
  state.model = m;
  state.view = [m.lo, m.hi];
  state.nash = m.nash();
  state.cartel = m.collusive();
  plane.setDomain(state.view, state.view);
  flow.setField(field);
  flow.clear();
  recomputeEquilibrium(false);
  updateNotes();
  state.runs = [];
  defaultRuns();
}

function recomputeEquilibrium(redrawRuns = true) {
  const m = state.model;
  state.eq = state.lambda === 0 ? state.nash : state.lambda === 1 ? state.cartel : m.equilibrium(state.lambda);
  const grid = 70;
  const [lo, hi] = state.view;
  state.br1 = [];
  state.br2 = [];
  for (let i = 0; i <= grid; i++) {
    const o = lo + ((hi - lo) * i) / grid;
    state.br1.push([m.bestResponse(0, o, state.lambda, 201), o]);
    state.br2.push([o, m.bestResponse(1, o, state.lambda, 201)]);
  }
  flow.setField(field);
  flow.clear();
  drawPlaneBg();
  renderBench();
  if (redrawRuns) restartRuns();
}

function field(p1, p2) {
  return state.model ? state.model.velocity(p1, p2, [1, 1], state.lambda) : [0, 0];
}

function defaultRuns() {
  const [lo, hi] = state.view;
  const c = state.cartel[0];
  addRun(c, c, false);
  addRun(lo + 0.08 * (hi - lo), hi - 0.08 * (hi - lo), false);
  addRun(hi - 0.05 * (hi - lo), hi - 0.3 * (hi - lo), false);
  drawChart();
}

function restartRuns() {
  const starts = state.runs.map((r) => r.pts[0]);
  state.runs = [];
  for (const [a, b] of starts) addRun(a, b, false);
  drawChart();
}

function updateNotes() {
  const notes = {
    logit: 'Each firm sells a differentiated product; customers choose by a logit rule with an outside option. Smaller μ means closer substitutes and fiercer price competition.',
    linear: 'Demand for each product falls with its own price and rises with the rival\'s. γ close to 1 means near-perfect substitutes.',
    cournot: 'Firms choose quantities; the price clears the market. Here the cartel restricts output below the Nash quantities.',
  };
  ui.notes.textContent = notes[state.modelKey];
  ui.title.textContent = isQuantity() ? 'Quantity plane' : 'Price plane';
}

function renderBench() {
  const m = state.model;
  const rows = [
    ['Nash (λ = 0)', state.nash],
    ['cartel (λ = 1)', state.cartel],
  ];
  if (state.lambda > 0 && state.lambda < 1) rows.push([`rest point at λ = ${state.lambda.toFixed(2)}`, state.eq]);
  ui.bench.replaceChildren();
  const unit = isQuantity() ? 'quantity' : 'price';
  for (const [label, p] of rows) {
    const [a] = m.profit(p[0], p[1]);
    const div = document.createElement('div');
    const lab = document.createElement('span');
    lab.textContent = label;
    const val = document.createElement('span');
    val.className = 'val';
    val.textContent = `${unit} ${fixed(p[0], 3)} · profit ${fixed(a, 3)}`;
    div.append(lab, val);
    ui.bench.appendChild(div);
  }
}

// ---------------------------------------------------------------------------
// Plane
// ---------------------------------------------------------------------------

const plane = new Plot(document.getElementById('plot-plane'), { equal: true, aspect: 0.9, pad: { top: 18, right: 22, bottom: 50, left: 62 }, maxHeight: 700 });
const flow = new FlowField(plane, {
  count: 480,
  sample: () => {
    const [lo, hi] = state.view;
    return [lo + (hi - lo) * Math.random(), lo + (hi - lo) * Math.random()];
  },
  inside: (x, y) => x >= state.view[0] && x <= state.view[1] && y >= state.view[0] && y <= state.view[1],
  field: (x, y) => field(x, y),
  maxAge: 80,
  stepFrac: 0.004,
});

function drawPlaneBg() {
  const T = tokens();
  const m = state.model;
  const ctx = plane.ctx.bg;
  plane.clear('bg');
  const [lo, hi] = state.view;
  const n = 90;
  const vals = new Float64Array(n * n);
  const lens = new Uint8Array(n * n);
  const [piN] = m.profit(state.nash[0], state.nash[1]);
  let vmin = Infinity;
  let vmax = -Infinity;
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const x = lo + ((i + 0.5) / n) * (hi - lo);
      const y = lo + ((j + 0.5) / n) * (hi - lo);
      const [a, b] = m.profit(x, y);
      const v = state.shade === 'joint' ? a + b : state.shade === 'p1' ? a : b;
      vals[j * n + i] = v;
      lens[j * n + i] = a > piN + 1e-9 && b > piN + 1e-9 ? 1 : 0;
      vmin = Math.min(vmin, v);
      vmax = Math.max(vmax, v);
    }
  }
  const hue = state.shade === 'p1' ? T.s1 : state.shade === 'p2' ? T.s2 : T.muted;
  const lensRgb = mixRgb(T.s3, T.surface, 1);
  raster(plane, ctx, (x, y) => {
    const i = Math.min(n - 1, Math.max(0, Math.floor(((x - lo) / (hi - lo)) * n)));
    const j = Math.min(n - 1, Math.max(0, Math.floor(((y - lo) / (hi - lo)) * n)));
    const t = (vals[j * n + i] - vmin) / (vmax - vmin || 1);
    let c = mixRgb(hue, T.surface, (state.shade === 'joint' ? 0.55 : 0.4) * t);
    if (lens[j * n + i]) c = c.map((v, k) => Math.round(v + 0.2 * (lensRgb[k] - v)));
    return [...c, 255];
  }, { res: n, smooth: true });
  frame(plane, ctx, T);
  const step = niceStep((hi - lo) / 5);
  const tk = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) tk.push(Math.round(v * 1000) / 1000);
  ticks(plane, ctx, T, { x: tk, y: tk, fmt: (v) => `${v}` });
  const what = isQuantity() ? 'quantity' : 'price';
  axisLabels(plane, ctx, T, { x: `${what} of firm 1`, y: `${what} of firm 2` });
  // best-reply curves (for the current λ)
  polyline(ctx, state.br1.map(([a, b]) => [plane.sx(a), plane.sy(b)]), { color: T.s1, width: 2 });
  polyline(ctx, state.br2.map(([a, b]) => [plane.sx(a), plane.sy(b)]), { color: T.s2, width: 2 });
  // benchmarks
  const [pn] = state.nash;
  const [pc] = state.cartel;
  if (state.lambda === 0) marker(ctx, plane.sx(pn), plane.sy(pn), 'sink', T, 6.5);
  else ring(ctx, plane.sx(pn), plane.sy(pn), 5.5, T.ink, 1.6, T.surface);
  diamond(ctx, plane.sx(pc), plane.sy(pc), 6.5, T, state.lambda === 1);
  if (state.lambda > 0 && state.lambda < 1) marker(ctx, plane.sx(state.eq[0]), plane.sy(state.eq[1]), 'sink', T, 6.5);
  text(ctx, 'Nash', plane.sx(pn) + 9, plane.sy(pn) + 16, { color: T.ink, size: 12, halo: T.surface });
  text(ctx, 'cartel', plane.sx(pc) + 10, plane.sy(pc) + 4, { color: T.ink, size: 12, halo: T.surface });
  renderLegend();
}

function diamond(ctx, x, y, r, T, filled) {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(Math.PI / 4);
  ctx.fillStyle = T.surface;
  ctx.fillRect(-r * 0.95, -r * 0.95, r * 1.9, r * 1.9);
  ctx.fillStyle = filled ? T.ink : T.surface;
  ctx.strokeStyle = T.ink;
  ctx.lineWidth = 1.5;
  ctx.fillRect(-r * 0.62, -r * 0.62, r * 1.24, r * 1.24);
  ctx.strokeRect(-r * 0.62, -r * 0.62, r * 1.24, r * 1.24);
  ctx.restore();
}

function renderLegend() {
  const shadeName = { joint: 'joint profit', p1: "firm 1's profit", p2: "firm 2's profit" }[state.shade];
  const lam = state.lambda === 0 ? '' : ` (with λ = ${state.lambda.toFixed(2)})`;
  ui.legend.innerHTML = [
    `<li><span class="key-line" style="color:var(--s1)"></span><span>firm 1's best reply${lam}</span></li>`,
    `<li><span class="key-line" style="color:var(--s2)"></span><span>firm 2's best reply${lam}</span></li>`,
    '<li><span class="key-dot" style="background:var(--ink)"></span><span>rest point of gradient play</span></li>',
    '<li><span class="key-ring" style="border-radius:2px;transform:rotate(45deg) scale(0.8)"></span><span>cartel</span></li>',
    '<li><span class="key-dot" style="background:color-mix(in srgb, var(--s3) 35%, var(--surface))"></span><span>both beat Nash</span></li>',
    `<li><span>shading: ${shadeName} (stronger shade, more profit)</span></li>`,
  ].join('');
}

function freshRun(a, b) {
  const [u, v] = state.model.profit(a, b);
  return { a, b, pts: [[a, b]], t: 0, n: 0, acc: 0, ts: [0], u1: [u], u2: [v], done: false };
}

function addRun(a, b, redraw = true) {
  state.runs.push(freshRun(a, b));
  if (state.runs.length > 10) state.runs.shift();
  if (redraw) drawChart();
}

function advance(dtSec) {
  const m = state.model;
  for (const r of state.runs) {
    if (r.done) continue;
    if (state.adjust === 'gradient') {
      const dt = 0.02;
      const steps = Math.max(1, Math.round((dtSec * timeRate()) / dt));
      for (let s = 0; s < steps; s++) {
        [r.a, r.b] = m.step(r.a, r.b, dt, [1, 1], state.lambda);
        r.t += dt;
        r.n += 1;
        if (r.n % 5 === 0) rec(r);
      }
      if (r.t > 400) r.done = true;
    } else {
      r.acc += dtSec * 3;
      while (r.acc >= 1) {
        r.acc -= 1;
        if (r.n % 2 === 0) r.a = m.bestResponse(0, r.b, state.lambda, 201);
        else r.b = m.bestResponse(1, r.a, state.lambda, 201);
        r.n += 1;
        r.t = r.n;
        rec(r);
        if (r.n >= 40) r.done = true;
      }
    }
  }
}

function rec(r) {
  r.pts.push([r.a, r.b]);
  const [u, v] = state.model.profit(r.a, r.b);
  r.ts.push(r.t);
  r.u1.push(u);
  r.u2.push(v);
}

function drawRuns() {
  const T = tokens();
  const ctx = plane.ctx.fg;
  plane.clear('fg');
  state.runs.forEach((r, i) => {
    const latest = i === state.runs.length - 1;
    const pts = r.pts.map(([a, b]) => [plane.sx(a), plane.sy(b)]);
    polyline(ctx, pts, { color: latest ? T.ink : T.ink2, width: latest ? 2 : 1.4, halo: alpha(T.surface, 0.85) });
    if (state.adjust === 'br') for (const [x, y] of pts) dot(ctx, x, y, 2.2, latest ? T.ink : T.ink2);
    ring(ctx, pts[0][0], pts[0][1], 4.5, T.ink2, 1.4, T.surface);
    const last = pts[pts.length - 1];
    dot(ctx, last[0], last[1], 4.5, latest ? T.ink : T.ink2, T.surface, 2);
  });
}

plane.on('down', (e) => {
  if (!e.inside) return;
  addRun(e.x, e.y);
});
plane.on('move', (e) => {
  state.hover = e.inside ? [e.x, e.y] : null;
  updateReadout();
});
plane.on('leave', () => { state.hover = null; updateReadout(); });

function updateReadout() {
  if (!state.hover) {
    ui.readout.textContent = 'Hover the plane to read profits and marginal profits.';
    return;
  }
  const m = state.model;
  const [a, b] = state.hover;
  const [u, v] = m.profit(a, b);
  const [g1, g2] = m.marginalProfit(a, b);
  const what = isQuantity() ? 'q' : 'p';
  ui.readout.innerHTML = [
    `<b>${what}₁ = ${fixed(a, 3)}, ${what}₂ = ${fixed(b, 3)}</b>`,
    `profits ${fixed(u, 3)} / ${fixed(v, 3)}`,
    `marginal profits ${fixed(g1, 3)} / ${fixed(g2, 3)}`,
  ].map((s) => `<span>${s}</span>`).join('');
}

const chart = new LineChart(document.getElementById('chart-profit'), {
  yDomain: [0, 0.4], height: 170, valueFormat: (v) => v.toFixed(3),
  series: [{ key: 'u1', label: 'firm 1', color: 's1' }, { key: 'u2', label: 'firm 2', color: 's2' }],
});

function drawChart() {
  const r = state.runs[state.runs.length - 1];
  const all = r ? [...r.u1, ...r.u2] : [0, 0.4];
  const hi = Math.max(...all, state.model.profit(state.cartel[0], state.cartel[1])[0]);
  const lo = Math.min(0, ...all);
  const step = niceStep((hi - lo) / 3);
  const tk = [];
  for (let v = Math.floor(lo / step) * step; v <= hi + step * 0.5; v += step) tk.push(Math.round(v * 1000) / 1000);
  chart.setYDomain([tk[0], tk[tk.length - 1]], tk);
  if (r) chart.setData(r.ts, { u1: r.u1, u2: r.u2 });
  else chart.setData([], {});
  chart.draw();
}

// ---------------------------------------------------------------------------
// Edgeworth cycles
// ---------------------------------------------------------------------------

const est = { capacity: 0.5, cost: 0.1, tick: 0.01, playing: true, path: [], acc: 0, model: null };
const eprice = new Plot(document.getElementById('plot-eprice'), { xDomain: [0, 80], yDomain: [0, 1], aspect: 0.42, minHeight: 220, maxHeight: 360, pad: { top: 14, right: 110, bottom: 34, left: 48 }, layers: ['bg', 'fg'] });
eprice.el.classList.add('static');
const ebr = new Plot(document.getElementById('plot-ebr'), { equal: true, aspect: 1, pad: { top: 12, right: 14, bottom: 46, left: 54 }, layers: ['bg', 'fg'], maxHeight: 380 });
ebr.el.classList.add('static');
const eCaption = document.getElementById('e-caption');

bindRange('cap', (v) => v.toFixed(2), (v) => { est.capacity = v; eRebuild(); });
bindRange('ecost', (v) => v.toFixed(2), (v) => { est.cost = v; eRebuild(); });
bindRange('tick', (v) => v.toFixed(3), (v) => { est.tick = v; eRebuild(); });
document.getElementById('eplay').addEventListener('click', (ev) => {
  est.playing = !est.playing;
  ev.currentTarget.textContent = est.playing ? 'Pause' : 'Play';
});
document.getElementById('erestart').addEventListener('click', () => eRebuild());

function eRebuild() {
  est.model = new mk.Edgeworth({ capacity: est.capacity, cost: est.cost, tick: est.tick });
  est.path = [[0.9, 0.9]];
  est.acc = 0;
  const e = est.model;
  const top = Math.max(0.6, e.monopolyPrice() + 0.05);
  ebr.setDomain([e.cost, top], [e.cost, top]);
  // best-reply curves on the price grid
  const grid = e.grid().filter((p) => p <= 0.85);
  est.br = grid.map((p) => [p, e.bestResponse(p)]);
  const serveAll = est.capacity >= 1 - est.cost;
  const long = e.simulate([0.9, 0.9], 400).slice(200).map((p) => p[0]);
  const lo = Math.min(...long);
  const hi = Math.max(...long);
  eCaption.textContent = serveAll
    ? `Each firm can serve the whole market at cost, so undercutting never stops paying: prices fall to one tick above cost (${(est.cost + est.tick).toFixed(3)}), the Bertrand paradox.`
    : `Each firm can serve at most ${est.capacity.toFixed(2)} of a market of ${(1 - est.cost).toFixed(2)} at cost. Prices keep cycling between ${lo.toFixed(2)} and ${hi.toFixed(2)}: undercut down to the floor, then relent to ${e.residualPrice().toFixed(2)}.`;
  drawEdgeworthBg();
}

function drawEdgeworthBg() {
  const T = tokens();
  const e = est.model;
  let ctx = eprice.ctx.bg;
  eprice.clear('bg');
  gridLines(eprice, ctx, T, { y: [0.25, 0.5, 0.75] });
  ctx.strokeStyle = T.axis;
  ctx.beginPath();
  ctx.moveTo(eprice.x0, Math.round(eprice.y1) + 0.5);
  ctx.lineTo(eprice.x1, Math.round(eprice.y1) + 0.5);
  ctx.stroke();
  ticks(eprice, ctx, T, { y: [0, 0.25, 0.5, 0.75, 1], fmt: (v) => `${v}` });
  const refs = [[e.monopolyPrice(), 'monopoly price'], [e.residualPrice(), 'relenting price'], [e.cost, 'cost']];
  for (const [y, lab] of refs) {
    ctx.strokeStyle = T.axis;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(eprice.x0, eprice.sy(y));
    ctx.lineTo(eprice.x1, eprice.sy(y));
    ctx.stroke();
    ctx.setLineDash([]);
    text(ctx, lab, eprice.x1 + 6, eprice.sy(y), { color: T.muted, size: 11, baseline: 'middle' });
  }
  axisLabels(eprice, ctx, T, { y: 'price' });
  text(ctx, 'period →', eprice.x1, eprice.h - 4, { color: T.muted, size: 11, align: 'right', baseline: 'bottom' });

  ctx = ebr.ctx.bg;
  ebr.clear('bg');
  const top = ebr.xd[1];
  ctx.strokeStyle = T.grid;
  ctx.beginPath();
  ctx.moveTo(ebr.sx(e.cost), ebr.sy(e.cost));
  ctx.lineTo(ebr.sx(top), ebr.sy(top));
  ctx.stroke();
  frame(ebr, ctx, T);
  const step = niceStep((top - e.cost) / 4);
  const tk = [];
  for (let v = Math.ceil(e.cost / step) * step; v <= top + 1e-9; v += step) tk.push(Math.round(v * 100) / 100);
  ticks(ebr, ctx, T, { x: tk, y: tk });
  axisLabels(ebr, ctx, T, { x: 'price of firm 1', y: 'price of firm 2' });
  const inView = est.br.filter(([p]) => p <= top);
  polyline(ctx, stairs(inView.map(([p, r]) => [ebr.sx(p), ebr.sy(Math.min(r, top))])), { color: T.s2, width: 1.8 });
  polyline(ctx, stairs(inView.map(([p, r]) => [ebr.sx(Math.min(r, top)), ebr.sy(p)]), true), { color: T.s1, width: 1.8 });
}

function stairs(pts, vertical = false) {
  const out = [];
  for (let i = 0; i < pts.length; i++) {
    if (i > 0) out.push(vertical ? [pts[i][0], pts[i - 1][1]] : [pts[i][0], pts[i - 1][1]]);
    out.push(pts[i]);
  }
  return out;
}

function eAdvance(dtSec) {
  if (!est.playing) return;
  est.acc += dtSec * 6;
  while (est.acc >= 1) {
    est.acc -= 1;
    const p = [...est.path[est.path.length - 1]];
    const i = (est.path.length - 1) % 2;
    p[i] = est.model.bestResponse(p[1 - i]);
    est.path.push(p);
    if (est.path.length > 2000) est.path.splice(0, est.path.length - 2000);
  }
}

function drawEdgeworth() {
  const T = tokens();
  const p = eprice;
  const ctx = p.ctx.fg;
  p.clear('fg');
  const window = 80;
  const start = Math.max(0, est.path.length - window - 1);
  const seg = est.path.slice(start);
  const x = (k) => p.sx(k);
  for (const [f, color] of [[0, T.s1], [1, T.s2]]) {
    const pts = [];
    seg.forEach((pp, k) => {
      if (k > 0) pts.push([x(k), p.sy(seg[k - 1][f])]);
      pts.push([x(k), p.sy(pp[f])]);
    });
    polyline(ctx, pts, { color, width: 2 });
  }
  const last = seg[seg.length - 1];
  dot(ctx, x(seg.length - 1), p.sy(last[0]), 3.5, T.s1, T.surface, 1.5);
  dot(ctx, x(seg.length - 1), p.sy(last[1]), 3.5, T.s2, T.surface, 1.5);
  const bctx = ebr.ctx.fg;
  ebr.clear('fg');
  const tail = est.path.slice(-24).map(([a, b]) => [ebr.sx(a), ebr.sy(b)]);
  polyline(bctx, tail, { color: T.ink, width: 1.3 });
  for (const [a, b] of tail) dot(bctx, a, b, 2.2, T.ink);
  if (tail.length) dot(bctx, tail[tail.length - 1][0], tail[tail.length - 1][1], 4.5, T.ink, T.surface, 2);
}


// ---------------------------------------------------------------------------
// Q-learning firms
// ---------------------------------------------------------------------------

const SCHEDULES = { fast: { beta: 1e-4, periods: 60000 }, baseline: { beta: 4e-6, periods: 1200000 } };
// seed 3 settles on one price and shows a clean punishment episode; later clicks draw new seeds
const qs = { qp: null, sess: null, done: 0, total: 0, block: 0, xs: [], p1: [], p2: [], seed: 3, training: false };
const qReadout = document.getElementById('qreadout');
const qCutBtn = document.getElementById('qcut');
const qCutCaption = document.getElementById('qcut-caption');
const qTrainChart = new LineChart(document.getElementById('chart-qtrain'), {
  yDomain: [1.4, 2.0], yTicks: [1.4, 1.6, 1.8, 2.0], height: 190, valueFormat: (v) => v.toFixed(3),
  series: [
    { key: 'p1', label: 'firm 1', color: 's1' },
    { key: 'p2', label: 'firm 2', color: 's2' },
    { key: 'mono', label: 'monopoly', color: 'muted', dash: [5, 4], width: 1.2 },
    { key: 'nash', label: 'Nash', color: 'muted', dash: [5, 4], width: 1.2 },
  ],
});
const qCutChart = new LineChart(document.getElementById('chart-qcut'), {
  yDomain: [1.4, 2.0], yTicks: [1.4, 1.6, 1.8, 2.0], height: 190, valueFormat: (v) => v.toFixed(3),
  series: [
    { key: 'p1', label: 'firm 1', color: 's1' },
    { key: 'p2', label: 'firm 2', color: 's2' },
    { key: 'mono', label: 'monopoly', color: 'muted', dash: [5, 4], width: 1.2 },
    { key: 'nash', label: 'Nash', color: 'muted', dash: [5, 4], width: 1.2 },
  ],
});

function qStart() {
  const sch = SCHEDULES[document.getElementById('qschedule').value];
  qs.qp = new QPricing({ beta: sch.beta });
  qs.sess = qs.qp.newSession(qs.seed++ * 9973);
  qs.total = sch.periods;
  qs.done = 0;
  qs.block = Math.max(1, Math.round(sch.periods / 150));
  qs.xs = [];
  qs.p1 = [];
  qs.p2 = [];
  qs.training = true;
  qCutBtn.disabled = true;
  qCutChart.setData([], {});
  qCutChart.draw();
  qCutCaption.textContent = 'Training…';
}

function qStep() {
  if (!qs.training) return;
  const perFrame = Math.max(qs.block, Math.round(qs.total / 90));
  const t0 = performance.now();
  let n = 0;
  while (qs.done < qs.total && n < perFrame && performance.now() - t0 < 12) {
    const [a, b] = qs.qp.train(qs.sess, qs.block);
    qs.done += qs.block;
    n += qs.block;
    qs.xs.push(qs.done / 1000);
    qs.p1.push(a);
    qs.p2.push(b);
  }
  const L = qs.xs.length;
  qTrainChart.setData(qs.xs, { p1: qs.p1, p2: qs.p2, mono: Array(L).fill(qs.qp.pMonopoly), nash: Array(L).fill(qs.qp.pNash) });
  qTrainChart.draw();
  qReadout.textContent = `period ${qs.done.toLocaleString()} of ${qs.total.toLocaleString()} · exploration ${(100 * Math.exp(-qs.qp.beta * qs.sess.t)).toFixed(1)}%`;
  if (qs.done >= qs.total) {
    qs.training = false;
    qCutBtn.disabled = false;
    const cyc = qs.qp.limitCycle(qs.sess, qs.sess.s);
    const gain = qs.qp.cycleProfitGain(qs.sess, qs.sess.s);
    const prices = cyc.map((st) => (qs.qp.prices[Math.floor(st / qs.qp.m)] + qs.qp.prices[st % qs.qp.m]) / 2);
    const avg = prices.reduce((x, y) => x + y, 0) / prices.length;
    qReadout.innerHTML = `<span><b>long-run price ${avg.toFixed(3)}</b></span><span>Nash ${qs.qp.pNash.toFixed(3)} · monopoly ${qs.qp.pMonopoly.toFixed(3)}</span><span>profit gain ${(100 * gain).toFixed(0)}% of the way from Nash to monopoly</span>${cyc.length > 1 ? `<span>play cycles through ${cyc.length} states</span>` : ''}`;
    qCut();
  }
}

function qCut() {
  const qp = qs.qp;
  const s0 = qp.limitCycle(qs.sess, qs.sess.s)[0];
  const before = 3;
  const path = qp.play(qs.sess, s0, 20, before);
  const xs = path.map((_, i) => i - before);
  const p1 = path.map(([a]) => qp.prices[a]);
  const p2 = path.map(([, b]) => qp.prices[b]);
  // LineChart's x axis starts at 0, so shift by the lead-in
  qCutChart.setData(xs.map((x) => x + before), { p1, p2, mono: Array(xs.length).fill(qp.pMonopoly), nash: Array(xs.length).fill(qp.pNash) });
  qCutChart.draw();
  const low = Math.min(...p2.slice(before + 1, before + 4));
  qCutCaption.textContent = `Firm 1 undercuts at x = ${before} (from ${p1[before - 1].toFixed(2)} to ${p1[before].toFixed(2)}). Firm 2's price then falls as low as ${low.toFixed(2)}${low < p2[before - 1] - 1e-9 ? ': a punishment' : ''}, and prices climb back afterwards. Gradient play has no memory, so it cannot punish.`;
}

document.getElementById('qtrain').addEventListener('click', qStart);
qCutBtn.addEventListener('click', qCut);

// ---------------------------------------------------------------------------
// Loop
// ---------------------------------------------------------------------------

let last = performance.now();
let frameNo = 0;
function loop(now) {
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  frameNo += 1;
  if (state.playing) advance(dt);
  flow.frame();
  drawRuns();
  eAdvance(dt);
  drawEdgeworth();
  qStep();
  if (frameNo % 6 === 0) drawChart();
  requestAnimationFrame(loop);
}

onThemeChange(() => {
  drawPlaneBg();
  drawEdgeworthBg();
  drawChart();
  qTrainChart.draw();
  qCutChart.draw();
  flow.clear();
});
plane.onResize(() => { if (state.model) drawPlaneBg(); });
eprice.onResize(() => { if (est.model) drawEdgeworthBg(); });
ebr.onResize(() => { if (est.model) drawEdgeworthBg(); });

rebuild();
eRebuild();
updateReadout();
qStart();
requestAnimationFrame(loop);
