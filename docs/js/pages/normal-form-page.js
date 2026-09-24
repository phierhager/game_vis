// Two-by-two flows: phase portraits of 2x2 games under first-order learning.

import { tokens, onThemeChange, alpha, mixRgb } from '../ui/theme.js';
import { Plot, frame, ticks, axisLabels, polyline, dot, ring, marker, text, raster, gridLines, MARKER_NAMES } from '../ui/plot.js';
import { FlowField } from '../ui/flow.js';
import { LineChart } from '../ui/chart.js';
import { initPage, bindRange, fixed } from '../ui/common.js';
import * as nf from '../lib/normalform.js';
import { mulberry32 } from '../lib/rng.js';

initPage();

const TIME_RATE = 2.5;
const DT = 0.01;
const DISCRETE_RATE = 14; // iterations per second
const MAX_RUNS = 12;

const state = {
  key: 'stag_hunt',
  game: nf.BIMATRIX_PRESETS.stag_hunt,
  rule: 'replicator',
  temperature: 10 ** -1.4,
  discrete: false,
  eta: 0.15,
  optimistic: false,
  ratio: 0,
  playing: true,
  basins: true,
  runs: [],
  hover: null,
  seed: 3,
};

const rates = () => [2 ** (state.ratio / 2), 2 ** (-state.ratio / 2)];

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

const ui = {
  game: document.getElementById('game'),
  rule: document.getElementById('rule'),
  discrete: document.getElementById('discrete'),
  optimistic: document.getElementById('optimistic'),
  basins: document.getElementById('basins'),
  play: document.getElementById('play'),
  random: document.getElementById('random'),
  clear: document.getElementById('clear'),
  tempControl: document.getElementById('temp-control'),
  etaControl: document.getElementById('eta-control'),
  optControl: document.getElementById('opt-control'),
  matrix: document.getElementById('matrix'),
  notes: document.getElementById('game-notes'),
  caption: document.getElementById('portrait-caption'),
  legend: document.getElementById('portrait-legend'),
  readout: document.getElementById('readout'),
  timeLegend: document.getElementById('time-legend'),
};

const temp = bindRange('temp', (v) => (10 ** v).toFixed(v < -1 ? 3 : 2), (v) => { state.temperature = 10 ** v; gameChanged(false); });
bindRange('eta', (v) => v.toFixed(2), (v) => { state.eta = v; });
const ratio = bindRange('ratio', (v) => {
  if (Math.abs(v) < 1e-9) return 'equal';
  const f = 2 ** Math.abs(v);
  return `${v > 0 ? 'row' : 'column'} ${Number.isInteger(f) ? f : f.toFixed(1)}×`;
}, (v) => { state.ratio = v; gameChanged(false); });

ui.game.addEventListener('change', () => {
  if (ui.game.value !== 'custom') setGame(ui.game.value, nf.BIMATRIX_PRESETS[ui.game.value]);
});
ui.rule.addEventListener('change', () => {
  state.rule = ui.rule.value;
  ui.tempControl.hidden = state.rule !== 'logit';
  gameChanged(false);
});
ui.discrete.addEventListener('change', () => {
  state.discrete = ui.discrete.checked;
  ui.etaControl.hidden = !state.discrete;
  ui.optControl.hidden = !state.discrete;
  resetRuns();
});
ui.optimistic.addEventListener('change', () => { state.optimistic = ui.optimistic.checked; resetRuns(); });
ui.basins.addEventListener('change', () => { state.basins = ui.basins.checked; drawPortraitBg(); });
ui.play.addEventListener('click', () => {
  state.playing = !state.playing;
  ui.play.textContent = state.playing ? 'Pause' : 'Play';
});
ui.random.addEventListener('click', () => {
  const rand = mulberry32(state.seed++);
  state.runs = [];
  for (let i = 0; i < 12; i++) addRun(0.04 + 0.92 * rand(), 0.04 + 0.92 * rand(), false);
});
ui.clear.addEventListener('click', () => { state.runs = []; drawChart(); });

function setGame(key, game) {
  state.key = key;
  state.game = game;
  ui.game.value = key;
  buildMatrix();
  gameChanged(true);
}

function gameChanged(clearRuns) {
  flow.setField((p, q) => state.game.velocity(p, q, state.rule, rates(), state.temperature));
  flow.clear();
  computeBasins();
  drawPortraitBg();
  drawAtlas();
  updateText();
  if (clearRuns) resetRuns();
  else for (const r of state.runs) Object.assign(r, freshRun(r.pts[0][0], r.pts[0][1]));
}

function resetRuns() {
  const starts = state.runs.length ? state.runs.map((r) => r.pts[0]) : defaultStarts();
  state.runs = [];
  for (const [p, q] of starts) addRun(p, q, false);
}

function defaultStarts() {
  if (state.key === 'matching_pennies') return [[0.75, 0.5]];
  return [[0.2, 0.85], [0.8, 0.3], [0.55, 0.6]];
}

// ---------------------------------------------------------------------------
// Payoff table
// ---------------------------------------------------------------------------

function buildMatrix() {
  const g = state.game;
  const t = ui.matrix;
  t.replaceChildren();
  const head = document.createElement('tr');
  head.appendChild(document.createElement('th'));
  for (const a of g.colActions) {
    const th = document.createElement('th');
    th.textContent = `column: ${a}`;
    head.appendChild(th);
  }
  t.appendChild(head);
  for (let i = 0; i < 2; i++) {
    const tr = document.createElement('tr');
    const th = document.createElement('th');
    th.className = 'rowh';
    th.textContent = `row: ${g.rowActions[i]}`;
    tr.appendChild(th);
    for (let j = 0; j < 2; j++) {
      const td = document.createElement('td');
      const a = numberInput(`a${i}${j}`, g.A[i][j], `row player's payoff for ${g.rowActions[i]} vs ${g.colActions[j]}`);
      const b = numberInput(`b${i}${j}`, g.B[i][j], `column player's payoff for ${g.rowActions[i]} vs ${g.colActions[j]}`);
      const sep = document.createElement('span');
      sep.className = 'sep';
      sep.textContent = ',';
      td.append(a, sep, b);
      tr.appendChild(td);
    }
    t.appendChild(tr);
  }
}

function numberInput(id, value, label) {
  const inp = document.createElement('input');
  inp.type = 'number';
  inp.step = '0.5';
  inp.id = id;
  inp.value = String(value);
  inp.setAttribute('aria-label', label);
  inp.addEventListener('change', readMatrix);
  return inp;
}

function readMatrix() {
  const v = (id) => {
    const x = parseFloat(document.getElementById(id).value);
    return Number.isFinite(x) ? x : 0;
  };
  const A = [[v('a00'), v('a01')], [v('a10'), v('a11')]];
  const B = [[v('b00'), v('b01')], [v('b10'), v('b11')]];
  const g = new nf.Bimatrix(A, B, { name: 'Custom game', rowActions: state.game.rowActions, colActions: state.game.colActions, notes: 'Your own payoffs.' });
  state.key = 'custom';
  state.game = g;
  ui.game.value = 'custom';
  gameChanged(true);
}

// ---------------------------------------------------------------------------
// Portrait
// ---------------------------------------------------------------------------

const plot = new Plot(document.getElementById('plot-portrait'), { equal: true, aspect: 0.9, pad: { top: 30, right: 30, bottom: 50, left: 60 }, maxHeight: 700 });
const flow = new FlowField(plot, {
  count: 520,
  sample: () => [Math.random(), Math.random()],
  inside: (x, y) => x >= 0 && x <= 1 && y >= 0 && y <= 1,
  field: () => [0, 0],
  maxAge: 80,
  stepFrac: 0.004,
});

let basinCodes = null;
let sinks = [];
const BN = 64;

function computeBasins() {
  const g = state.game;
  const eqs = g.nashEquilibria();
  sinks = eqs.filter((e) => g.stability(e, stabilityRule()) === 'sink');
  basinCodes = null;
  if (sinks.length < 2) return;
  const codes = new Int8Array(BN * BN);
  for (let j = 0; j < BN; j++) {
    for (let i = 0; i < BN; i++) {
      let p = (i + 0.5) / BN;
      let q = (j + 0.5) / BN;
      for (let s = 0; s < 800; s++) [p, q] = g.step(p, q, 0.05, state.rule, rates(), state.temperature);
      let best = -1;
      let bd = 0.12;
      sinks.forEach(([sp, sq], k) => {
        const d = Math.hypot(p - sp, q - sq);
        if (d < bd) {
          bd = d;
          best = k;
        }
      });
      codes[j * BN + i] = best;
    }
  }
  basinCodes = codes;
}

function stabilityRule() {
  return ['replicator', 'projection', 'softmax_pg'].includes(state.rule) ? state.rule : 'replicator';
}

function drawPortraitBg() {
  const T = tokens();
  const g = state.game;
  const ctx = plot.ctx.bg;
  plot.clear('bg');
  if (state.basins && basinCodes) {
    const cols = [T.s1, T.s2, T.s3].map((c) => mixRgb(c, T.surface, T.dark ? 0.15 : 0.2));
    raster(plot, ctx, (x, y) => {
      const i = Math.min(BN - 1, Math.floor(x * BN));
      const j = Math.min(BN - 1, Math.floor(y * BN));
      const c = basinCodes[j * BN + i];
      return c < 0 ? null : [...cols[c % 3], 255];
    }, { res: BN, smooth: false });
  }
  if (g.isSymmetric) {
    ctx.strokeStyle = T.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(plot.sx(0), plot.sy(0));
    ctx.lineTo(plot.sx(1), plot.sy(1));
    ctx.stroke();
  }
  const ind = g.indifference();
  ctx.save();
  ctx.setLineDash([5, 4]);
  ctx.strokeStyle = T.muted;
  ctx.lineWidth = 1.2;
  if (ind.rowIndifferentAtQ !== null) {
    ctx.beginPath();
    ctx.moveTo(plot.sx(0), plot.sy(ind.rowIndifferentAtQ));
    ctx.lineTo(plot.sx(1), plot.sy(ind.rowIndifferentAtQ));
    ctx.stroke();
  }
  if (ind.colIndifferentAtP !== null) {
    ctx.beginPath();
    ctx.moveTo(plot.sx(ind.colIndifferentAtP), plot.sy(0));
    ctx.lineTo(plot.sx(ind.colIndifferentAtP), plot.sy(1));
    ctx.stroke();
  }
  ctx.restore();
  frame(plot, ctx, T);
  ticks(plot, ctx, T, { x: [0, 0.5, 1], y: [0, 0.5, 1] });
  axisLabels(plot, ctx, T, { x: `P(row plays ${g.rowActions[0]})`, y: `P(column plays ${g.colActions[0]})` });
  // corner names
  const ab = (labels) => {
    for (let n = 1; n < 6; n++) if (labels[0].slice(0, n) !== labels[1].slice(0, n)) return [labels[0].slice(0, n), labels[1].slice(0, n)];
    return labels;
  };
  const ra = ab(g.rowActions);
  const ca = ab(g.colActions);
  for (const [x, y, i, j] of [[1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 0, 1, 1]]) {
    text(ctx, `${ra[i]},${ca[j]}`, plot.sx(x) + (x ? 8 : -8), plot.sy(y) + (y ? -8 : 16), { color: T.muted, size: 11.5, align: x ? 'left' : 'right' });
  }
  const kinds = new Set();
  for (const e of g.nashEquilibria()) {
    const k = g.stability(e, stabilityRule());
    kinds.add(k);
    marker(ctx, plot.sx(e[0]), plot.sy(e[1]), k, T, 6.5);
  }
  renderLegend([...kinds]);
}

function renderLegend(kinds) {
  ui.legend.replaceChildren();
  const add = (html, label) => {
    const li = document.createElement('li');
    li.innerHTML = html;
    const s = document.createElement('span');
    s.textContent = label;
    li.appendChild(s);
    ui.legend.appendChild(li);
  };
  for (const k of ['sink', 'saddle', 'source', 'center', 'degenerate']) {
    if (!kinds.includes(k)) continue;
    const glyph = {
      sink: '<span class="key-dot" style="background:var(--ink)"></span>',
      saddle: '<span class="key-dot" style="background:linear-gradient(90deg,var(--ink) 50%,var(--surface) 50%);border:1.5px solid var(--ink)"></span>',
      source: '<span class="key-ring"></span>',
      center: '<span class="key-ring" style="background:radial-gradient(var(--ink) 30%,transparent 34%)"></span>',
      degenerate: '<span class="key-ring" style="border-radius:2px;transform:rotate(45deg)"></span>',
    }[k];
    add(glyph, `Nash equilibrium, ${MARKER_NAMES[k]}`);
  }
  add('<span class="key-line" style="background:none;border-top:2px dashed var(--muted);height:0"></span>', 'indifference line');
  add('<span class="key-line" style="color:var(--ink)"></span>', 'learning run');
}

function freshRun(p, q) {
  const [ur, uc] = state.game.payoffs(p, q);
  return { p, q, pts: [[p, q]], t: 0, n: 0, prev: null, done: false, ts: [0], ur: [ur], uc: [uc], acc: 0 };
}

function addRun(p, q, redrawChart = true) {
  state.runs.push(freshRun(p, q));
  if (state.runs.length > MAX_RUNS) state.runs.shift();
  if (redrawChart) drawChart();
}

function advanceRuns(dtSec) {
  const g = state.game;
  for (const r of state.runs) {
    if (r.done) continue;
    if (state.discrete) {
      r.acc += dtSec * DISCRETE_RATE;
      while (r.acc >= 1) {
        r.acc -= 1;
        const sig = g.advantages(r.p, r.q);
        let use = sig;
        if (state.optimistic && r.prev) use = [2 * sig[0] - r.prev[0], 2 * sig[1] - r.prev[1]];
        r.prev = sig;
        [r.p, r.q] = g.discreteStep(r.p, r.q, state.rule, state.eta, rates(), state.temperature, use);
        r.n += 1;
        r.t = r.n;
        record(r);
        if (r.n >= 600) r.done = true;
      }
    } else {
      const steps = Math.max(1, Math.round((dtSec * TIME_RATE) / DT));
      for (let s = 0; s < steps; s++) {
        [r.p, r.q] = g.step(r.p, r.q, DT, state.rule, rates(), state.temperature);
        r.t += DT;
        r.n += 1;
        if (r.n % 4 === 0) record(r);
      }
      if (r.t > 80) r.done = true;
    }
  }
}

function record(r) {
  r.pts.push([r.p, r.q]);
  const [ur, uc] = state.game.payoffs(r.p, r.q);
  r.ts.push(r.t);
  r.ur.push(ur);
  r.uc.push(uc);
}

function drawRuns() {
  const T = tokens();
  const ctx = plot.ctx.fg;
  plot.clear('fg');
  state.runs.forEach((r, i) => {
    const latest = i === state.runs.length - 1;
    const pts = r.pts.map(([p, q]) => [plot.sx(p), plot.sy(q)]);
    polyline(ctx, pts, { color: latest ? T.ink : T.ink2, width: latest ? 2 : 1.4, halo: alpha(T.surface, 0.85), alpha: latest ? 1 : 0.85 });
    if (state.discrete) for (const [x, y] of pts.slice(-400)) dot(ctx, x, y, 1.7, latest ? T.ink : T.ink2);
    ring(ctx, pts[0][0], pts[0][1], 4.5, T.ink2, 1.4, T.surface);
    const last = pts[pts.length - 1];
    dot(ctx, last[0], last[1], 4.5, latest ? T.ink : T.ink2, T.surface, 2);
  });
  if (state.hover) {
    const { p, q } = state.hover;
    ring(ctx, plot.sx(p), plot.sy(q), 3, T.ink, 1.2);
  }
}

plot.on('down', (e) => {
  if (!e.inside) return;
  addRun(Math.min(1, Math.max(0, e.x)), Math.min(1, Math.max(0, e.y)));
});
plot.on('move', (e) => {
  state.hover = e.inside ? { p: e.x, q: e.y } : null;
  updateReadout();
});
plot.on('leave', () => { state.hover = null; updateReadout(); });

function updateReadout() {
  const g = state.game;
  if (!state.hover) {
    ui.readout.textContent = 'Hover the square to read the flow at a point.';
    return;
  }
  const { p, q } = state.hover;
  const [vp, vq] = g.velocity(p, q, state.rule, rates(), state.temperature);
  const [ur, uc] = g.payoffs(p, q);
  const [dr, dc] = g.advantages(p, q);
  ui.readout.innerHTML = [
    `<b>p = ${fixed(p)}, q = ${fixed(q)}</b>`,
    `signal Δu = (${fixed(dr)}, ${fixed(dc)})`,
    `flow (${fixed(vp, 3)}, ${fixed(vq, 3)})`,
    `payoffs ${fixed(ur)} / ${fixed(uc)}`,
  ].map((s) => `<span>${s}</span>`).join('');
}

function updateText() {
  const g = state.game;
  const notes = state.key === 'custom' ? 'Your own payoffs. Each cell reads (row payoff, column payoff).' : g.notes;
  ui.notes.textContent = notes;
  const rule = ui.rule.options[ui.rule.selectedIndex].text;
  ui.caption.textContent = `${g.name} under ${rule.toLowerCase()}${state.discrete ? ' (runs use discrete steps; tracers show the continuous-time flow)' : ''}. Tints mark which stable outcome each start reaches.`;
}

// ---------------------------------------------------------------------------
// Atlas
// ---------------------------------------------------------------------------

const atlas = new Plot(document.getElementById('plot-atlas'), { xDomain: [0, 2], yDomain: [-1, 1], equal: true, aspect: 0.95, pad: { top: 12, right: 12, bottom: 46, left: 52 }, layers: ['bg', 'fg'], maxHeight: 380 });
let atlasHover = null;

function normalizedTS(g) {
  if (!g.isSymmetric) return null;
  const R = g.A[0][0];
  const S = g.A[0][1];
  const T = g.A[1][0];
  const P = g.A[1][1];
  if (!(R > P)) return null;
  return [(T - P) / (R - P), (S - P) / (R - P)];
}

function drawAtlas() {
  const T = tokens();
  const ctx = atlas.ctx.bg;
  atlas.clear('bg');
  const regions = [
    [0, 1, 0, 1, 'Harmony'],
    [1, 2, 0, 1, 'Snowdrift'],
    [0, 1, -1, 0, 'Stag Hunt'],
    [1, 2, -1, 0, "Prisoner's\nDilemma"],
  ];
  regions.forEach(([x0, x1, y0, y1, name], i) => {
    ctx.fillStyle = alpha(T.ink, i % 3 === 0 ? 0.035 : 0.065);
    ctx.fillRect(atlas.sx(x0), atlas.sy(y1), atlas.sx(x1) - atlas.sx(x0), atlas.sy(y0) - atlas.sy(y1));
    const lines = name.split('\n');
    lines.forEach((ln, k) => text(ctx, ln, atlas.sx((x0 + x1) / 2), atlas.sy((y0 + y1) / 2) + (k - (lines.length - 1) / 2) * 15, { color: T.ink2, size: 12.5, weight: 600, align: 'center', baseline: 'middle' }));
  });
  gridLines(atlas, ctx, T, { x: [1], y: [0] });
  frame(atlas, ctx, T);
  ticks(atlas, ctx, T, { x: [0, 1, 2], y: [-1, 0, 1] });
  axisLabels(atlas, ctx, T, { x: 'temptation T', y: "sucker's payoff S" });
  drawAtlasFg();
}

function drawAtlasFg() {
  const T = tokens();
  const ctx = atlas.ctx.fg;
  atlas.clear('fg');
  const ts = normalizedTS(state.game);
  if (ts) {
    const [x, y] = [Math.min(2, Math.max(0, ts[0])), Math.min(1, Math.max(-1, ts[1]))];
    dot(ctx, atlas.sx(x), atlas.sy(y), 5.5, T.ink, T.surface, 2);
  }
  if (atlasHover) {
    const { x, y } = atlasHover;
    ring(ctx, atlas.sx(x), atlas.sy(y), 5, T.ink2, 1.4);
    text(ctx, `T ${x.toFixed(2)}, S ${y.toFixed(2)}`, atlas.sx(x) + (x > 1.3 ? -9 : 9), atlas.sy(y) - 9, { color: T.ink, size: 11.5, align: x > 1.3 ? 'right' : 'left', halo: T.surface, family: 'mono' });
  }
}

atlas.on('move', (e) => {
  atlasHover = e.inside ? { x: e.x, y: e.y } : null;
  drawAtlasFg();
});
atlas.on('leave', () => { atlasHover = null; drawAtlasFg(); });
atlas.on('down', (e) => {
  if (!e.inside) return;
  const Tv = Math.round(e.x * 20) / 20;
  const Sv = Math.round(e.y * 20) / 20;
  const g = nf.symmetric2x2(1, Sv, Tv, 0);
  g.notes = `${g.name}: reward 1, sucker's payoff ${Sv}, temptation ${Tv}, punishment 0.`;
  setGame('custom', g);
});

// ---------------------------------------------------------------------------
// Chart
// ---------------------------------------------------------------------------

const chart = new LineChart(document.getElementById('chart-time'), {
  yDomain: [0, 1], height: 150,
  series: [{ key: 'ur', label: 'row payoff', color: 'ink' }, { key: 'uc', label: 'column payoff', color: 'muted', dash: [5, 4] }],
});

function drawChart() {
  const g = state.game;
  const all = [...g.A.flat(), ...g.B.flat()];
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const pad = (hi - lo) * 0.05 || 1;
  chart.setYDomain([lo - pad, hi + pad], niceTicks(lo, hi));
  const r = state.runs[state.runs.length - 1];
  if (r) chart.setData(r.ts, { ur: r.ur, uc: r.uc });
  else chart.setData([], {});
  chart.draw();
  ui.timeLegend.innerHTML = '<li><span class="key-line" style="color:var(--ink)"></span><span>row payoff</span></li><li><span class="key-line" style="background:none;border-top:2px dashed var(--muted);height:0"></span><span>column payoff</span></li>' + `<li><span>${state.discrete ? 'x axis: step' : 'x axis: time'}</span></li>`;
}

function niceTicks(lo, hi) {
  const span = hi - lo || 1;
  const step = [0.5, 1, 2, 2.5, 5, 10].find((s) => span / s <= 5) || 10;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Math.round(v * 100) / 100);
  return out;
}

// ---------------------------------------------------------------------------
// Loop
// ---------------------------------------------------------------------------

let last = performance.now();
let frameNo = 0;
function loop(now) {
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  frameNo += 1;
  if (state.playing) advanceRuns(dt);
  flow.frame();
  drawRuns();
  if (frameNo % 6 === 0) drawChart();
  requestAnimationFrame(loop);
}

for (const b of document.querySelectorAll('[data-try]')) {
  b.addEventListener('click', () => {
    const k = b.getAttribute('data-try');
    ui.discrete.checked = false;
    state.discrete = false;
    state.optimistic = false;
    ui.optimistic.checked = false;
    ratio.set(0);
    state.ratio = 0;
    ui.rule.value = 'replicator';
    state.rule = 'replicator';
    if (k === 'mp-discrete') {
      ui.discrete.checked = true;
      state.discrete = true;
      setGame('matching_pennies', nf.BIMATRIX_PRESETS.matching_pennies);
    } else if (k === 'stag-speed') {
      ratio.set(-2);
      state.ratio = -2;
      setGame('stag_hunt', nf.BIMATRIX_PRESETS.stag_hunt);
    } else if (k === 'chicken-br') {
      ui.rule.value = 'best_response';
      state.rule = 'best_response';
      setGame('chicken', nf.BIMATRIX_PRESETS.chicken);
    } else if (k === 'pd-logit') {
      ui.rule.value = 'logit';
      state.rule = 'logit';
      temp.set(0.3);
      state.temperature = 10 ** 0.3;
      setGame('prisoners_dilemma', nf.BIMATRIX_PRESETS.prisoners_dilemma);
    }
    ui.etaControl.hidden = !state.discrete;
    ui.optControl.hidden = !state.discrete;
    ui.tempControl.hidden = state.rule !== 'logit';
    document.querySelector('.controls').scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
}

onThemeChange(() => {
  drawPortraitBg();
  drawAtlas();
  drawChart();
  flow.clear();
});

plot.onResize(() => drawPortraitBg());
atlas.onResize(() => drawAtlas());

setGame('stag_hunt', nf.BIMATRIX_PRESETS.stag_hunt);
updateReadout();
requestAnimationFrame(loop);
