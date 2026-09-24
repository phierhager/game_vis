// Triangle flows: one population, three strategies.

import { tokens, onThemeChange, alpha } from '../ui/theme.js';
import { Plot, polyline, dot, ring, marker, text, baryToXY, xyToBary, TRI_H, trianglePath, MARKER_NAMES } from '../ui/plot.js';
import { FlowField } from '../ui/flow.js';
import { LineChart } from '../ui/chart.js';
import { initPage, bindRange, fixed } from '../ui/common.js';
import * as nf from '../lib/normalform.js';
import { discreteStep } from '../lib/simplex.js';
import { mulberry32 } from '../lib/rng.js';

initPage();

const TIME_RATE = 2.5;
const DT = 0.01;
const DISCRETE_RATE = 14;
const MAX_RUNS = 12;

const state = {
  key: 'rps_zero_sum',
  game: nf.SYMMETRIC_PRESETS.rps_zero_sum,
  rule: 'replicator',
  temperature: 0.1,
  discrete: false,
  eta: 0.2,
  playing: true,
  runs: [],
  hover: null,
  seed: 5,
};

const ui = {
  game: document.getElementById('game'),
  rule: document.getElementById('rule'),
  discrete: document.getElementById('discrete'),
  play: document.getElementById('play'),
  random: document.getElementById('random'),
  clear: document.getElementById('clear'),
  tempControl: document.getElementById('temp-control'),
  etaControl: document.getElementById('eta-control'),
  matrix: document.getElementById('matrix'),
  notes: document.getElementById('game-notes'),
  caption: document.getElementById('tri-caption'),
  legend: document.getElementById('tri-legend'),
  readout: document.getElementById('readout'),
  timeLegend: document.getElementById('time-legend'),
};

bindRange('temp', (v) => (10 ** v).toFixed(2), (v) => { state.temperature = 10 ** v; changed(false); });
bindRange('eta', (v) => v.toFixed(2), (v) => { state.eta = v; });

ui.game.addEventListener('change', () => {
  if (ui.game.value !== 'custom') setGame(ui.game.value, nf.SYMMETRIC_PRESETS[ui.game.value]);
});
ui.rule.addEventListener('change', () => {
  state.rule = ui.rule.value;
  ui.tempControl.hidden = state.rule !== 'logit';
  changed(false);
});
ui.discrete.addEventListener('change', () => {
  state.discrete = ui.discrete.checked;
  ui.etaControl.hidden = !state.discrete;
  restartRuns();
});
ui.play.addEventListener('click', () => {
  state.playing = !state.playing;
  ui.play.textContent = state.playing ? 'Pause' : 'Play';
});
ui.random.addEventListener('click', () => {
  const rand = mulberry32(state.seed++);
  state.runs = [];
  for (let i = 0; i < 10; i++) {
    const e = [0, 1, 2].map(() => -Math.log(Math.max(rand(), 1e-9)));
    const s = e[0] + e[1] + e[2];
    addRun(e.map((v) => 0.02 + 0.94 * (v / s)), false);
  }
});
ui.clear.addEventListener('click', () => { state.runs = []; drawChart(); });

function setGame(key, game) {
  state.key = key;
  state.game = game;
  ui.game.value = key;
  buildMatrix();
  changed(true);
}

function changed(clear) {
  flow.setField(fieldXY);
  flow.clear();
  drawBg();
  updateText();
  if (clear) state.runs = defaultStarts().map((x) => freshRun(x));
  else restartRuns();
  drawChart();
}

function restartRuns() {
  state.runs = state.runs.map((r) => freshRun(r.xs[0]));
}

function defaultStarts() {
  switch (state.key) {
    case 'rps_zero_sum': return [[0.6, 0.2, 0.2], [0.45, 0.3, 0.25]];
    case 'rps_stable': return [[0.8, 0.1, 0.1]];
    case 'rps_unstable': return [[0.36, 0.33, 0.31]];
    case 'repeated_pd': return [[0.1, 0.05, 0.85], [0.3, 0.3, 0.4]];
    default: return [[0.3, 0.3, 0.4], [0.5, 0.3, 0.2]];
  }
}

// ---------------------------------------------------------------------------
// Matrix editor
// ---------------------------------------------------------------------------

function buildMatrix() {
  const g = state.game;
  const t = ui.matrix;
  t.replaceChildren();
  const head = document.createElement('tr');
  head.appendChild(document.createElement('th'));
  for (const lab of g.labels) {
    const th = document.createElement('th');
    th.textContent = `vs ${lab}`;
    head.appendChild(th);
  }
  t.appendChild(head);
  g.A.forEach((row, i) => {
    const tr = document.createElement('tr');
    const th = document.createElement('th');
    th.className = 'rowh';
    th.textContent = g.labels[i];
    tr.appendChild(th);
    row.forEach((v, j) => {
      const td = document.createElement('td');
      const inp = document.createElement('input');
      inp.type = 'number';
      inp.step = '0.5';
      inp.id = `m${i}${j}`;
      inp.value = String(v);
      inp.setAttribute('aria-label', `payoff of ${g.labels[i]} against ${g.labels[j]}`);
      inp.addEventListener('change', readMatrix);
      td.appendChild(inp);
      tr.appendChild(td);
    });
    t.appendChild(tr);
  });
}

function readMatrix() {
  const A = [0, 1, 2].map((i) => [0, 1, 2].map((j) => {
    const x = parseFloat(document.getElementById(`m${i}${j}`).value);
    return Number.isFinite(x) ? x : 0;
  }));
  const g = new nf.Symmetric(A, { name: 'Custom game', labels: state.game.labels, notes: 'Your own payoffs: each entry is what the row strategy earns against the column strategy.' });
  state.key = 'custom';
  state.game = g;
  ui.game.value = 'custom';
  changed(true);
}

// ---------------------------------------------------------------------------
// Triangle
// ---------------------------------------------------------------------------

const plot = new Plot(document.getElementById('plot-tri'), { xDomain: [-0.08, 1.08], yDomain: [-0.09, TRI_H + 0.07], equal: true, aspect: 0.86, pad: { top: 16, right: 16, bottom: 16, left: 16 }, maxHeight: 700 });

const insideTri = (X, Y) => {
  const b = xyToBary(X, Y);
  return b[0] >= 0 && b[1] >= 0 && b[2] >= 0;
};

function fieldBary(x) {
  return state.game.velocity(x, state.rule, state.temperature);
}

function fieldXY(X, Y) {
  const b = xyToBary(X, Y).map((v) => Math.max(v, 0));
  const s = b[0] + b[1] + b[2];
  const v = fieldBary(b.map((c) => c / s));
  return [v[1] + 0.5 * v[2], TRI_H * v[2]];
}

const flow = new FlowField(plot, {
  count: 520,
  sample: () => {
    const e = [0, 1, 2].map(() => -Math.log(Math.max(Math.random(), 1e-9)));
    const s = e[0] + e[1] + e[2];
    return baryToXY(e.map((v) => v / s));
  },
  inside: insideTri,
  field: fieldXY,
  maxAge: 90,
  stepFrac: 0.004,
});

function drawBg() {
  const T = tokens();
  const ctx = plot.ctx.bg;
  plot.clear('bg');
  trianglePath(plot, ctx);
  ctx.fillStyle = T.surface;
  ctx.fill();
  ctx.strokeStyle = T.axis;
  ctx.lineWidth = 1;
  ctx.stroke();
  // faint grid of equal shares
  ctx.strokeStyle = T.grid;
  ctx.beginPath();
  for (const f of [1 / 3, 2 / 3]) {
    for (let k = 0; k < 3; k++) {
      const a = [0, 0, 0];
      const b = [0, 0, 0];
      a[k] = f;
      b[k] = f;
      a[(k + 1) % 3] = 1 - f;
      b[(k + 2) % 3] = 1 - f;
      const [ax, ay] = baryToXY(a);
      const [bx, by] = baryToXY(b);
      ctx.moveTo(plot.sx(ax), plot.sy(ay));
      ctx.lineTo(plot.sx(bx), plot.sy(by));
    }
  }
  ctx.stroke();
  const g = state.game;
  const corners = [[0, 0], [1, 0], [0.5, TRI_H]];
  // Corner names, each keyed by the colour its share takes in the chart below.
  g.labels.forEach((lab, i) => {
    const [x, y] = corners[i];
    const cy = plot.sy(y) + (i === 2 ? -14 : 18);
    ctx.font = '600 13px system-ui, sans-serif';
    const w = ctx.measureText(lab).width;
    const left = i === 2 ? plot.sx(x) - w / 2 + 7 : i === 0 ? plot.sx(x) - w + 4 : plot.sx(x) + 10;
    dot(ctx, left - 9, cy, 4, T.series[i]);
    text(ctx, lab, left, cy, { color: T.ink, size: 13, weight: 600, baseline: 'middle' });
  });
  const kinds = new Set();
  for (const rp of g.restPoints()) {
    kinds.add(rp.stability);
    const [x, y] = baryToXY(rp.x);
    marker(ctx, plot.sx(x), plot.sy(y), rp.stability, T, 6.5);
  }
  renderLegend([...kinds]);
}

function renderLegend(kinds) {
  ui.legend.replaceChildren();
  const glyphs = {
    sink: '<span class="key-dot" style="background:var(--ink)"></span>',
    saddle: '<span class="key-dot" style="background:linear-gradient(90deg,var(--ink) 50%,var(--surface) 50%);border:1.5px solid var(--ink)"></span>',
    source: '<span class="key-ring"></span>',
    center: '<span class="key-ring" style="background:radial-gradient(var(--ink) 30%,transparent 34%)"></span>',
    degenerate: '<span class="key-ring" style="border-radius:2px;transform:rotate(45deg) scale(0.85)"></span>',
  };
  for (const k of ['sink', 'saddle', 'source', 'center', 'degenerate']) {
    if (!kinds.includes(k)) continue;
    const li = document.createElement('li');
    li.innerHTML = glyphs[k];
    const s = document.createElement('span');
    s.textContent = `rest point, ${MARKER_NAMES[k]}`;
    li.appendChild(s);
    ui.legend.appendChild(li);
  }
  const li = document.createElement('li');
  li.innerHTML = '<span class="key-line" style="color:var(--ink)"></span><span>population path</span>';
  ui.legend.appendChild(li);
}

function freshRun(x0) {
  const x = [...x0];
  return { x, xs: [x0], t: 0, n: 0, ts: [0], shares: [[...x0]], acc: 0, done: false };
}

function addRun(x, redraw = true) {
  state.runs.push(freshRun(x));
  if (state.runs.length > MAX_RUNS) state.runs.shift();
  if (redraw) drawChart();
}

function advance(dtSec) {
  const g = state.game;
  for (const r of state.runs) {
    if (r.done) continue;
    if (state.discrete) {
      r.acc += dtSec * DISCRETE_RATE;
      while (r.acc >= 1) {
        r.acc -= 1;
        r.x = discreteStep(state.rule, r.x, g.signal(r.x), state.eta, state.temperature);
        r.n += 1;
        r.t = r.n;
        rec(r);
        if (r.n > 800) r.done = true;
      }
    } else {
      const steps = Math.max(1, Math.round((dtSec * TIME_RATE) / DT));
      for (let s = 0; s < steps; s++) {
        r.x = g.step(r.x, DT, state.rule, state.temperature);
        r.t += DT;
        r.n += 1;
        if (r.n % 4 === 0) rec(r);
      }
      if (r.t > 150) r.done = true;
    }
  }
}

function rec(r) {
  r.xs.push([...r.x]);
  r.ts.push(r.t);
  r.shares.push([...r.x]);
}

function drawRuns() {
  const T = tokens();
  const ctx = plot.ctx.fg;
  plot.clear('fg');
  state.runs.forEach((r, i) => {
    const latest = i === state.runs.length - 1;
    const pts = r.xs.map((x) => {
      const [X, Y] = baryToXY(x);
      return [plot.sx(X), plot.sy(Y)];
    });
    polyline(ctx, pts, { color: latest ? T.ink : T.ink2, width: latest ? 2 : 1.4, halo: alpha(T.surface, 0.85) });
    if (state.discrete) for (const [x, y] of pts.slice(-400)) dot(ctx, x, y, 1.6, latest ? T.ink : T.ink2);
    ring(ctx, pts[0][0], pts[0][1], 4.5, T.ink2, 1.4, T.surface);
    const last = pts[pts.length - 1];
    dot(ctx, last[0], last[1], 4.5, latest ? T.ink : T.ink2, T.surface, 2);
  });
  if (state.hover) {
    const [X, Y] = baryToXY(state.hover);
    ring(ctx, plot.sx(X), plot.sy(Y), 3, T.ink, 1.2);
  }
}

plot.on('down', (e) => {
  if (!insideTri(e.x, e.y)) return;
  const b = xyToBary(e.x, e.y).map((v) => Math.max(v, 0.002));
  const s = b[0] + b[1] + b[2];
  addRun(b.map((v) => v / s));
});
plot.on('move', (e) => {
  state.hover = insideTri(e.x, e.y) ? xyToBary(e.x, e.y) : null;
  updateReadout();
});
plot.on('leave', () => { state.hover = null; updateReadout(); });

function updateReadout() {
  const g = state.game;
  if (!state.hover) {
    ui.readout.textContent = 'Hover the triangle to read a population mix.';
    return;
  }
  const x = state.hover;
  const f = g.signal(x);
  const avg = g.meanPayoff(x);
  ui.readout.innerHTML = [
    `<b>${g.labels.map((l, i) => `${l} ${Math.round(100 * x[i])}%`).join(' · ')}</b>`,
    `payoffs ${f.map((v) => fixed(v)).join(' / ')}`,
    `average ${fixed(avg)}`,
  ].map((s) => `<span>${s}</span>`).join('');
}

function updateText() {
  const g = state.game;
  ui.notes.textContent = g.notes || '';
  const rule = ui.rule.options[ui.rule.selectedIndex].text;
  ui.caption.textContent = `${g.name} under ${rule.toLowerCase()}${state.discrete ? ', runs in discrete steps' : ''}. Markers are rest points of the replicator dynamics with their stability.`;
}

// ---------------------------------------------------------------------------
// Chart
// ---------------------------------------------------------------------------

const chart = new LineChart(document.getElementById('chart-time'), {
  yDomain: [0, 1], yTicks: [0, 0.5, 1], height: 170, valueFormat: (v) => `${Math.round(100 * v)}%`,
  series: [0, 1, 2].map((i) => ({ key: `x${i}`, label: '', color: `s${i + 1}` })),
});

function drawChart() {
  const g = state.game;
  chart.series.forEach((s, i) => { s.label = g.labels[i]; });
  const r = state.runs[state.runs.length - 1];
  if (r) chart.setData(r.ts, { x0: r.shares.map((x) => x[0]), x1: r.shares.map((x) => x[1]), x2: r.shares.map((x) => x[2]) });
  else chart.setData([], {});
  chart.draw();
  ui.timeLegend.replaceChildren();
  g.labels.forEach((lab, i) => {
    const li = document.createElement('li');
    const k = document.createElement('span');
    k.className = 'key-line';
    k.style.color = `var(--s${i + 1})`;
    const s = document.createElement('span');
    s.textContent = lab;
    li.append(k, s);
    ui.timeLegend.appendChild(li);
  });
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
  if (state.playing) advance(dt);
  flow.frame();
  drawRuns();
  if (frameNo % 6 === 0) drawChart();
  requestAnimationFrame(loop);
}

for (const b of document.querySelectorAll('[data-try]')) {
  b.addEventListener('click', () => {
    const k = b.getAttribute('data-try');
    state.discrete = false;
    ui.discrete.checked = false;
    state.rule = 'replicator';
    ui.rule.value = 'replicator';
    if (k === 'rps-discrete') {
      state.discrete = true;
      ui.discrete.checked = true;
      setGame('rps_zero_sum', nf.SYMMETRIC_PRESETS.rps_zero_sum);
    } else if (k === 'coord-br') {
      state.rule = 'best_response';
      ui.rule.value = 'best_response';
      setGame('coordination3', nf.SYMMETRIC_PRESETS.coordination3);
    } else if (k === 'pd-tft') {
      setGame('repeated_pd', nf.SYMMETRIC_PRESETS.repeated_pd);
    }
    ui.etaControl.hidden = !state.discrete;
    ui.tempControl.hidden = state.rule !== 'logit';
    document.querySelector('.controls').scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
}

onThemeChange(() => {
  drawBg();
  drawChart();
  flow.clear();
});
plot.onResize(() => drawBg());

setGame('rps_zero_sum', nf.SYMMETRIC_PRESETS.rps_zero_sum);
updateReadout();
requestAnimationFrame(loop);
