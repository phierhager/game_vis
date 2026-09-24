// Learning to signal: linked views of a sender-receiver game under first-order learning.

import { tokens, onThemeChange, alpha, mixRgb } from '../ui/theme.js';
import { Plot, frame, ticks, axisLabels, polyline, dot, ring, marker, text, raster, baryToXY, xyToBary, TRI_H, trianglePath, gridLines } from '../ui/plot.js';
import { FlowField } from '../ui/flow.js';
import { LineChart } from '../ui/chart.js';
import { initPage, bindRange, pct, fixed } from '../ui/common.js';
import { cornerGlyphs, signalFlow } from '../ui/signaling-draw.js';
import * as sg from '../lib/signaling.js';
import { binaryField } from '../lib/simplex.js';
import { mulberry32 } from '../lib/rng.js';

initPage();

// ---------------------------------------------------------------------------
// Scenarios and state
// ---------------------------------------------------------------------------

const SCENARIOS = {
  lewis_even: { family: 'lewis2', prior: 0.5, ratio: 0, rule: 'replicator' },
  lewis_skewed: { family: 'lewis2', prior: 0.8, ratio: 0, rule: 'replicator' },
  impatient: { family: 'lewis2', prior: 0.8, ratio: -3, rule: 'replicator' },
  handicap_cheap: { family: 'handicap', costLow: 0.6, ratio: 0, rule: 'replicator' },
  handicap_dear: { family: 'handicap', costLow: 1.3, ratio: 0, rule: 'replicator' },
  lewis3: { family: 'lewis3', ratio: 0, rule: 'replicator' },
};

const H = 0.05; // integration substep
const TIME_RATE = 6; // model time per second at speed 1
const T_MAX = 600;
const REC_EVERY = 2; // record history every 2 substeps (0.1 time units)
const N_SWARM_BINARY = 200;
const N_SWARM_GENERAL = 80;

const state = {
  family: 'lewis2',
  prior: 0.5,
  costLow: 0.6,
  rule: 'replicator',
  ratio: 0,
  exploration: 0,
  speed: 0,
  playing: true,
  swarmOn: true,
  t: 0,
  steps: 0,
  game: null,
  start: null,
  cur: null,
  hist: null,
  swarm: [],
  swarmStart: [],
  outcomes: null,
  hover: -1,
  seed: 7,
  pausedAtEnd: false,
};

const rates = () => [2 ** (state.ratio / 2), 2 ** (-state.ratio / 2)];
const opts = () => ({ rule: state.rule, rates: rates(), temperature: 0.05, exploration: state.exploration });
const binary = () => state.family !== 'lewis3';

function makeGame() {
  if (state.family === 'lewis2') return sg.lewis(2, state.prior);
  if (state.family === 'handicap') return sg.costlySignaling({ priorHigh: 0.4, costHigh: 0.3, costLow: state.costLow });
  return sg.lewis(3);
}

// ---------------------------------------------------------------------------
// DOM
// ---------------------------------------------------------------------------

const ui = {
  scenario: document.getElementById('scenario'),
  rule: document.getElementById('rule'),
  play: document.getElementById('play'),
  restart: document.getElementById('restart'),
  shuffle: document.getElementById('shuffle'),
  swarm: document.getElementById('swarm'),
  priorControl: document.getElementById('prior-control'),
  costControl: document.getElementById('cost-control'),
  readout: document.getElementById('readout'),
  stackbar: document.getElementById('stackbar'),
  stackrows: document.getElementById('stackrows'),
  outcomeHint: document.getElementById('outcome-hint'),
  basinBlock: document.getElementById('basin-block'),
  basinHint: document.getElementById('basin-hint'),
  planeTitle: document.getElementById('plane-title'),
  planeCaption: document.getElementById('plane-caption'),
  planeHint: document.getElementById('plane-hint'),
  senderCaption: document.getElementById('sender-caption'),
  receiverCaption: document.getElementById('receiver-caption'),
  senderHint: document.getElementById('sender-hint'),
  receiverHint: document.getElementById('receiver-hint'),
  flowHint: document.getElementById('flow-hint'),
};

const prior = bindRange('prior', (v) => v.toFixed(2), (v) => { state.prior = v; rebuild({ keepStart: true }); });
const cost = bindRange('cost', (v) => v.toFixed(2), (v) => { state.costLow = v; rebuild({ keepStart: true }); });
const ratio = bindRange('ratio', ratioLabel, (v) => { state.ratio = v; rebuild({ keepStart: true }); });
bindRange('explore', (v) => v.toFixed(3), (v) => { state.exploration = v; rebuild({ keepStart: true }); });
bindRange('speed', (v) => `${(2 ** v).toFixed(v < 0 ? 2 : 0)}×`, (v) => { state.speed = v; });

function ratioLabel(v) {
  if (Math.abs(v) < 1e-9) return 'equal';
  const f = 2 ** Math.abs(v);
  const s = Number.isInteger(f) ? `${f}` : f.toFixed(1);
  return v > 0 ? `sender ${s}×` : `receiver ${s}×`;
}

ui.scenario.addEventListener('change', () => applyScenario(ui.scenario.value));
ui.rule.addEventListener('change', () => { state.rule = ui.rule.value; rebuild({ keepStart: true }); });
ui.play.addEventListener('click', () => setPlaying(!state.playing));
ui.restart.addEventListener('click', () => restart());
ui.shuffle.addEventListener('click', () => { state.seed += 1; rebuild({ keepStart: false }); });
ui.swarm.addEventListener('change', () => { state.swarmOn = ui.swarm.checked; });

function setPlaying(on) {
  state.playing = on;
  if (on && state.t >= T_MAX) restart();
  ui.play.textContent = on ? 'Pause' : 'Play';
}

// ---------------------------------------------------------------------------
// Plots
// ---------------------------------------------------------------------------

const squarePad = { top: 34, right: 46, bottom: 50, left: 58 };
const plotS = new Plot(document.getElementById('plot-sender'), { equal: true, aspect: 0.98, pad: squarePad });
const plotR = new Plot(document.getElementById('plot-receiver'), { equal: true, aspect: 0.98, pad: squarePad });
const plotP = new Plot(document.getElementById('plot-plane'), { equal: true, aspect: 0.98, xDomain: [-1, 1], yDomain: [-1, 1], pad: { top: 30, right: 22, bottom: 50, left: 58 } });
const plotF = new Plot(document.getElementById('plot-flow'), { aspect: 0.72, pad: { top: 0, right: 0, bottom: 0, left: 0 }, layers: ['fg'], minHeight: 200, maxHeight: 360 });
plotF.el.classList.add('static');
const plotB = new Plot(document.getElementById('plot-basin'), { equal: true, aspect: 0.9, pad: { top: 8, right: 16, bottom: 42, left: 52 }, layers: ['bg', 'fg'], maxHeight: 320 });

const payoffChart = new LineChart(document.getElementById('chart-payoff'), {
  yDomain: [0, 1], yTicks: [0, 0.5, 1], height: 128,
  series: [{ key: 'us', label: 'sender payoff', color: 'ink' }, { key: 'ur', label: 'receiver payoff', color: 'muted', dash: [5, 4] }],
});
const infoChart = new LineChart(document.getElementById('chart-info'), {
  yDomain: [0, 1], yTicks: [0, 0.5, 1], height: 128, valueFormat: (v) => `${v.toFixed(3)} bits`,
  series: [{ key: 'iA', label: 'I(state; action)', color: 'ink' }, { key: 'iM', label: 'I(state; message)', color: 'muted', dash: [5, 4] }],
});

function legend(id, items) {
  const ul = document.getElementById(id);
  ul.replaceChildren();
  for (const [label, dashed] of items) {
    const li = document.createElement('li');
    const key = document.createElement('span');
    key.className = 'key-line';
    key.style.background = dashed ? 'none' : 'currentColor';
    key.style.borderTop = dashed ? '2px dashed currentColor' : 'none';
    key.style.height = dashed ? '0' : '2.5px';
    key.style.color = dashed ? 'var(--muted)' : 'var(--ink)';
    const lab = document.createElement('span');
    lab.textContent = label;
    li.append(key, lab);
    ul.appendChild(li);
  }
}

// Particle flows: each square shows where its agent would go if the other stood still.
const flowS = new FlowField(plotS, {
  count: 260,
  sample: () => [Math.random(), Math.random()],
  inside: (x, y) => x >= 0 && x <= 1 && y >= 0 && y <= 1,
  field: () => [0, 0],
  maxAge: 70,
});
const flowR = new FlowField(plotR, {
  count: 260,
  sample: () => [Math.random(), Math.random()],
  inside: (x, y) => x >= 0 && x <= 1 && y >= 0 && y <= 1,
  field: () => [0, 0],
  maxAge: 70,
});

function binaryFields() {
  const g = state.game;
  const o = opts();
  const [r0, r1] = o.rates;
  const eps = o.exploration;
  // Sender advantages depend on the receiver's current strategy and vice versa.
  const [d1, d2, e1, e2] = g.binaryAdvantages(state.cur);
  flowS.setField((x, y) => [r0 * binaryField(o.rule, x, d1, o.temperature) + eps * (0.5 - x), r0 * binaryField(o.rule, y, d2, o.temperature) + eps * (0.5 - y)], false);
  flowR.setField((x, y) => [r1 * binaryField(o.rule, x, e1, o.temperature) + eps * (0.5 - x), r1 * binaryField(o.rule, y, e2, o.temperature) + eps * (0.5 - y)], false);
}

// A fixed speed reference per game and rule, so tracer speed shows signal strength.
function setFlowReference() {
  const g = state.game;
  const o = opts();
  let refS = 0;
  let refR = 0;
  for (const X of [[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 1, 1], [0.5, 0.5, 1, 0]]) {
    const [d1, d2, e1, e2] = g.binaryAdvantages(X);
    for (let i = 0; i < 12; i++) {
      for (let j = 0; j < 12; j++) {
        const x = (i + 0.5) / 12;
        const y = (j + 0.5) / 12;
        refS = Math.max(refS, Math.hypot(binaryField(o.rule, x, d1), binaryField(o.rule, y, d2)) * o.rates[0]);
        refR = Math.max(refR, Math.hypot(binaryField(o.rule, x, e1), binaryField(o.rule, y, e2)) * o.rates[1]);
      }
    }
  }
  flowS.ref = Math.max(refS * 0.6, 1e-6);
  flowR.ref = Math.max(refR * 0.6, 1e-6);
}

// ---------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------

function freshHistory() {
  return { t: [], pts: [], plane: [], us: [], ur: [], iM: [], iA: [] };
}

// Every k-th element so long histories stay cheap to draw (the last one is always kept).
function thin(arr, max = 1200) {
  if (arr.length <= max) return arr;
  const k = Math.ceil(arr.length / max);
  const out = [];
  for (let i = 0; i < arr.length; i += k) out.push(arr[i]);
  if (out[out.length - 1] !== arr[arr.length - 1]) out.push(arr[arr.length - 1]);
  return out;
}

function record() {
  const g = state.game;
  const h = state.hist;
  const { S, R } = matrices(state.cur);
  const [us, ur] = g.payoffs(S, R);
  const info = g.information(S, R);
  h.t.push(state.t);
  h.pts.push(Float64Array.from(state.cur));
  h.plane.push(binary() ? [state.cur[0] - state.cur[1], state.cur[2] - state.cur[3]] : [info.message / g.entropy(), info.action / g.entropy()]);
  h.us.push(us);
  h.ur.push(ur);
  h.iM.push(info.message);
  h.iA.push(info.action);
}

function matrices(y) {
  if (binary()) return sg.binaryToMatrices(y);
  const g = state.game;
  return { S: y.subarray(0, g.n * g.k), R: y.subarray(g.n * g.k) };
}

function stepState(y, dt) {
  if (binary()) sg.stepBinary(state.game, y, dt, opts());
  else sg.step(state.game, y, dt, { ...opts(), rows: state.rows });
}

function randomStart(rand) {
  if (binary()) return Float64Array.from([rand(), rand(), rand(), rand()]);
  return sg.randomStart(state.game, rand);
}

function applyScenario(key) {
  const s = SCENARIOS[key];
  state.family = s.family;
  state.ratio = s.ratio;
  state.rule = s.rule;
  ratio.set(s.ratio);
  ui.rule.value = s.rule;
  if (s.prior !== undefined) {
    state.prior = s.prior;
    prior.set(s.prior);
  }
  if (s.costLow !== undefined) {
    state.costLow = s.costLow;
    cost.set(s.costLow);
  }
  ui.priorControl.hidden = s.family !== 'lewis2';
  ui.costControl.hidden = s.family !== 'handicap';
  state.seed = 7;
  rebuild({ keepStart: false });
}

function defaultStart() {
  if (state.family === 'handicap') return Float64Array.from([0.55, 0.35, 0.5, 0.35]);
  if (state.family === 'lewis2') return Float64Array.from([0.6, 0.42, 0.55, 0.46]);
  return sg.randomStart(state.game, mulberry32(11));
}

function rebuild({ keepStart }) {
  const prevBinary = state.start && state.start.length === 4;
  state.game = makeGame();
  state.rows = binary() ? null : sg.rowLayout(state.game);
  if (!keepStart || !state.start || prevBinary !== binary()) state.start = defaultStart();
  const rand = mulberry32(state.seed * 7919);
  const n = binary() ? N_SWARM_BINARY : N_SWARM_GENERAL;
  state.swarmStart = Array.from({ length: n }, () => randomStart(rand));
  if (state.seed !== 7 && !keepStart) state.start = randomStart(mulberry32(state.seed * 104729));
  configureViews();
  restart();
  scheduleOutcomes();
  if (binary()) scheduleBasins();
}

function restart() {
  state.t = 0;
  state.steps = 0;
  state.cur = Float64Array.from(state.start);
  state.swarm = state.swarmStart.map((y) => Float64Array.from(y));
  state.hist = freshHistory();
  state.pausedAtEnd = false;
  record();
  flowS.clear();
  flowR.clear();
  if (binary()) {
    setFlowReference();
    binaryFields();
  }
  if (!state.playing) setPlaying(true);
  drawStatic();
}

function advance(dtModel) {
  // the 12-dimensional three-state flow is smooth enough for a coarser step
  const h = binary() ? H : 2 * H;
  const n = Math.min(40, Math.max(1, Math.round(dtModel / h)));
  for (let i = 0; i < n; i++) {
    if (state.t >= T_MAX) {
      state.pausedAtEnd = true;
      setPlaying(false);
      break;
    }
    stepState(state.cur, h);
    if (state.swarmOn || !binary()) for (const y of state.swarm) stepState(y, h);
    state.t += h;
    state.steps += 1;
    if (binary() ? state.steps % REC_EVERY === 0 : true) record();
  }
}

// ---------------------------------------------------------------------------
// Outcomes: every swarm run's destination (2x2x2), or live categories (3x3x3)
// ---------------------------------------------------------------------------

let outcomeTimer = null;
let outcomeJob = null;

function scheduleOutcomes() {
  state.outcomes = null;
  if (outcomeTimer) clearTimeout(outcomeTimer);
  if (outcomeJob) outcomeJob.cancelled = true;
  if (!binary()) {
    renderOutcomes();
    return;
  }
  renderOutcomes();
  outcomeTimer = setTimeout(() => {
    const job = { cancelled: false };
    outcomeJob = job;
    const starts = state.swarmStart;
    const codes = new Array(starts.length).fill(-1);
    const o = { ...opts(), tMax: horizon(), dt: 0.1 };
    let i = 0;
    const work = () => {
      if (job.cancelled) return;
      const t0 = performance.now();
      while (i < starts.length && performance.now() - t0 < 10) {
        codes[i] = sg.runBinary(state.game, starts[i], o).code;
        i += 1;
      }
      if (i < starts.length) requestAnimationFrame(work);
      else {
        state.outcomes = codes;
        renderOutcomes();
      }
    };
    requestAnimationFrame(work);
  }, 120);
}

function horizon() {
  return state.rule === 'softmax_pg' ? 1500 : 400;
}

function outcomeColors(T) {
  return [T.s1, T.s2, T.muted, T.s3];
}

// Live category of a three-state run: settled on 3 or 2 meanings, or still moving.
function generalCategory(y) {
  const g = state.game;
  const { S, R } = matrices(y);
  const k = sg.distinguishedStates(g, S, R);
  if (k > 2.95) return 0;
  if (Math.abs(k - 2) < 0.03) return 1;
  return 2;
}

const GENERAL_LABELS = ['signaling system (3 meanings)', 'partial pooling (2 meanings)', 'still learning'];

function generalColors(T) {
  return [T.ink, T.s7, T.muted];
}

function renderOutcomes() {
  const T = tokens();
  let labels;
  let colors;
  let counts;
  let total;
  if (binary()) {
    labels = sg.outcomeLabels(state.game);
    colors = outcomeColors(T);
    counts = [0, 0, 0, 0];
    total = state.swarmStart.length;
    if (state.outcomes) for (const c of state.outcomes) counts[c] += 1;
    ui.outcomeHint.textContent = state.outcomes ? `${total} random starts, run to t = ${horizon()}` : 'computing…';
  } else {
    labels = GENERAL_LABELS;
    colors = generalColors(T);
    counts = [0, 0, 0];
    for (const y of state.swarm) counts[generalCategory(y)] += 1;
    total = state.swarm.length;
    ui.outcomeHint.textContent = `${total} random starts, now at t = ${Math.round(state.t)}`;
  }
  const known = counts.reduce((a, b) => a + b, 0);
  ui.stackbar.replaceChildren();
  ui.stackrows.replaceChildren();
  counts.forEach((c, i) => {
    if (binary() && i === 3 && c === 0) return;
    const span = document.createElement('span');
    span.style.flexGrow = String(c);
    span.style.background = colors[i];
    span.title = `${labels[i]}: ${known ? pct(c / known) : '–'}`;
    ui.stackbar.appendChild(span);
    const row = document.createElement('div');
    const key = document.createElement('span');
    key.className = 'key-dot';
    key.style.background = colors[i];
    const lab = document.createElement('span');
    lab.textContent = labels[i];
    const val = document.createElement('span');
    val.className = 'val';
    val.textContent = known ? pct(c / known) : '…';
    row.append(key, lab, val);
    ui.stackrows.appendChild(row);
  });
  ui.stackbar.setAttribute('aria-label', labels.map((l, i) => `${l} ${known ? pct(counts[i] / known) : ''}`).join(', '));
}

// ---------------------------------------------------------------------------
// Basin slice (2x2x2): every sender start, with the receiver's start fixed
// ---------------------------------------------------------------------------

const BASIN_N = 56;
let basinAxis = 'sender';
let basinTimer = null;
let basinJob = null;
let basinCodes = null;

function scheduleBasins() {
  if (basinTimer) clearTimeout(basinTimer);
  if (basinJob) basinJob.cancelled = true;
  basinCodes = null;
  drawBasin();
  if (!binary()) return;
  basinTimer = setTimeout(() => {
    const job = { cancelled: false };
    basinJob = job;
    const codes = new Int8Array(BASIN_N * BASIN_N).fill(-1);
    basinCodes = codes;
    const fixedPart = basinAxis === 'sender' ? [state.start[2], state.start[3]] : [state.start[0], state.start[1]];
    const o = { ...opts(), tMax: state.rule === 'softmax_pg' ? 1200 : 300, dt: 0.2 };
    let idx = 0;
    const work = () => {
      if (job.cancelled) return;
      const t0 = performance.now();
      while (idx < codes.length && performance.now() - t0 < 9) {
        const u = ((idx % BASIN_N) + 0.5) / BASIN_N;
        const v = (Math.floor(idx / BASIN_N) + 0.5) / BASIN_N;
        const X0 = basinAxis === 'sender' ? [u, v, fixedPart[0], fixedPart[1]] : [fixedPart[0], fixedPart[1], u, v];
        codes[idx] = sg.runBinary(state.game, X0, o).code;
        idx += 1;
      }
      drawBasin();
      if (idx < codes.length) requestAnimationFrame(work);
    };
    requestAnimationFrame(work);
  }, 250);
}

function drawBasin() {
  const T = tokens();
  const p = plotB;
  p.clear('bg');
  p.clear('fg');
  const ctx = p.ctx.bg;
  if (!binary()) return;
  const cols = outcomeColors(T).map((c) => mixRgb(c, T.surface, 0.55));
  if (basinCodes) {
    raster(p, ctx, (x, y) => {
      const i = Math.min(BASIN_N - 1, Math.floor(x * BASIN_N));
      const j = Math.min(BASIN_N - 1, Math.floor(y * BASIN_N));
      const c = basinCodes[j * BASIN_N + i];
      if (c < 0) return null;
      return [...cols[c], 255];
    }, { res: BASIN_N, smooth: false });
  }
  frame(p, ctx, T);
  ticks(p, ctx, T, { x: [0, 0.5, 1], y: [0, 0.5, 1] });
  const g = state.game;
  const [m1, m2] = g.messageLabels;
  const [t1, t2] = g.stateLabels;
  const [a1] = g.actionLabels;
  const off = basinAxis === 'sender' ? 0 : 2;
  if (basinAxis === 'sender') axisLabels(p, ctx, T, { x: `sender start: P(${m1} | ${t1})`, y: `P(${m1} | ${t2})`, size: 11.5 });
  else axisLabels(p, ctx, T, { x: `receiver start: P(${a1} | ${m1})`, y: `P(${a1} | ${m2})`, size: 11.5 });
  const fg = p.ctx.fg;
  ring(fg, p.sx(state.start[off]), p.sy(state.start[off + 1]), 5, T.ink, 1.8, T.surface);
}

document.getElementById('basin-axis').addEventListener('change', (e) => {
  basinAxis = e.target.value;
  ui.basinHint.textContent = basinAxis === 'sender'
    ? 'Basin slice: every sender start, with the receiver starting where the ink run does'
    : 'Basin slice: every receiver start, with the sender starting where the ink run does';
  scheduleBasins();
});

// ---------------------------------------------------------------------------
// Views
// ---------------------------------------------------------------------------

function configureViews() {
  const g = state.game;
  if (binary()) {
    for (const p of [plotS, plotR]) {
      p.pad = squarePad;
      p.setDomain([0, 1], [0, 1]);
    }
    plotP.setDomain([-1, 1], [-1, 1]);
    flowS.enabled = true;
    flowR.enabled = true;
    ui.basinBlock.hidden = false;
    ui.planeTitle.textContent = 'Meaning plane';
    ui.planeHint.textContent = 'hover a dot to follow that run; click to adopt it';
    ui.senderHint.textContent = 'drag to set the start';
    ui.receiverHint.textContent = 'drag to set the start';
    const [t1, t2] = g.stateLabels;
    const [m1] = g.messageLabels;
    const [a1] = g.actionLabels;
    ui.senderCaption.textContent = `Each point is a sender strategy: x = P(${m1} | ${t1}), y = P(${m1} | ${t2}). Tracers show where the sender would move if the receiver stood still; shading is the sender's payoff given the receiver. Corner diagrams are the pure strategies.`;
    ui.receiverCaption.textContent = `Each point is a receiver strategy: x = P(${a1} | ${m1}), y = P(${a1} | ${g.messageLabels[1]}). The diagonal means "ignore the message". As the sender learns, this landscape tilts under the receiver.`;
    ui.planeCaption.textContent = state.family === 'handicap'
      ? 'Sender separation (how differently the two types display) against receiver separation (how differently displays are treated). Honest signaling is the top-right corner; the ring marks the hybrid equilibrium when it exists.'
      : 'Sender separation against receiver separation, for this run (ink) and 200 others. Conventions sit in two corners; the axes are where one side has stopped discriminating. With equal priors the common payoff is (1 + dx·dy)/2: a saddle.';
  } else {
    for (const p of [plotS, plotR]) {
      p.pad = { top: 26, right: 22, bottom: 30, left: 22 };
      p.setDomain([-0.06, 1.06], [-0.08, TRI_H + 0.06]);
    }
    plotP.setDomain([0, 1], [0, 1]);
    flowS.enabled = false;
    flowR.enabled = false;
    flowS.clear();
    flowR.clear();
    ui.basinBlock.hidden = true;
    ui.planeTitle.textContent = 'Information plane';
    ui.planeHint.textContent = 'each dot is one run';
    ui.senderHint.textContent = 'drag a hollow dot to move a start';
    ui.receiverHint.textContent = 'drag a hollow dot to move a start';
    ui.senderCaption.textContent = 'Each state is a dot in the triangle of messages: its position is the mix of messages sent in that state. A signaling system puts the three dots in three different corners.';
    ui.receiverCaption.textContent = 'Each message is a dot in the triangle of actions (corner aᵢ fits state tᵢ). An unused message gets no signal and freezes where it is.';
    ui.planeCaption.textContent = 'How much the messages carry about the state (x) against how much reaches the action (y), as shares of the 1.58 bits available. Nothing can sit above the diagonal: the receiver cannot learn more than the messages carry.';
  }
  ui.flowHint.textContent = g.name.includes('Costly') ? 'high and low types' : '';
  legend('time-legend-payoff', [['sender payoff', false], ['receiver payoff', true]]);
  legend('time-legend-info', [['I(state; action)', false], ['I(state; message)', true]]);
  const H2 = g.entropy();
  const top = Math.max(1, Math.ceil(H2 * 2) / 2);
  infoChart.setYDomain([0, top], top > 1 ? [0, 0.5, 1, 1.5] : [0, 0.5, 1]);
  const lo = Math.min(0, ...g.uS.flat(), ...g.uR.flat()) - Math.max(0, ...g.cost.flat());
  payoffChart.setYDomain([Math.min(0, lo), 1], lo < 0 ? [-0.5, 0, 0.5, 1] : [0, 0.5, 1]);
}

function drawStatic() {
  drawSquares(true);
  drawPlane(true);
  drawBasin();
}

// Payoff landscape of a binary agent: linear in its own two probabilities.
function shadeLinear(p, ctx, d1, d2, T, strength) {
  const len = Math.hypot(d1, d2);
  if (len < 1e-12) return;
  const cx = (p.x0 + p.x1) / 2;
  const cy = (p.y0 + p.y1) / 2;
  const half = (p.x1 - p.x0) / 2;
  const ux = d1 / len;
  const uy = -d2 / len;
  const reach = half * (Math.abs(ux) + Math.abs(uy));
  const grad = ctx.createLinearGradient(cx - ux * reach, cy - uy * reach, cx + ux * reach, cy + uy * reach);
  grad.addColorStop(0, alpha(T.ink, 0));
  grad.addColorStop(1, alpha(T.ink, (T.dark ? 0.16 : 0.1) * strength));
  ctx.fillStyle = grad;
  ctx.fillRect(p.x0, p.y0, p.x1 - p.x0, p.y1 - p.y0);
}

function drawSquares() {
  const T = tokens();
  if (binary()) drawBinarySquares(T);
  else drawTriangles(T);
}

function swarmColor(i, T) {
  if (!binary()) return generalColors(T)[generalCategory(state.swarm[i])];
  if (!state.outcomes) return T.muted;
  return outcomeColors(T)[state.outcomes[i]];
}

function drawBinarySquares(T) {
  const g = state.game;
  const X = state.cur;
  const [d1, d2, f1, f2] = g.binaryAdvantages(X);
  const [t1, t2] = g.stateLabels;
  const [m1, m2] = g.messageLabels;
  const [a1] = g.actionLabels;
  const specs = [
    [plotS, 'sender', d1, d2, 0, `P(${m1} | ${t1})`, `P(${m1} | ${t2})`],
    [plotR, 'receiver', f1, f2, 2, `P(${a1} | ${m1})`, `P(${a1} | ${m2})`],
  ];
  const maxD = Math.max(...g.prior) * 1.2;
  for (const [p, role, e1, e2, off, xl, yl] of specs) {
    const bg = p.ctx.bg;
    p.clear('bg');
    shadeLinear(p, bg, e1, e2, T, Math.min(1, (Math.abs(e1) + Math.abs(e2)) / maxD));
    // diagonal = uninformative sender / message-blind receiver
    bg.strokeStyle = T.grid;
    bg.lineWidth = 1;
    bg.beginPath();
    bg.moveTo(p.sx(0), p.sy(0));
    bg.lineTo(p.sx(1), p.sy(1));
    bg.stroke();
    frame(p, bg, T);
    ticks(p, bg, T, { x: [0, 0.5, 1], y: [0, 0.5, 1] });
    axisLabels(p, bg, T, { x: xl, y: yl });
    cornerGlyphs(p, bg, T, role);

    const fg = p.ctx.fg;
    p.clear('fg');
    if (state.swarmOn) {
      for (let i = 0; i < state.swarm.length; i++) {
        const y = state.swarm[i];
        dot(fg, p.sx(y[off]), p.sy(y[off + 1]), 2.3, alpha(swarmColor(i, T), 0.85));
      }
    }
    const pts = thin(state.hist.pts).map((y) => [p.sx(y[off]), p.sy(y[off + 1])]);
    pts.push([p.sx(X[off]), p.sy(X[off + 1])]);
    polyline(fg, pts, { color: T.ink, width: 2, halo: alpha(T.surface, 0.9) });
    ring(fg, p.sx(state.start[off]), p.sy(state.start[off + 1]), 5, T.ink, 1.6, T.surface);
    dot(fg, p.sx(X[off]), p.sy(X[off + 1]), 5, T.ink, T.surface, 2);
    if (state.hover >= 0 && state.swarmOn) {
      const y = state.swarm[state.hover];
      ring(fg, p.sx(y[off]), p.sy(y[off + 1]), 7, T.ink, 1.6);
    }
  }
}

function drawTriangles(T) {
  const g = state.game;
  const k = g.k;
  const l = g.l;
  const n = g.n;
  const hist = thin(state.hist.pts, 600);
  const msgInk = [T.ink, T.ink2, T.muted];
  const specs = [
    [plotS, g.messageLabels, (y, r) => Array.from(y.subarray(r * k, r * k + k)), n, (r) => T.series[r], g.stateLabels],
    [plotR, g.actionLabels, (y, r) => Array.from(y.subarray(n * k + r * l, n * k + r * l + l)), k, (r) => msgInk[r % 3], g.messageLabels],
  ];
  for (const [p, corners, rowOf, rows, color, rowLabels] of specs) {
    const bg = p.ctx.bg;
    p.clear('bg');
    trianglePath(p, bg);
    bg.strokeStyle = T.axis;
    bg.lineWidth = 1;
    bg.stroke();
    const vx = [[0, 0], [1, 0], [0.5, TRI_H]];
    corners.forEach((lab, i) => {
      const [x, y] = vx[i];
      const px = p.sx(x);
      const py = p.sy(y);
      const dy = i === 2 ? -10 : 18;
      text(bg, lab, px, py + dy, { color: T.ink2, size: 12, align: 'center', baseline: 'middle' });
      if (p === plotR) dot(bg, px, py, 3.5, T.series[i]);
    });
    const fg = p.ctx.fg;
    p.clear('fg');
    for (let r = 0; r < rows; r++) {
      const pts = hist.map((y) => {
        const [X, Y] = baryToXY(rowOf(y, r));
        return [p.sx(X), p.sy(Y)];
      });
      const [cx, cy] = baryToXY(rowOf(state.cur, r));
      pts.push([p.sx(cx), p.sy(cy)]);
      polyline(fg, pts, { color: color(r), width: 1.8, halo: alpha(T.surface, 0.9) });
      const [sx0, sy0] = baryToXY(rowOf(state.start, r));
      ring(fg, p.sx(sx0), p.sy(sy0), 5, color(r), 1.8, T.surface);
      dot(fg, p.sx(cx), p.sy(cy), 5, color(r), T.surface, 2);
      text(fg, rowLabels[r], p.sx(sx0) + 8, p.sy(sy0) - 8, { color: T.ink2, size: 11, halo: T.surface });
    }
  }
}

function planeCoords(y) {
  if (binary()) return [y[0] - y[1], y[2] - y[3]];
  const g = state.game;
  const { S, R } = matrices(y);
  const info = g.information(S, R);
  const Hn = g.entropy();
  return [info.message / Hn, info.action / Hn];
}

function drawPlane() {
  const T = tokens();
  const p = plotP;
  const bg = p.ctx.bg;
  p.clear('bg');
  const g = state.game;
  if (binary()) {
    gridLines(p, bg, T, { x: [0], y: [0] });
    if (state.family === 'lewis2' && Math.abs(state.prior - 0.5) < 1e-9) {
      // contours of the common payoff (1 + dx dy) / 2
      bg.strokeStyle = T.grid;
      bg.lineWidth = 1;
      for (const c of [0.1, 0.25, 0.45, 0.7]) {
        for (const sgn of [1, -1]) {
          for (const side of [1, -1]) {
            bg.beginPath();
            let first = true;
            for (let i = 0; i <= 80; i++) {
              const x = side * (c + ((1 - c) * i) / 80);
              const y = (sgn * c) / x;
              if (Math.abs(y) > 1) continue;
              if (first) bg.moveTo(p.sx(x), p.sy(y));
              else bg.lineTo(p.sx(x), p.sy(y));
              first = false;
            }
            bg.stroke();
          }
        }
      }
    }
    frame(p, bg, T);
    ticks(p, bg, T, { x: [-1, 0, 1], y: [-1, 0, 1] });
    axisLabels(p, bg, T, { x: 'sender separation', y: 'receiver separation' });
    const labels = sg.outcomeLabels(g);
    text(bg, labels[0], p.sx(1) - 6, p.sy(1) + 14, { color: T.ink2, size: 11.5, align: 'right', halo: T.surface });
    text(bg, labels[1], p.sx(-1) + 6, p.sy(-1) - 8, { color: T.ink2, size: 11.5, align: 'left', halo: T.surface });
    const hyb = sg.hybridEquilibrium(g);
    if (hyb) {
      marker(bg, p.sx(hyb[0] - hyb[1]), p.sy(hyb[2] - hyb[3]), 'center', T);
      text(bg, 'hybrid equilibrium', p.sx(hyb[0] - hyb[1]) - 10, p.sy(hyb[2] - hyb[3]) + 18, { color: T.ink2, size: 11, align: 'right', halo: T.surface });
    }
  } else {
    // forbidden region above the diagonal
    bg.fillStyle = alpha(T.ink, T.dark ? 0.07 : 0.045);
    bg.beginPath();
    bg.moveTo(p.sx(0), p.sy(0));
    bg.lineTo(p.sx(1), p.sy(1));
    bg.lineTo(p.sx(0), p.sy(1));
    bg.closePath();
    bg.fill();
    gridLines(p, bg, T, { x: [0.5], y: [0.5] });
    bg.strokeStyle = T.axis;
    bg.beginPath();
    bg.moveTo(p.sx(0), p.sy(0));
    bg.lineTo(p.sx(1), p.sy(1));
    bg.stroke();
    frame(p, bg, T);
    ticks(p, bg, T, { x: [0, 0.5, 1], y: [0, 0.5, 1] });
    axisLabels(p, bg, T, { x: 'I(state; message) / H(state)', y: 'I(state; action) / H(state)' });
    text(bg, 'out of reach', p.sx(0.05), p.sy(0.9), { color: T.muted, size: 11.5 });
    text(bg, 'signaling system', p.sx(1) - 10, p.sy(1) + 16, { color: T.ink2, size: 11.5, align: 'right', halo: T.surface });
  }
  const fg = p.ctx.fg;
  p.clear('fg');
  if (state.swarmOn || !binary()) {
    for (let i = 0; i < state.swarm.length; i++) {
      const [x, y] = planeCoords(state.swarm[i]);
      dot(fg, p.sx(x), p.sy(y), 2.4, alpha(swarmColor(i, T), 0.85));
    }
  }
  const pts = thin(state.hist.plane).map(([x, yy]) => [p.sx(x), p.sy(yy)]);
  const [cx, cy] = planeCoords(state.cur);
  pts.push([p.sx(cx), p.sy(cy)]);
  polyline(fg, pts, { color: T.ink, width: 2, halo: alpha(T.surface, 0.9) });
  dot(fg, p.sx(cx), p.sy(cy), 5, T.ink, T.surface, 2);
  if (state.hover >= 0) {
    const [hx, hy] = planeCoords(state.swarm[state.hover]);
    ring(fg, p.sx(hx), p.sy(hy), 7, T.ink, 1.6);
  }
}

function drawFlow() {
  const T = tokens();
  const p = plotF;
  p.clear('fg');
  const { S, R } = matrices(state.cur);
  signalFlow(p.ctx.fg, { x: 4, y: 4, w: p.w - 8, h: p.h - 8 }, state.game, S, R, T);
}

function drawCharts() {
  const h = state.hist;
  payoffChart.setData(h.t, { us: h.us, ur: h.ur });
  payoffChart.draw();
  infoChart.setData(h.t, { iA: h.iA, iM: h.iM });
  infoChart.draw();
}

function drawReadout() {
  const g = state.game;
  const { S, R } = matrices(state.cur);
  const [us, ur] = g.payoffs(S, R);
  const info = g.information(S, R);
  const parts = [`<b>t = ${state.t.toFixed(1)}</b>`];
  if (binary()) {
    const X = state.cur;
    parts.push(`sender (${fixed(X[0])}, ${fixed(X[1])})`, `receiver (${fixed(X[2])}, ${fixed(X[3])})`);
  }
  parts.push(`payoffs ${fixed(us, 3)} / ${fixed(ur, 3)}`, `I(state; action) ${fixed(info.action, 3)} bits`);
  if (state.pausedAtEnd) parts.push(`paused at t = ${T_MAX}`);
  ui.readout.innerHTML = parts.map((s) => `<span>${s}</span>`).join('');
}

// ---------------------------------------------------------------------------
// Interaction
// ---------------------------------------------------------------------------

function nearestSwarm(p, px, py, coords) {
  let best = -1;
  let bestD = 9;
  for (let i = 0; i < state.swarm.length; i++) {
    const [x, y] = coords(state.swarm[i]);
    const d = Math.hypot(p.sx(x) - px, p.sy(y) - py);
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }
  return best;
}

function adopt(i) {
  state.start = Float64Array.from(state.swarmStart[i]);
  state.hover = -1;
  restart();
  scheduleBasins();
}

const clamp01 = (v) => Math.min(1, Math.max(0, v));

for (const [p, off] of [[plotS, 0], [plotR, 2]]) {
  let dragRow = -1;
  p.on('down', (e) => {
    if (binary()) {
      if (state.swarmOn) {
        const i = nearestSwarm(p, e.px, e.py, (y) => [y[off], y[off + 1]]);
        if (i >= 0 && Math.hypot(e.px - p.sx(state.cur[off]), e.py - p.sy(state.cur[off + 1])) > 8) {
          adopt(i);
          return;
        }
      }
      state.start[off] = clamp01(e.x);
      state.start[off + 1] = clamp01(e.y);
      restart();
      dragRow = 0;
    } else {
      const g = state.game;
      const rows = off === 0 ? g.n : g.k;
      let best = -1;
      let bestD = 22;
      for (let r = 0; r < rows; r++) {
        const row = rowOfStart(off, r);
        const [x, y] = baryToXY(row);
        const d = Math.hypot(p.sx(x) - e.px, p.sy(y) - e.py);
        if (d < bestD) {
          bestD = d;
          best = r;
        }
      }
      dragRow = best;
    }
  });
  p.on('move', (e) => {
    if (e.dragging && dragRow >= 0) {
      if (binary()) {
        state.start[off] = clamp01(e.x);
        state.start[off + 1] = clamp01(e.y);
        restart();
      } else {
        setStartRow(off, dragRow, e.x, e.y);
        restart();
      }
      return;
    }
    if (binary() && state.swarmOn) {
      const i = nearestSwarm(p, e.px, e.py, (y) => [y[off], y[off + 1]]);
      state.hover = i;
    }
  });
  p.on('up', () => {
    if (dragRow >= 0 && binary()) scheduleBasins();
    dragRow = -1;
  });
  p.on('leave', () => { state.hover = -1; });
}

function rowOfStart(off, r) {
  const g = state.game;
  if (off === 0) return Array.from(state.start.subarray(r * g.k, r * g.k + g.k));
  return Array.from(state.start.subarray(g.n * g.k + r * g.l, g.n * g.k + r * g.l + g.l));
}

function setStartRow(off, r, X, Y) {
  const g = state.game;
  let b = xyToBary(X, Y).map((v) => Math.max(v, 0.002));
  const s = b.reduce((a, c) => a + c, 0);
  b = b.map((v) => v / s);
  const o = off === 0 ? r * g.k : g.n * g.k + r * g.l;
  for (let i = 0; i < 3; i++) state.start[o + i] = b[i];
}

plotP.on('move', (e) => {
  if (!state.swarmOn && binary()) return;
  state.hover = binary() ? nearestSwarm(plotP, e.px, e.py, planeCoords) : -1;
});
plotP.on('down', (e) => {
  if (!binary()) return;
  const i = nearestSwarm(plotP, e.px, e.py, planeCoords);
  if (i >= 0) adopt(i);
});
plotP.on('leave', () => { state.hover = -1; });

for (const b of document.querySelectorAll('[data-try]')) {
  b.addEventListener('click', () => {
    const k = b.getAttribute('data-try');
    if (k === 'saddle') {
      ui.scenario.value = 'lewis_even';
      applyScenario('lewis_even');
      state.start = Float64Array.from([0.52, 0.49, 0.515, 0.5]);
      restart();
      scheduleBasins();
    } else if (k === 'trap') {
      ui.scenario.value = 'impatient';
      applyScenario('impatient');
    } else if (k === 'patient') {
      ui.scenario.value = 'lewis_skewed';
      applyScenario('lewis_skewed');
      state.ratio = 3;
      ratio.set(3);
      rebuild({ keepStart: true });
    } else if (k === 'cycle') {
      ui.scenario.value = 'handicap_cheap';
      applyScenario('handicap_cheap');
      state.rule = 'projection';
      ui.rule.value = 'projection';
      rebuild({ keepStart: true });
    } else if (k === 'three') {
      ui.scenario.value = 'lewis3';
      applyScenario('lewis3');
    }
    document.querySelector('.controls').scrollIntoView({ behavior: 'smooth', block: 'start' });
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
  if (state.playing) advance(dt * TIME_RATE * 2 ** state.speed);
  if (binary()) {
    binaryFields();
    flowS.frame();
    flowR.frame();
  }
  drawSquares();
  drawPlane();
  drawFlow();
  if (frameNo % 4 === 0 || !state.playing) {
    drawCharts();
    drawReadout();
    if (!binary() && frameNo % 12 === 0) renderOutcomes();
  }
  requestAnimationFrame(loop);
}

onThemeChange(() => {
  renderOutcomes();
  drawStatic();
  drawCharts();
  flowS.clear();
  flowR.clear();
});

applyScenario('lewis_even');
requestAnimationFrame(loop);
