// Landing page: a live swarm of signaling-game runs, plus the static gallery.

import { tokens, onThemeChange, alpha } from '../ui/theme.js';
import { Plot, frame, ticks, axisLabels, dot, gridLines, text } from '../ui/plot.js';
import { initPage } from '../ui/common.js';
import * as sg from '../lib/signaling.js';
import { mulberry32 } from '../lib/rng.js';

initPage();

const FIGURES = [
  ['signaling_squares', 'A learning run is two paths at once', 'Sender and receiver strategies in the Lewis game move together; every run ends in one of the two conventions.'],
  ['meaning_plane', 'Convention formation is escape from a saddle', 'Skewed priors open a trap where the receiver stops listening before the sender has started to say anything.'],
  ['flow_filmstrip', 'Meaning, forming', 'Three states, three messages: band widths are probability, colours follow the state. One run finds a signaling system, one gets stuck in partial pooling.'],
  ['pooling_trap', 'When does language fail to form?', 'The share of runs that end without information grows with prior skew, and explodes when the receiver learns faster than the sender.'],
  ['basin_slices', 'Basins, one slice at a time', 'Every sender starting point, coloured by where learning ends up; the grey band of pooling widens as the prior tilts.'],
  ['costly_cycles', 'Honesty and trust chase each other', 'In the handicap game cheap faking produces closed orbits around a hybrid equilibrium; expensive faking makes honesty stable.'],
  ['lewis3_triangles', 'Each agent is a dot in its own triangle', 'Three-state Lewis game: the sender\'s states and the receiver\'s messages each move in a simplex of their own.'],
  ['normal_form_gallery', 'Six 2×2 games as flows', 'Replicator dynamics with Nash equilibria and basins of attraction.'],
  ['discrete_vs_continuous', 'Same signal, different clocks', 'Matching Pennies: continuous orbits close, discrete steps spiral out, optimism spirals in, smoothing settles.'],
  ['game_atlas', 'An atlas of symmetric 2×2 games', 'Harmony, Snowdrift, Stag Hunt and Prisoner\'s Dilemma as regions of one map.'],
  ['simplex_gallery', 'Flows on the triangle', 'Three kinds of rock-paper-scissors, coordination, Hawk-Dove-Bourgeois and the repeated Prisoner\'s Dilemma.'],
  ['bertrand_plane', 'Marginal profit leads to Nash, not to the cartel', 'Gradient play in a logit Bertrand duopoly, and how sympathy between firms lifts prices.'],
  ['edgeworth_cycles', 'Edgeworth cycles', 'Capacity limits turn the Bertrand price war into an endless sawtooth.'],
];

function buildGallery() {
  const g = document.getElementById('gallery');
  for (const [file, title, text_] of FIGURES) {
    const fig = document.createElement('figure');
    const pic = document.createElement('picture');
    const src = document.createElement('source');
    src.srcset = `figures/${file}_dark.png`;
    src.media = '(prefers-color-scheme: dark)';
    const img = document.createElement('img');
    img.src = `figures/${file}_light.png`;
    img.alt = title;
    img.loading = 'lazy';
    pic.append(src, img);
    const cap = document.createElement('figcaption');
    const b = document.createElement('b');
    b.textContent = `${title}. `;
    cap.append(b, document.createTextNode(text_));
    fig.append(pic, cap);
    g.appendChild(fig);
  }
  syncPictures();
}

// <picture> follows the OS setting; make it follow the page's theme toggle too.
function syncPictures() {
  const dark = tokens().dark;
  for (const pic of document.querySelectorAll('picture')) {
    const src = pic.querySelector('source');
    src.media = dark ? 'all' : 'not all';
  }
}

// ---------------------------------------------------------------------------
// Hero swarm
// ---------------------------------------------------------------------------

const game = sg.lewis(2, 0.75);
const opts = { rule: 'replicator', rates: [1, 1], temperature: 0.05, exploration: 0 };
const N = 200;
const plot = new Plot(document.getElementById('hero-plot'), { xDomain: [-1, 1], yDomain: [-1, 1], equal: true, aspect: 0.82, pad: { top: 12, right: 16, bottom: 44, left: 52 }, layers: ['bg', 'fg'], maxHeight: 460 });
plot.el.classList.add('static');
const clock = document.getElementById('hero-clock');
let swarm = [];
let codes = [];
let t = 0;
let seed = 1;

function reset() {
  const rand = mulberry32(seed++ * 7717);
  const starts = Array.from({ length: N }, () => [rand(), rand(), rand(), rand()]);
  codes = starts.map((x) => sg.runBinary(game, x, { ...opts, tMax: 300, dt: 0.1 }).code);
  swarm = starts.map((x) => Float64Array.from(x));
  t = 0;
  // start a little way in, so the first still frame already shows structure
  advance(4);
}

function advance(dtModel) {
  const h = 0.05;
  const n = Math.round(dtModel / h);
  for (let i = 0; i < n; i++) for (const y of swarm) sg.stepBinary(game, y, h, opts);
  t += n * h;
}

function drawBg() {
  const T = tokens();
  const ctx = plot.ctx.bg;
  plot.clear('bg');
  gridLines(plot, ctx, T, { x: [0], y: [0] });
  frame(plot, ctx, T);
  ticks(plot, ctx, T, { x: [-1, 0, 1], y: [-1, 0, 1] });
  axisLabels(plot, ctx, T, { x: 'sender separation', y: 'receiver separation' });
  text(ctx, 'm1 means t1', plot.sx(1) - 12, plot.sy(1) + 16, { color: T.ink2, size: 11.5, align: 'right', halo: T.surface });
  text(ctx, 'm1 means t2', plot.sx(-1) + 12, plot.sy(-1) - 10, { color: T.ink2, size: 11.5, halo: T.surface });
}

function draw() {
  const T = tokens();
  const ctx = plot.ctx.fg;
  plot.clear('fg');
  const colors = [T.s1, T.s2, T.muted, T.s3];
  for (let i = 0; i < swarm.length; i++) {
    const y = swarm[i];
    dot(ctx, plot.sx(y[0] - y[1]), plot.sy(y[2] - y[3]), 2.6, alpha(colors[codes[i]], 0.9));
  }
  clock.textContent = `t = ${t.toFixed(0)}`;
}

const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
let last = performance.now();
function loop(now) {
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  if (!reduce) {
    advance(dt * 5);
    if (t > 70) reset();
  }
  draw();
  requestAnimationFrame(loop);
}

plot.onResize(() => drawBg());
onThemeChange(() => {
  drawBg();
  syncPictures();
});

buildGallery();
reset();
if (reduce) advance(60);
drawBg();
requestAnimationFrame(loop);
