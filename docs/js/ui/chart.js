// Small line chart with a crosshair that snaps to the nearest sample and a
// tooltip listing every series at that x.

import { Plot, text, gridLines, polyline, dot } from './plot.js';
import { tokens } from './theme.js';

export class LineChart {
  constructor(container, { yDomain = [0, 1], yTicks = [0, 0.5, 1], yFormat = (v) => `${v}`, xLabel = 'time', series = [], height = 150, valueFormat = (v) => v.toFixed(3) } = {}) {
    this.container = container;
    this.yTicks = yTicks;
    this.yFormat = yFormat;
    this.xLabel = xLabel;
    this.series = series;
    this.valueFormat = valueFormat;
    this.xs = [];
    this.data = {};
    this.hover = null;
    const host = document.createElement('div');
    container.appendChild(host);
    this.plot = new Plot(host, { xDomain: [0, 1], yDomain, aspect: 0, minHeight: height, maxHeight: height, pad: { top: 8, right: 12, bottom: 26, left: 40 }, layers: ['bg', 'fg'] });
    host.classList.add('static');
    this.tip = document.createElement('div');
    this.tip.className = 'tip';
    this.tip.hidden = true;
    host.appendChild(this.tip);
    this.plot.onResize(() => this.draw());
    this.plot.on('move', (e) => {
      this.hover = e.inside ? e.x : null;
      this.drawHover();
    });
    this.plot.on('leave', () => {
      this.hover = null;
      this.drawHover();
    });
  }

  setData(xs, data) {
    this.xs = xs;
    this.data = data;
    const x1 = xs.length ? Math.max(xs[xs.length - 1], 1e-9) : 1;
    this.plot.xd = [0, x1];
    this.plot._layout();
  }

  setYDomain(yd, ticks) {
    this.plot.yd = yd;
    if (ticks) this.yTicks = ticks;
    this.plot._layout();
  }

  draw() {
    const T = tokens();
    const p = this.plot;
    const ctx = p.ctx.bg;
    p.clear('bg');
    gridLines(p, ctx, T, { y: this.yTicks });
    ctx.strokeStyle = T.axis;
    ctx.beginPath();
    ctx.moveTo(p.x0, Math.round(p.y1) + 0.5);
    ctx.lineTo(p.x1, Math.round(p.y1) + 0.5);
    ctx.stroke();
    for (const v of this.yTicks) text(ctx, this.yFormat(v), p.x0 - 6, p.sy(v), { color: T.muted, size: 11, align: 'right', baseline: 'middle' });
    const x1 = p.xd[1];
    const step = niceStep(x1 / 4);
    for (let v = 0; v <= x1 + 1e-9; v += step) text(ctx, fmtTime(v), p.sx(v), p.y1 + 15, { color: T.muted, size: 11, align: 'center' });
    for (const s of this.series) {
      const ys = this.data[s.key];
      if (!ys || ys.length < 2) continue;
      const pts = [];
      const stride = Math.max(1, Math.floor(ys.length / (p.x1 - p.x0)));
      for (let i = 0; i < ys.length; i += stride) pts.push([p.sx(this.xs[i]), p.sy(clampY(ys[i], p.yd))]);
      pts.push([p.sx(this.xs[ys.length - 1]), p.sy(clampY(ys[ys.length - 1], p.yd))]);
      polyline(ctx, pts, { color: T[s.color] || s.color, width: s.width || 2, dash: s.dash || null });
      const last = pts[pts.length - 1];
      dot(ctx, last[0], last[1], 3, T[s.color] || s.color, T.surface, 1.5);
    }
    this.drawHover();
  }

  drawHover() {
    const T = tokens();
    const p = this.plot;
    const ctx = p.ctx.fg;
    p.clear('fg');
    if (this.hover === null || this.xs.length < 2) {
      this.tip.hidden = true;
      return;
    }
    let i = bisect(this.xs, this.hover);
    i = Math.max(0, Math.min(this.xs.length - 1, i));
    const x = p.sx(this.xs[i]);
    ctx.strokeStyle = T.axis;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(Math.round(x) + 0.5, p.y0);
    ctx.lineTo(Math.round(x) + 0.5, p.y1);
    ctx.stroke();
    this.tip.replaceChildren();
    const head = document.createElement('div');
    head.textContent = `t = ${fmtTime(this.xs[i])}`;
    this.tip.appendChild(head);
    for (const s of this.series) {
      const ys = this.data[s.key];
      if (!ys || i >= ys.length) continue;
      const color = T[s.color] || s.color;
      dot(ctx, x, p.sy(clampY(ys[i], p.yd)), 3.5, color, T.surface, 1.5);
      const row = document.createElement('div');
      row.className = 'row';
      const key = document.createElement('span');
      key.className = 'key-line';
      key.style.color = color;
      const val = document.createElement('b');
      val.textContent = this.valueFormat(ys[i]);
      const lab = document.createElement('span');
      lab.textContent = s.label;
      row.append(key, val, lab);
      this.tip.appendChild(row);
    }
    this.tip.hidden = false;
    const tw = this.tip.offsetWidth;
    const left = x + 12 + tw > p.w ? x - 12 - tw : x + 12;
    this.tip.style.left = `${left}px`;
    this.tip.style.top = `${p.y0 + 4}px`;
  }
}

function clampY(v, yd) {
  return Math.max(yd[0], Math.min(yd[1], v));
}

function bisect(xs, x) {
  let lo = 0;
  let hi = xs.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] < x) lo = mid;
    else hi = mid;
  }
  return Math.abs(xs[lo] - x) < Math.abs(xs[hi] - x) ? lo : hi;
}

export function niceStep(raw) {
  if (!(raw > 0)) return 1;
  const p = 10 ** Math.floor(Math.log10(raw));
  const m = raw / p;
  return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
}

function fmtTime(v) {
  return Math.abs(v) >= 100 || Number.isInteger(v) ? `${Math.round(v)}` : v.toFixed(1);
}
