// Particle flow: tracers advected by a vector field leave fading trails,
// so a phase portrait reads as moving water instead of a grid of arrows.

import { alpha as withAlpha, tokens } from './theme.js';
import { arrowHead } from './plot.js';

const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

export class FlowField {
  constructor(plot, { layer = 'fx', count = 320, field, sample, inside, maxAge = 90, width = 1.1, stepFrac = 0.0045, colorKey = 'muted', opacity = 0.75 } = {}) {
    Object.assign(this, { plot, layer, count, field, sample, inside, maxAge, width, stepFrac, colorKey, opacity });
    this.particles = [];
    this.ref = 1;
    this.enabled = true;
    this.reset();
    plot.onResize(() => this.clear());
  }

  setField(field, renormalize = true) {
    this.field = field;
    if (renormalize) this.normalize();
  }

  normalize() {
    const speeds = [];
    const { xd, yd } = this.plot;
    for (let i = 0; i < 18; i++) {
      for (let j = 0; j < 18; j++) {
        const x = xd[0] + ((i + 0.5) / 18) * (xd[1] - xd[0]);
        const y = yd[0] + ((j + 0.5) / 18) * (yd[1] - yd[0]);
        if (this.inside && !this.inside(x, y)) continue;
        const [vx, vy] = this.field(x, y);
        const s = Math.hypot(vx, vy);
        if (Number.isFinite(s)) speeds.push(s);
      }
    }
    speeds.sort((a, b) => a - b);
    const q = speeds[Math.floor(speeds.length * 0.9)] || 0;
    this.ref = q > 1e-12 ? q : 1;
  }

  reset() {
    this.particles = Array.from({ length: this.count }, () => this._spawn(true));
    this.normalize();
  }

  _spawn(randomAge = false) {
    const [x, y] = this.sample();
    return { x, y, age: randomAge ? Math.floor(Math.random() * this.maxAge) : 0 };
  }

  clear() {
    this.plot.clear(this.layer);
  }

  frame() {
    const plot = this.plot;
    const ctx = plot.ctx[this.layer];
    if (!this.enabled) return;
    if (reduceMotion.matches) {
      this.drawStatic();
      return;
    }
    ctx.save();
    ctx.globalCompositeOperation = 'destination-out';
    ctx.fillStyle = 'rgba(0,0,0,0.085)';
    ctx.fillRect(0, 0, plot.w, plot.h);
    ctx.restore();
    const T = tokens();
    ctx.strokeStyle = withAlpha(T[this.colorKey] || T.muted, this.opacity);
    ctx.lineWidth = this.width;
    ctx.lineCap = 'round';
    const span = Math.max(plot.xd[1] - plot.xd[0], plot.yd[1] - plot.yd[0]);
    const k = (this.stepFrac * span) / this.ref;
    ctx.beginPath();
    for (let i = 0; i < this.particles.length; i++) {
      const p = this.particles[i];
      const [vx, vy] = this.field(p.x, p.y);
      const nx = p.x + vx * k;
      const ny = p.y + vy * k;
      p.age += 1;
      const still = Math.hypot(vx, vy) * k < span * 1e-5;
      if (p.age > this.maxAge || !Number.isFinite(nx) || (this.inside && !this.inside(nx, ny)) || (still && p.age > 12)) {
        this.particles[i] = this._spawn();
        continue;
      }
      ctx.moveTo(plot.sx(p.x), plot.sy(p.y));
      ctx.lineTo(plot.sx(nx), plot.sy(ny));
      p.x = nx;
      p.y = ny;
    }
    ctx.stroke();
  }

  // Reduced motion: a still field of short arrows instead of moving tracers.
  drawStatic(n = 15) {
    const plot = this.plot;
    const ctx = plot.ctx[this.layer];
    plot.clear(this.layer);
    const T = tokens();
    const color = withAlpha(T[this.colorKey] || T.muted, 0.9);
    const cell = (plot.x1 - plot.x0) / n;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const x = plot.xd[0] + ((i + 0.5) / n) * (plot.xd[1] - plot.xd[0]);
        const y = plot.yd[0] + ((j + 0.5) / n) * (plot.yd[1] - plot.yd[0]);
        if (this.inside && !this.inside(x, y)) continue;
        const [vx, vy] = this.field(x, y);
        const s = Math.hypot(vx, vy);
        if (!(s > 0)) continue;
        const len = cell * 0.42 * Math.min(1, Math.sqrt(s / this.ref));
        const ux = (vx / s) * len;
        const uy = (-vy / s) * len * ((plot.y1 - plot.y0) / (plot.yd[1] - plot.yd[0])) / ((plot.x1 - plot.x0) / (plot.xd[1] - plot.xd[0]));
        const px = plot.sx(x);
        const py = plot.sy(y);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(px - ux / 2, py - uy / 2);
        ctx.lineTo(px + ux / 2, py + uy / 2);
        ctx.stroke();
        arrowHead(ctx, px - ux / 2, py - uy / 2, px + ux / 2, py + uy / 2, 4, color);
      }
    }
  }
}

export function prefersReducedMotion() {
  return reduceMotion.matches;
}
