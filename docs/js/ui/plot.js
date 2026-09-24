// HiDPI canvas plots with data-coordinate scales, stacked layers and pointer events.

export class Plot {
  constructor(container, { xDomain = [0, 1], yDomain = [0, 1], aspect = 1, pad = {}, layers = ['bg', 'fx', 'fg'], equal = false, minHeight = 160, maxHeight = 640 } = {}) {
    this.el = container;
    this.el.classList.add('plot');
    this.xd = xDomain;
    this.yd = yDomain;
    this.aspect = aspect;
    this.pad = { top: 14, right: 14, bottom: 34, left: 40, ...pad };
    this.equal = equal;
    this.minHeight = minHeight;
    this.maxHeight = maxHeight;
    this.canvases = {};
    this.ctx = {};
    for (const name of layers) {
      const c = document.createElement('canvas');
      c.setAttribute('aria-hidden', 'true');
      this.el.appendChild(c);
      this.canvases[name] = c;
      this.ctx[name] = c.getContext('2d');
    }
    this.top = this.canvases[layers[layers.length - 1]];
    this.handlers = { down: [], move: [], up: [], leave: [] };
    this.resizeCbs = [];
    this.w = 0;
    this.h = 0;
    this._bindPointer();
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(this.el);
    this.resize();
  }

  // Change the data window. Callers redraw explicitly afterwards.
  setDomain(xd, yd) {
    this.xd = xd;
    this.yd = yd;
    this._layout();
  }

  onResize(cb) {
    this.resizeCbs.push(cb);
  }

  resize() {
    const w = Math.max(10, Math.round(this.el.clientWidth));
    const h = Math.round(Math.min(this.maxHeight, Math.max(this.minHeight, w * this.aspect)));
    if (w === this.w && h === this.h) return;
    this.w = w;
    this.h = h;
    this.el.style.height = `${h}px`;
    this.dpr = Math.min(window.devicePixelRatio || 1, 2.5);
    for (const [name, c] of Object.entries(this.canvases)) {
      c.width = Math.round(w * this.dpr);
      c.height = Math.round(h * this.dpr);
      this.ctx[name].setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    }
    this._layout();
    for (const cb of this.resizeCbs) cb();
  }

  _layout() {
    let x0 = this.pad.left;
    let x1 = this.w - this.pad.right;
    let y0 = this.pad.top;
    let y1 = this.h - this.pad.bottom;
    if (this.equal) {
      const sx = (x1 - x0) / (this.xd[1] - this.xd[0]);
      const sy = (y1 - y0) / (this.yd[1] - this.yd[0]);
      const s = Math.min(sx, sy);
      const cw = s * (this.xd[1] - this.xd[0]);
      const ch = s * (this.yd[1] - this.yd[0]);
      const cx = (x0 + x1) / 2;
      const cy = (y0 + y1) / 2;
      x0 = cx - cw / 2;
      x1 = cx + cw / 2;
      y0 = cy - ch / 2;
      y1 = cy + ch / 2;
    }
    Object.assign(this, { x0, x1, y0, y1 });
  }

  sx(v) { return this.x0 + ((v - this.xd[0]) / (this.xd[1] - this.xd[0])) * (this.x1 - this.x0); }
  sy(v) { return this.y1 - ((v - this.yd[0]) / (this.yd[1] - this.yd[0])) * (this.y1 - this.y0); }
  ix(px) { return this.xd[0] + ((px - this.x0) / (this.x1 - this.x0)) * (this.xd[1] - this.xd[0]); }
  iy(py) { return this.yd[0] + ((this.y1 - py) / (this.y1 - this.y0)) * (this.yd[1] - this.yd[0]); }

  clear(layer) {
    const c = this.ctx[layer];
    c.save();
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.clearRect(0, 0, this.canvases[layer].width, this.canvases[layer].height);
    c.restore();
  }

  on(type, fn) {
    this.handlers[type].push(fn);
  }

  _bindPointer() {
    const ev = (e) => {
      const r = this.top.getBoundingClientRect();
      const px = e.clientX - r.left;
      const py = e.clientY - r.top;
      return { px, py, x: this.ix(px), y: this.iy(py), inside: px >= this.x0 && px <= this.x1 && py >= this.y0 && py <= this.y1, event: e };
    };
    let dragging = false;
    this.top.addEventListener('pointerdown', (e) => {
      dragging = true;
      this.top.setPointerCapture(e.pointerId);
      for (const f of this.handlers.down) f(ev(e));
    });
    this.top.addEventListener('pointermove', (e) => {
      const d = ev(e);
      d.dragging = dragging;
      for (const f of this.handlers.move) f(d);
    });
    const up = (e) => {
      if (!dragging) return;
      dragging = false;
      for (const f of this.handlers.up) f(ev(e));
    };
    this.top.addEventListener('pointerup', up);
    this.top.addEventListener('pointercancel', up);
    this.top.addEventListener('pointerleave', (e) => {
      for (const f of this.handlers.leave) f(ev(e));
    });
  }
}

// ---------------------------------------------------------------------------
// Drawing helpers (all coordinates in CSS pixels)
// ---------------------------------------------------------------------------

export function font(size = 12, weight = 400, family = 'ui') {
  const fam = family === 'mono'
    ? 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace'
    : 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';
  return `${weight} ${size}px ${fam}`;
}

export function text(ctx, s, x, y, { color = '#000', size = 12, weight = 400, align = 'left', baseline = 'alphabetic', family = 'ui', halo = null } = {}) {
  ctx.font = font(size, weight, family);
  ctx.textAlign = align;
  ctx.textBaseline = baseline;
  if (halo) {
    ctx.lineWidth = 4;
    ctx.strokeStyle = halo;
    ctx.lineJoin = 'round';
    ctx.strokeText(s, x, y);
  }
  ctx.fillStyle = color;
  ctx.fillText(s, x, y);
}

export function frame(plot, ctx, T) {
  ctx.strokeStyle = T.axis;
  ctx.lineWidth = 1;
  ctx.strokeRect(Math.round(plot.x0) + 0.5, Math.round(plot.y0) + 0.5, Math.round(plot.x1 - plot.x0), Math.round(plot.y1 - plot.y0));
}

export function ticks(plot, ctx, T, { x = [], y = [], fmt = (v) => `${v}`, xfmt = null, yfmt = null, size = 11 } = {}) {
  for (const v of x) {
    text(ctx, (xfmt || fmt)(v), plot.sx(v), plot.y1 + 15, { color: T.muted, size, align: 'center' });
  }
  for (const v of y) {
    text(ctx, (yfmt || fmt)(v), plot.x0 - 7, plot.sy(v), { color: T.muted, size, align: 'right', baseline: 'middle' });
  }
}

export function axisLabels(plot, ctx, T, { x = '', y = '', size = 12 } = {}) {
  if (x) text(ctx, x, (plot.x0 + plot.x1) / 2, plot.h - 4, { color: T.ink2, size, align: 'center', baseline: 'bottom' });
  if (y) {
    ctx.save();
    ctx.translate(12, (plot.y0 + plot.y1) / 2);
    ctx.rotate(-Math.PI / 2);
    text(ctx, y, 0, 0, { color: T.ink2, size, align: 'center', baseline: 'middle' });
    ctx.restore();
  }
}

export function gridLines(plot, ctx, T, { x = [], y = [] } = {}) {
  ctx.strokeStyle = T.grid;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (const v of x) {
    const px = Math.round(plot.sx(v)) + 0.5;
    ctx.moveTo(px, plot.y0);
    ctx.lineTo(px, plot.y1);
  }
  for (const v of y) {
    const py = Math.round(plot.sy(v)) + 0.5;
    ctx.moveTo(plot.x0, py);
    ctx.lineTo(plot.x1, py);
  }
  ctx.stroke();
}

export function arrowHead(ctx, x0, y0, x1, y1, size = 7, color = '#000') {
  const a = Math.atan2(y1 - y0, x1 - x0);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - size * Math.cos(a - 0.42), y1 - size * Math.sin(a - 0.42));
  ctx.lineTo(x1 - size * Math.cos(a + 0.42), y1 - size * Math.sin(a + 0.42));
  ctx.closePath();
  ctx.fill();
}

export function arrow(ctx, x0, y0, x1, y1, { color = '#000', width = 1.2, head = 7 } = {}) {
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
  arrowHead(ctx, x0, y0, x1, y1, head, color);
}

export function polyline(ctx, pts, { color = '#000', width = 2, halo = null, alpha = 1, dash = null } = {}) {
  if (pts.length < 2) return;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  if (dash) ctx.setLineDash(dash);
  const path = () => {
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
  };
  if (halo) {
    path();
    ctx.strokeStyle = halo;
    ctx.lineWidth = width + 3;
    ctx.stroke();
  }
  path();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
  ctx.restore();
}

export function dot(ctx, x, y, r, color, ring = null, ringWidth = 2) {
  if (ring) {
    ctx.fillStyle = ring;
    ctx.beginPath();
    ctx.arc(x, y, r + ringWidth, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fill();
}

export function ring(ctx, x, y, r, color, width = 1.6, fill = null) {
  if (fill) {
    ctx.fillStyle = fill;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.stroke();
}

// Rest-point glyphs: sink filled, source open, saddle half-filled, center ringed dot, other a diamond.
export function marker(ctx, x, y, kind, T, r = 5.5) {
  ctx.fillStyle = T.surface;
  ctx.beginPath();
  ctx.arc(x, y, r + 2.5, 0, Math.PI * 2);
  ctx.fill();
  const ink = T.ink;
  if (kind === 'sink') {
    dot(ctx, x, y, r, ink);
  } else if (kind === 'source') {
    ring(ctx, x, y, r - 0.8, ink, 1.6, T.surface);
  } else if (kind === 'saddle') {
    ring(ctx, x, y, r - 0.8, ink, 1.6, T.surface);
    ctx.fillStyle = ink;
    ctx.beginPath();
    ctx.arc(x, y, r - 0.8, Math.PI / 2, (3 * Math.PI) / 2);
    ctx.closePath();
    ctx.fill();
  } else if (kind === 'center') {
    ring(ctx, x, y, r - 0.8, ink, 1.6, T.surface);
    dot(ctx, x, y, 1.8, ink);
  } else {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(Math.PI / 4);
    ctx.fillStyle = T.surface;
    ctx.fillRect(-r * 0.7, -r * 0.7, r * 1.4, r * 1.4);
    ctx.strokeStyle = ink;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(-r * 0.7, -r * 0.7, r * 1.4, r * 1.4);
    ctx.restore();
  }
}

export const MARKER_NAMES = { sink: 'stable', source: 'unstable', saddle: 'saddle', center: 'neutral (cycles)', degenerate: 'non-hyperbolic' };

// Sample f(x, y) -> [r, g, b, a] on a coarse grid and draw it smoothly over the plot area.
export function raster(plot, ctx, f, { res = 96, smooth = true, clip = null } = {}) {
  const w = Math.max(2, Math.round(res));
  const hgt = Math.max(2, Math.round((res * (plot.y1 - plot.y0)) / Math.max(1, plot.x1 - plot.x0)));
  const off = document.createElement('canvas');
  off.width = w;
  off.height = hgt;
  const octx = off.getContext('2d');
  const img = octx.createImageData(w, hgt);
  for (let j = 0; j < hgt; j++) {
    const y = plot.yd[1] - ((j + 0.5) / hgt) * (plot.yd[1] - plot.yd[0]);
    for (let i = 0; i < w; i++) {
      const x = plot.xd[0] + ((i + 0.5) / w) * (plot.xd[1] - plot.xd[0]);
      const c = f(x, y);
      const k = 4 * (j * w + i);
      if (!c) {
        img.data[k + 3] = 0;
        continue;
      }
      img.data[k] = c[0];
      img.data[k + 1] = c[1];
      img.data[k + 2] = c[2];
      img.data[k + 3] = c[3] ?? 255;
    }
  }
  octx.putImageData(img, 0, 0);
  ctx.save();
  if (clip) {
    clip(ctx);
    ctx.clip();
  }
  ctx.imageSmoothingEnabled = smooth;
  ctx.drawImage(off, plot.x0, plot.y0, plot.x1 - plot.x0, plot.y1 - plot.y0);
  ctx.restore();
}

// Barycentric (x1, x2, x3) -> data (X, Y) on the equilateral triangle (0,0), (1,0), (0.5, h).
export const TRI_H = Math.sqrt(3) / 2;

export function baryToXY(x) {
  return [x[1] + 0.5 * x[2], TRI_H * x[2]];
}

export function xyToBary(X, Y) {
  const x3 = Y / TRI_H;
  const x2 = X - 0.5 * x3;
  return [1 - x2 - x3, x2, x3];
}

export function trianglePath(plot, ctx) {
  ctx.beginPath();
  ctx.moveTo(plot.sx(0), plot.sy(0));
  ctx.lineTo(plot.sx(1), plot.sy(0));
  ctx.lineTo(plot.sx(0.5), plot.sy(TRI_H));
  ctx.closePath();
}
