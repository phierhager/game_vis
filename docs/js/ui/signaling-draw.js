// Drawing pieces specific to signaling games: pure-strategy corner glyphs
// and the states -> messages -> actions flow diagram.

import { mix } from './theme.js';
import { arrowHead, text } from './plot.js';

// A pure 2x2 mapping as a tiny wiring diagram centred at (cx, cy).
// targets[i] is the right-hand node that left node i points to.
export function mappingGlyph(ctx, cx, cy, targets, leftColors, rightColors, T, { w = 22, h = 16 } = {}) {
  const lx = cx - w / 2;
  const rx = cx + w / 2;
  const ys = [cy - h / 2, cy + h / 2];
  ctx.lineWidth = 1;
  for (let i = 0; i < 2; i++) {
    const y0 = ys[i];
    const y1 = ys[targets[i]];
    const a = Math.atan2(y1 - y0, rx - lx);
    const x0s = lx + 3.5 * Math.cos(a);
    const y0s = y0 + 3.5 * Math.sin(a);
    const x1s = rx - 4 * Math.cos(a);
    const y1s = y1 - 4 * Math.sin(a);
    ctx.strokeStyle = T.ink2;
    ctx.beginPath();
    ctx.moveTo(x0s, y0s);
    ctx.lineTo(x1s, y1s);
    ctx.stroke();
    arrowHead(ctx, x0s, y0s, x1s, y1s, 4.5, T.ink2);
  }
  for (let i = 0; i < 2; i++) {
    ctx.fillStyle = leftColors[i];
    ctx.beginPath();
    ctx.arc(lx, ys[i], 2.6, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = T.surface;
    ctx.strokeStyle = rightColors[i];
    ctx.lineWidth = 1.3;
    ctx.beginPath();
    ctx.arc(rx, ys[i], 2.6, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
}

// Corner glyphs for the sender square (states -> messages) or receiver square
// (messages -> actions). Corner (x, y) has x = P(first | first row), y = P(first | second row).
export function cornerGlyphs(plot, ctx, T, role) {
  const corners = [
    [1, 0, [0, 1]],
    [0, 1, [1, 0]],
    [1, 1, [0, 0]],
    [0, 0, [1, 1]],
  ];
  const state = [T.s1, T.s2];
  const grey = [T.ink2, T.ink2];
  for (const [x, y, tg] of corners) {
    const cx = plot.sx(x) + (x ? 25 : -25);
    const cy = plot.sy(y) + (y ? -19 : 19);
    if (role === 'sender') mappingGlyph(ctx, cx, cy, tg, state, grey, T);
    else mappingGlyph(ctx, cx, cy, tg, grey, state, T);
  }
}

// States -> messages -> actions. Band widths are probability flow and keep the
// colour of the state they carry. Draws inside the rectangle (x, y, w, h).
export function signalFlow(ctx, rect, game, S, R, T, { labels = true, alpha = 0.62 } = {}) {
  const { n, k, l } = game;
  const pi = game.prior;
  const flowTM = [];
  for (let t = 0; t < n; t++) for (let m = 0; m < k; m++) flowTM.push(pi[t] * S[t * k + m]);
  const pm = Array(k).fill(0);
  const pa = Array(l).fill(0);
  for (let t = 0; t < n; t++) for (let m = 0; m < k; m++) pm[m] += flowTM[t * k + m];
  for (let t = 0; t < n; t++) for (let m = 0; m < k; m++) for (let a = 0; a < l; a++) pa[a] += flowTM[t * k + m] * R[m * l + a];

  const labelW = labels ? 64 : 8;
  const x0 = rect.x + labelW;
  const x2 = rect.x + rect.w - labelW;
  const x1 = (x0 + x2) / 2;
  const nodeW = 7;
  const gap = Math.max(8, rect.h * 0.05);
  const rows = Math.max(n, k, l);
  const scale = rect.h - 16 - gap * (rows - 1);
  const top = rect.y + 14;
  const tops = (masses) => {
    const out = [];
    let y = top;
    for (const mval of masses) {
      out.push(y);
      y += mval * scale + gap;
    }
    return out;
  };
  const tTop = tops(pi);
  const mTop = tops(pm);
  const aTop = tops(pa);
  const stateColor = (t) => T.series[t % T.series.length];

  const band = (xa, ya0, ya1, xb, yb0, yb1, color) => {
    const xm = (xa + xb) / 2;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(xa, ya0);
    ctx.bezierCurveTo(xm, ya0, xm, yb0, xb, yb0);
    ctx.lineTo(xb, yb1);
    ctx.bezierCurveTo(xm, yb1, xm, ya1, xa, ya1);
    ctx.closePath();
    ctx.fill();
  };

  const tCur = [...tTop];
  const mCur = [...mTop];
  const mIn = {};
  for (let m = 0; m < k; m++) {
    for (let t = 0; t < n; t++) {
      const w = flowTM[t * k + m] * scale;
      if (w <= 0.05) continue;
      band(x0 + nodeW / 2, tCur[t], tCur[t] + w, x1 - nodeW / 2, mCur[m], mCur[m] + w, mix(stateColor(t), T.surface, alpha));
      mIn[`${t},${m}`] = mCur[m];
      tCur[t] += w;
      mCur[m] += w;
    }
  }
  const aCur = [...aTop];
  for (let a = 0; a < l; a++) {
    for (let m = 0; m < k; m++) {
      for (let t = 0; t < n; t++) {
        const w = flowTM[t * k + m] * R[m * l + a] * scale;
        if (w <= 0.05) continue;
        const key = `${t},${m}`;
        const ya = mIn[key];
        mIn[key] = ya + w;
        band(x1 + nodeW / 2, ya, ya + w, x2 - nodeW / 2, aCur[a], aCur[a] + w, mix(stateColor(t), T.surface, alpha));
        aCur[a] += w;
      }
    }
  }
  // nodes
  for (let t = 0; t < n; t++) {
    ctx.fillStyle = stateColor(t);
    ctx.fillRect(x0 - nodeW / 2, tTop[t], nodeW, Math.max(pi[t] * scale, 1.5));
    if (labels) text(ctx, game.stateLabels[t], x0 - nodeW / 2 - 6, tTop[t] + (pi[t] * scale) / 2, { color: T.ink2, size: 12, align: 'right', baseline: 'middle' });
  }
  for (let m = 0; m < k; m++) {
    ctx.fillStyle = T.ink2;
    ctx.fillRect(x1 - nodeW / 2, mTop[m], nodeW, Math.max(pm[m] * scale, 1.5));
    if (labels) text(ctx, game.messageLabels[m], x1, mTop[m] - 3, { color: T.ink2, size: 11, align: 'center', baseline: 'bottom', halo: T.surface });
  }
  for (let a = 0; a < l; a++) {
    ctx.fillStyle = T.ink2;
    ctx.fillRect(x2 - nodeW / 2, aTop[a], nodeW, Math.max(pa[a] * scale, 1.5));
    if (labels) {
      const y = aTop[a] + (pa[a] * scale) / 2;
      text(ctx, game.actionLabels[a], x2 + nodeW / 2 + 6, y, { color: T.ink2, size: 12, align: 'left', baseline: 'middle' });
    }
  }
}
