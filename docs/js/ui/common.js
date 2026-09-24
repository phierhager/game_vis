// Page chrome and small DOM helpers shared by every explorer.

import { initThemeToggle } from './theme.js';

export function initPage() {
  const b = document.querySelector('.theme-toggle');
  if (b) initThemeToggle(b);
}

export const $ = (sel, root = document) => root.querySelector(sel);

// Bind a range input to a callback; keeps its <output> in sync.
export function bindRange(id, format, onInput) {
  const input = document.getElementById(id);
  const out = input.closest('.control')?.querySelector('output');
  const update = () => {
    const v = parseFloat(input.value);
    if (out) out.textContent = format(v);
    return v;
  };
  input.addEventListener('input', () => onInput(update()));
  update();
  return {
    input,
    get value() { return parseFloat(input.value); },
    set(v) {
      input.value = v;
      update();
    },
  };
}

export function bindSelect(id, onChange) {
  const el = document.getElementById(id);
  el.addEventListener('change', () => onChange(el.value));
  return el;
}

export function pct(v, digits = 0) {
  return `${(100 * v).toFixed(digits)}%`;
}

export function fixed(v, d = 2) {
  return Number.isFinite(v) ? v.toFixed(d) : '–';
}

export function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') e.className = v;
    else if (k === 'style') Object.assign(e.style, v);
    else e.setAttribute(k, v);
  }
  for (const c of children) e.append(c);
  return e;
}

// Run `work(budgetMs)` in slices across animation frames until it returns true.
export function sliced(work, onDone = () => {}) {
  let cancelled = false;
  const frame = () => {
    if (cancelled) return;
    const done = work(8);
    if (done) onDone();
    else requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
  return () => { cancelled = true; };
}
