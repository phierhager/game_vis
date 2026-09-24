// Read colour tokens from CSS so canvases follow the page theme.

const NAMES = ['page', 'surface', 'raised', 'ink', 'ink-2', 'muted', 'grid', 'axis', 'neutral', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8'];

export function tokens() {
  const cs = getComputedStyle(document.documentElement);
  const t = {};
  for (const n of NAMES) t[n.replace('-', '')] = cs.getPropertyValue(`--${n}`).trim() || '#888888';
  t.series = [t.s1, t.s2, t.s3, t.s4, t.s5, t.s6, t.s7, t.s8];
  t.dark = isDark();
  return t;
}

export function isDark() {
  const set = document.documentElement.getAttribute('data-theme');
  if (set) return set === 'dark';
  return window.matchMedia('(prefers-color-scheme: dark)').matches;
}

export function onThemeChange(cb) {
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', cb);
  new MutationObserver(cb).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
}

export function hexToRgb(hex) {
  let h = hex.replace('#', '');
  if (h.length === 3) h = h.split('').map((c) => c + c).join('');
  const v = parseInt(h, 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

export function alpha(hex, a) {
  const [r, g, b] = hexToRgb(hex);
  return `rgba(${r},${g},${b},${a})`;
}

// Blend `color` toward `base`: amount 1 is the pure colour.
export function mix(color, base, amount) {
  const c = hexToRgb(color);
  const s = hexToRgb(base);
  const m = c.map((v, i) => Math.round(s[i] + amount * (v - s[i])));
  return `rgb(${m[0]},${m[1]},${m[2]})`;
}

export function mixRgb(color, base, amount) {
  const c = hexToRgb(color);
  const s = hexToRgb(base);
  return c.map((v, i) => Math.round(s[i] + amount * (v - s[i])));
}

// Theme toggle: cycles system -> light -> dark, remembered per viewer when storage allows.
export function initThemeToggle(button) {
  const KEY = 'ppp-theme';
  const label = () => {
    const set = document.documentElement.getAttribute('data-theme');
    button.textContent = set === 'dark' ? 'Dark' : set === 'light' ? 'Light' : 'Auto';
    button.setAttribute('aria-label', `Colour theme: ${button.textContent}. Click to change.`);
  };
  try {
    const saved = localStorage.getItem(KEY);
    if (saved === 'light' || saved === 'dark') document.documentElement.setAttribute('data-theme', saved);
  } catch (e) {
    /* storage unavailable: keep the system theme */
  }
  label();
  button.addEventListener('click', () => {
    const set = document.documentElement.getAttribute('data-theme');
    const next = set === null ? 'light' : set === 'light' ? 'dark' : null;
    if (next) document.documentElement.setAttribute('data-theme', next);
    else document.documentElement.removeAttribute('data-theme');
    try {
      if (next) localStorage.setItem(KEY, next);
      else localStorage.removeItem(KEY);
    } catch (e) {
      /* ignore */
    }
    label();
  });
}
