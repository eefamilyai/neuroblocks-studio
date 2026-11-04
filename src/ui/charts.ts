/**
 * A small, fast canvas line chart: smoothing (cached and extended
 * incrementally), "nice" axis ticks, hover read-out, dashed overlays for
 * earlier runs. Redraws only when something changed.
 */
import { icon } from './icons.ts';
import { fmtSteps, h } from './dom.ts';

export interface Series {
  name: string;
  color: string;
  xs: ArrayLike<number>;
  ys: ArrayLike<number>;
  /** draw raw points as faint dots under the smoothed line */
  dots?: boolean;
  dashed?: boolean;
  width?: number;
  /** don't smooth (e.g. epsilon, success rate) */
  raw?: boolean;
  /** fill the area under the line */
  fill?: boolean;
}

const FONT = '10px Inter, ui-sans-serif, system-ui, sans-serif';

/** "Nice" tick step: 1, 2, 2.5 or 5 × 10ⁿ, giving about `n` ticks over `span`. */
export function niceStep(span: number, n = 4): number {
  if (!(span > 0)) return 1;
  const raw = span / n;
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const m = raw / p;
  return (m < 1.5 ? 1 : m < 2.25 ? 2 : m < 3.5 ? 2.5 : m < 7.5 ? 5 : 10) * p;
}

function tickLabel(v: number, step: number): string {
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e5 || (a < 1e-3 && step < 1e-3)) return v.toExponential(1);
  const dec = Math.max(0, Math.min(6, -Math.floor(Math.log10(step) + 1e-9)));
  return v.toFixed(step % 1 === 0 && dec === 0 ? 0 : dec);
}

interface SmoothCache {
  ys: ArrayLike<number>;
  n: number;
  alpha: number;
  acc: number;
  debias: number;
  last: number;
  out: Float32Array;
}

export class LineChart {
  readonly el: HTMLElement;
  private canvas: HTMLCanvasElement;
  private tip: HTMLElement;
  private legend: HTMLElement;
  series: Series[] = [];
  smoothing = 0.9;
  private hoverX: number | null = null;
  private dirty = true;
  private yRange: [number, number] | null;
  private cache = new WeakMap<ArrayLike<number>, SmoothCache>();
  private legendKey = '';
  /** called after drawing, e.g. to overlay markers */
  onDraw: ((g: CanvasRenderingContext2D, sx: (x: number) => number, plot: { L: number; T: number; W: number; H: number }) => void) | null = null;
  xLabel: (x: number) => string = fmtSteps;

  constructor(title: string, help: string, opts: { tall?: boolean; yRange?: [number, number] } = {}) {
    this.canvas = h('canvas');
    this.tip = h('div', { class: 'chart-tip', style: { display: 'none' } });
    this.legend = h('span', { class: 'legend' });
    this.yRange = opts.yRange ?? null;
    this.el = h('div', { class: 'chart' + (opts.tall ? ' tall' : '') }, this.canvas, h('div', { class: 'chart-title' }, h('span', null, title), this.legend), help ? h('div', { class: 'chart-help', title: help }, icon('help')) : null, this.tip);
    this.canvas.addEventListener('mousemove', (e) => {
      const r = this.canvas.getBoundingClientRect();
      this.hoverX = e.clientX - r.left;
      this.dirty = true;
      this.draw();
    });
    this.canvas.addEventListener('mouseleave', () => {
      this.hoverX = null;
      this.tip.style.display = 'none';
      this.dirty = true;
      this.draw();
    });
    new ResizeObserver(() => {
      this.dirty = true;
      this.draw();
    }).observe(this.el);
  }

  setSeries(s: Series[]): void {
    this.series = s;
    const key = s
      .filter((se) => !se.name.startsWith('_') && se.name)
      .map((se) => se.name + se.color)
      .join('|');
    if (key !== this.legendKey) {
      this.legendKey = key;
      this.legend.textContent = '';
      for (const se of s) if (se.name && !se.name.startsWith('_')) this.legend.append(h('span', null, h('i', { style: { background: se.color } }), se.name));
    }
    this.dirty = true;
  }

  invalidate(): void {
    this.dirty = true;
  }

  /** TensorBoard-style debiased EMA, continued from where the last call stopped when data was only appended. */
  private smooth(ys: ArrayLike<number>, raw: boolean): ArrayLike<number> {
    const a = raw ? 0 : this.smoothing;
    if (a <= 0) return ys;
    let c = this.cache.get(ys);
    if (!c || c.alpha !== a || ys.length < c.n) {
      c = { ys, n: 0, alpha: a, acc: 0, debias: 0, last: NaN, out: new Float32Array(Math.max(64, ys.length * 1.5)) };
      this.cache.set(ys, c);
    }
    if (c.out.length < ys.length) {
      const o = new Float32Array(Math.ceil(ys.length * 1.5));
      o.set(c.out.subarray(0, c.n));
      c.out = o;
    }
    for (let i = c.n; i < ys.length; i++) {
      const y = ys[i];
      if (Number.isFinite(y)) {
        c.acc = c.acc * a + (1 - a) * y;
        c.debias = c.debias * a + (1 - a);
        c.last = c.acc / c.debias;
      }
      c.out[i] = c.last;
    }
    c.n = ys.length;
    return c.out;
  }

  draw(): void {
    if (!this.dirty || !this.el.isConnected) return;
    const W = this.el.clientWidth;
    const H = this.el.clientHeight;
    if (W < 10 || H < 10) return;
    this.dirty = false;
    const c = this.canvas;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) {
      c.width = Math.round(W * dpr);
      c.height = Math.round(H * dpr);
    }
    const g = c.getContext('2d')!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const smoothed = this.series.map((s) => this.smooth(s.ys, !!s.raw));
    let x0 = Infinity;
    let x1 = -Infinity;
    let y0 = Infinity;
    let y1 = -Infinity;
    this.series.forEach((s, k) => {
      const n = Math.min(s.xs.length, s.ys.length);
      if (!n) return;
      x0 = Math.min(x0, s.xs[0]);
      x1 = Math.max(x1, s.xs[n - 1]);
      const ys = s.dots ? s.ys : smoothed[k];
      for (let i = 0; i < n; i++) {
        const y = ys[i];
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    });
    g.font = FONT;
    if (!Number.isFinite(x0) || !Number.isFinite(y0)) {
      g.fillStyle = '#566078';
      g.textAlign = 'center';
      g.fillText('no data yet', W / 2, H / 2 + 8);
      return;
    }
    if (this.yRange) {
      y0 = Math.min(y0, this.yRange[0]);
      y1 = Math.max(y1, this.yRange[1]);
    }
    if (y1 - y0 < 1e-9) {
      const m = Math.abs(y0) * 0.1 || 1;
      y0 -= m;
      y1 += m;
    }
    const step = niceStep(y1 - y0, 4);
    const ty0 = Math.floor(y0 / step) * step;
    const ty1 = Math.ceil(y1 / step) * step;
    y0 = ty0;
    y1 = ty1;
    if (x1 <= x0) x1 = x0 + 1;
    // left margin fits the widest tick label
    let labW = 0;
    const ticks: number[] = [];
    for (let v = ty0, i = 0; v <= ty1 + step * 0.01 && i < 12; v += step, i++) {
      const vv = Math.abs(v) < step * 1e-6 ? 0 : v;
      ticks.push(vv);
      labW = Math.max(labW, g.measureText(tickLabel(vv, step)).width);
    }
    const L = Math.ceil(labW) + 12;
    const R = 10;
    const T = 26;
    const B = 18;
    const pw = W - L - R;
    const ph = H - T - B;
    const sx = (x: number) => L + ((x - x0) / (x1 - x0)) * pw;
    const sy = (y: number) => T + (1 - (y - y0) / (y1 - y0)) * ph;
    // grid + y labels
    g.lineWidth = 1;
    g.textAlign = 'right';
    for (const v of ticks) {
      const py = Math.round(sy(v)) + 0.5;
      g.strokeStyle = v === 0 ? 'rgba(255,255,255,0.13)' : 'rgba(255,255,255,0.045)';
      g.beginPath();
      g.moveTo(L, py);
      g.lineTo(W - R, py);
      g.stroke();
      g.fillStyle = '#5f687d';
      g.fillText(tickLabel(v, step), L - 6, py + 3.5);
    }
    // x labels
    const xstep = niceStep(x1 - x0, Math.max(2, Math.floor(pw / 70)));
    g.textAlign = 'center';
    g.fillStyle = '#5f687d';
    let prevLabel = '';
    for (let v = Math.ceil(x0 / xstep) * xstep; v <= x1; v += xstep) {
      const px = sx(v);
      if (px < L + 8 || px > W - R - 8) continue;
      const lab = this.xLabel(v);
      if (lab === prevLabel) continue;
      prevLabel = lab;
      g.fillText(lab, px, H - 5);
    }
    // series
    g.save();
    g.beginPath();
    g.rect(L, T - 2, pw, ph + 4);
    g.clip();
    this.series.forEach((s, k) => {
      const n = Math.min(s.xs.length, s.ys.length);
      if (!n) return;
      if (s.dots) {
        g.fillStyle = s.color + '38';
        const st = Math.max(1, Math.floor(n / 1600));
        for (let i = 0; i < n; i += st) {
          const y = s.ys[i];
          if (Number.isFinite(y)) g.fillRect(sx(s.xs[i]) - 1, sy(y) - 1, 2, 2);
        }
      }
      const ys = smoothed[k];
      // one point per pixel column at most
      const st = Math.max(1, Math.floor(n / Math.max(50, pw)));
      g.beginPath();
      let started = false;
      let firstX = 0;
      let lastX = 0;
      for (let i = 0; i < n; i += st) {
        const y = ys[i];
        if (!Number.isFinite(y)) continue;
        const px = sx(s.xs[i]);
        const py = sy(y);
        if (!started) {
          g.moveTo(px, py);
          firstX = px;
          started = true;
        } else g.lineTo(px, py);
        lastX = px;
      }
      const yl = ys[n - 1];
      if (Number.isFinite(yl)) {
        g.lineTo(sx(s.xs[n - 1]), sy(yl));
        lastX = sx(s.xs[n - 1]);
      }
      g.strokeStyle = s.color;
      g.lineWidth = s.width ?? 1.75;
      g.lineJoin = 'round';
      g.setLineDash(s.dashed ? [4, 4] : []);
      g.stroke();
      g.setLineDash([]);
      if (s.fill && started) {
        g.lineTo(lastX, sy(Math.max(y0, Math.min(y1, 0))));
        g.lineTo(firstX, sy(Math.max(y0, Math.min(y1, 0))));
        g.closePath();
        g.fillStyle = s.color + '1f';
        g.fill();
      }
    });
    g.restore();
    this.onDraw?.(g, sx, { L, T, W: pw, H: ph });
    // hover read-out
    if (this.hoverX !== null && this.hoverX > L && this.hoverX < W - R) {
      const xv = x0 + ((this.hoverX - L) / pw) * (x1 - x0);
      g.strokeStyle = 'rgba(255,255,255,0.22)';
      g.beginPath();
      g.moveTo(Math.round(this.hoverX) + 0.5, T);
      g.lineTo(Math.round(this.hoverX) + 0.5, T + ph);
      g.stroke();
      const lines: string[] = [`${this.xLabel === fmtSteps ? 'step ' : ''}${this.xLabel(xv)}`];
      this.series.forEach((s, k) => {
        const n = Math.min(s.xs.length, s.ys.length);
        if (!n || s.name.startsWith('_')) return;
        let lo = 0;
        let hi = n - 1;
        while (lo < hi) {
          const mid = (lo + hi) >> 1;
          if (s.xs[mid] < xv) lo = mid + 1;
          else hi = mid;
        }
        const v = smoothed[k][lo];
        if (Number.isFinite(v)) {
          lines.push(`${s.name || 'value'}: ${Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : v.toFixed(3)}`);
          g.fillStyle = s.color;
          g.beginPath();
          g.arc(sx(s.xs[lo]), sy(v), 3, 0, Math.PI * 2);
          g.fill();
        }
      });
      this.tip.innerHTML = lines.join('<br>');
      this.tip.style.display = 'block';
      const tw = this.tip.offsetWidth;
      this.tip.style.left = (this.hoverX + 12 + tw > W ? this.hoverX - tw - 12 : this.hoverX + 12) + 'px';
      this.tip.style.top = '28px';
    }
  }
}
