/**
 * A small, fast canvas line chart with smoothing, min/max range bands,
 * hover read-out and "previous run" overlays for comparing experiments.
 */
import { fmt, fmtSteps, h } from './dom.ts';

export interface Series {
  name: string;
  color: string;
  xs: number[];
  ys: number[];
  /** draw raw points as faint dots under the smoothed line */
  dots?: boolean;
  dashed?: boolean;
  width?: number;
  /** don't smooth (e.g. epsilon) */
  raw?: boolean;
}

export class LineChart {
  readonly el: HTMLElement;
  private canvas: HTMLCanvasElement;
  private tip: HTMLElement;
  private legend: HTMLElement;
  series: Series[] = [];
  smoothing = 0.9;
  yLabel = '';
  logY = false;
  private hoverX: number | null = null;
  private dirty = true;
  private yRange: [number, number] | null;

  constructor(title: string, help: string, opts: { tall?: boolean; yRange?: [number, number] } = {}) {
    this.canvas = h('canvas');
    this.tip = h('div', { class: 'chart-tip', style: { display: 'none' } });
    this.legend = h('span', { class: 'legend' });
    this.yRange = opts.yRange ?? null;
    this.el = h(
      'div',
      { class: 'chart' + (opts.tall ? ' tall' : '') },
      this.canvas,
      h('div', { class: 'chart-title' }, title, this.legend),
      h('div', { class: 'chart-help', title: help }, '?'),
      this.tip,
    );
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
    this.legend.innerHTML = '';
    for (const se of s) if (!se.name.startsWith('_')) this.legend.append(h('span', null, h('i', { style: { background: se.color } }), se.name));
    this.dirty = true;
  }

  invalidate(): void {
    this.dirty = true;
  }

  private smooth(ys: number[]): number[] {
    const out = new Array<number>(ys.length);
    const a = this.smoothing;
    let last = NaN;
    let debias = 0;
    let acc = 0;
    for (let i = 0; i < ys.length; i++) {
      const y = ys[i];
      if (!Number.isFinite(y)) {
        out[i] = last;
        continue;
      }
      // TensorBoard-style debiased exponential moving average
      acc = acc * a + (1 - a) * y;
      debias = debias * a + (1 - a);
      last = acc / debias;
      out[i] = last;
    }
    return out;
  }

  draw(): void {
    if (!this.dirty) return;
    this.dirty = false;
    const c = this.canvas;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const W = Math.max(10, this.el.clientWidth);
    const H = Math.max(10, this.el.clientHeight);
    if (c.width !== W * dpr || c.height !== H * dpr) {
      c.width = W * dpr;
      c.height = H * dpr;
    }
    const g = c.getContext('2d')!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const L = 44;
    const R = 10;
    const T = 24;
    const B = 18;
    const pw = W - L - R;
    const ph = H - T - B;
    const smoothed = this.series.map((s) => (s.raw ? s.ys : this.smooth(s.ys)));
    let x0 = Infinity;
    let x1 = -Infinity;
    let y0 = Infinity;
    let y1 = -Infinity;
    this.series.forEach((s, k) => {
      if (!s.xs.length) return;
      x0 = Math.min(x0, s.xs[0]);
      x1 = Math.max(x1, s.xs[s.xs.length - 1]);
      const ys = s.dots ? s.ys : smoothed[k];
      for (const y of ys) {
        if (!Number.isFinite(y)) continue;
        y0 = Math.min(y0, y);
        y1 = Math.max(y1, y);
      }
    });
    g.font = '10px Nunito, sans-serif';
    if (!Number.isFinite(x0)) {
      g.fillStyle = '#5b6388';
      g.textAlign = 'center';
      g.fillText('waiting for data… press ▶ Train', W / 2, H / 2 + 6);
      return;
    }
    if (this.yRange) {
      y0 = Math.min(y0, this.yRange[0]);
      y1 = Math.max(y1, this.yRange[1]);
    }
    if (y1 - y0 < 1e-9) {
      y0 -= 1;
      y1 += 1;
    }
    const pad = (y1 - y0) * 0.08;
    y0 -= pad;
    y1 += pad;
    if (x1 <= x0) x1 = x0 + 1;
    const sx = (x: number) => L + ((x - x0) / (x1 - x0)) * pw;
    const sy = (y: number) => T + (1 - (y - y0) / (y1 - y0)) * ph;
    // grid
    g.strokeStyle = 'rgba(255,255,255,0.06)';
    g.fillStyle = '#7d86ad';
    g.lineWidth = 1;
    g.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
      const y = y0 + ((y1 - y0) * i) / 4;
      const py = Math.round(sy(y)) + 0.5;
      g.beginPath();
      g.moveTo(L, py);
      g.lineTo(W - R, py);
      g.stroke();
      g.fillText(fmt(y, 2), L - 4, py + 3);
    }
    g.textAlign = 'center';
    for (let i = 0; i <= 4; i++) {
      const x = x0 + ((x1 - x0) * i) / 4;
      g.fillText(fmtSteps(x), sx(x), H - 5);
    }
    if (y0 < 0 && y1 > 0) {
      g.strokeStyle = 'rgba(255,255,255,0.18)';
      g.beginPath();
      g.moveTo(L, sy(0));
      g.lineTo(W - R, sy(0));
      g.stroke();
    }
    // series
    this.series.forEach((s, k) => {
      const n = s.xs.length;
      if (!n) return;
      if (s.dots) {
        g.fillStyle = s.color + '40';
        const step = Math.max(1, Math.floor(n / 1500));
        for (let i = 0; i < n; i += step) g.fillRect(sx(s.xs[i]) - 1, sy(s.ys[i]) - 1, 2, 2);
      }
      const ys = smoothed[k];
      g.strokeStyle = s.color;
      g.lineWidth = s.width ?? 2;
      g.setLineDash(s.dashed ? [5, 4] : []);
      g.beginPath();
      const step = Math.max(1, Math.floor(n / 1200));
      let started = false;
      for (let i = 0; i < n; i += step) {
        const y = ys[i];
        if (!Number.isFinite(y)) continue;
        const px = sx(s.xs[i]);
        const py = sy(y);
        if (!started) {
          g.moveTo(px, py);
          started = true;
        } else g.lineTo(px, py);
      }
      const yl = ys[n - 1];
      if (Number.isFinite(yl)) g.lineTo(sx(s.xs[n - 1]), sy(yl));
      g.stroke();
      g.setLineDash([]);
    });
    // hover read-out
    if (this.hoverX !== null && this.hoverX > L) {
      const xv = x0 + ((this.hoverX - L) / pw) * (x1 - x0);
      g.strokeStyle = 'rgba(255,255,255,0.3)';
      g.beginPath();
      g.moveTo(this.hoverX, T);
      g.lineTo(this.hoverX, T + ph);
      g.stroke();
      const lines: string[] = [`step ${fmtSteps(xv)}`];
      this.series.forEach((s, k) => {
        if (!s.xs.length || s.name.startsWith('_')) return;
        let lo = 0;
        let hi = s.xs.length - 1;
        while (lo < hi) {
          const mid = (lo + hi) >> 1;
          if (s.xs[mid] < xv) lo = mid + 1;
          else hi = mid;
        }
        const v = smoothed[k][lo];
        if (Number.isFinite(v)) {
          lines.push(`${s.name}: ${fmt(v, 3)}`);
          g.fillStyle = s.color;
          g.beginPath();
          g.arc(sx(s.xs[lo]), sy(v), 3, 0, Math.PI * 2);
          g.fill();
        }
      });
      this.tip.innerHTML = lines.join('<br>');
      this.tip.style.display = 'block';
      const tx = this.hoverX + 12;
      this.tip.style.left = Math.min(tx, W - this.tip.offsetWidth - 4) + 'px';
      this.tip.style.top = '26px';
    }
  }
}
