/**
 * Embeddings: every token is a vector of d numbers. PCA squashes them to 2D
 * (the two directions along which the vectors differ most). Tokens that the
 * model uses in similar ways end up close together.
 */
import { clear, fmt, h, section, seg, setText } from '../../ui/dom.ts';
import { icon } from '../../ui/icons.ts';
import { showToken } from '../tokenizer.ts';
import type { LLMStudio } from './LLMStudio.ts';

/** Top-2 principal components of N row vectors of size d. */
export function pca2(X: Float32Array, N: number, d: number): { pts: Float32Array; explained: [number, number] } {
  const mean = new Float64Array(d);
  for (let n = 0; n < N; n++) for (let i = 0; i < d; i++) mean[i] += X[n * d + i] / N;
  const C = new Float64Array(d * d);
  const row = new Float64Array(d);
  for (let n = 0; n < N; n++) {
    for (let i = 0; i < d; i++) row[i] = X[n * d + i] - mean[i];
    for (let i = 0; i < d; i++) {
      const ri = row[i];
      if (ri === 0) continue;
      const o = i * d;
      for (let j = i; j < d; j++) C[o + j] += ri * row[j];
    }
  }
  let trace = 0;
  for (let i = 0; i < d; i++) {
    for (let j = i; j < d; j++) {
      C[i * d + j] /= N;
      C[j * d + i] = C[i * d + j];
    }
    trace += C[i * d + i];
  }
  const vecs: Float64Array[] = [];
  const vals: number[] = [];
  const tmp = new Float64Array(d);
  for (let k = 0; k < 2; k++) {
    const v = new Float64Array(d);
    for (let i = 0; i < d; i++) v[i] = Math.sin(i * 1.7 + k * 0.9 + 0.3) + 0.1;
    let lambda = 0;
    for (let it = 0; it < 200; it++) {
      for (let i = 0; i < d; i++) {
        let s = 0;
        for (let j = 0; j < d; j++) s += C[i * d + j] * v[j];
        tmp[i] = s;
      }
      // deflate: remove the components already found
      for (const u of vecs) {
        let dot = 0;
        for (let i = 0; i < d; i++) dot += tmp[i] * u[i];
        for (let i = 0; i < d; i++) tmp[i] -= dot * u[i];
      }
      let norm = 0;
      for (let i = 0; i < d; i++) norm += tmp[i] * tmp[i];
      norm = Math.sqrt(norm);
      if (norm < 1e-12) break;
      let diff = 0;
      for (let i = 0; i < d; i++) {
        const nv = tmp[i] / norm;
        diff += Math.abs(nv - v[i]);
        v[i] = nv;
      }
      lambda = norm;
      if (diff < 1e-9) break;
    }
    vecs.push(v);
    vals.push(lambda);
  }
  const pts = new Float32Array(N * 2);
  for (let n = 0; n < N; n++)
    for (let k = 0; k < 2; k++) {
      let s = 0;
      for (let i = 0; i < d; i++) s += (X[n * d + i] - mean[i]) * vecs[k][i];
      pts[n * 2 + k] = s;
    }
  return { pts, explained: [trace > 0 ? vals[0] / trace : 0, trace > 0 ? vals[1] / trace : 0] };
}

const CATS: [string, string][] = [
  ['space / new line', '#8b93a7'],
  ['vowel', '#f26b7d'],
  ['consonant', '#6e8bff'],
  ['digit', '#f2b84b'],
  ['punctuation', '#3ccf8e'],
  ['word piece', '#9b7bff'],
  ['capital', '#5ab4ff'],
];

function category(s: string): number {
  if (!s.trim()) return 0;
  const t = s.trim();
  if (t.length === 1) {
    if (/[aeiou]/.test(t)) return 1;
    if (/[A-Z]/.test(t)) return 6;
    if (/[a-z]/.test(t)) return 2;
    if (/[0-9]/.test(t)) return 3;
    return 4;
  }
  if (/[A-Za-z]/.test(t)) return /^[A-Z]/.test(t) ? 6 : 5;
  if (/[0-9]/.test(t)) return 3;
  return 4;
}

export class EmbeddingView {
  readonly el: HTMLElement;
  private studio: LLMStudio;
  private mode: 'tokens' | 'positions' = 'tokens';
  private canvas: HTMLCanvasElement;
  private wrap: HTMLElement;
  private info: HTMLElement;
  private legend: HTMLElement;
  private near: HTMLElement;
  private msg: HTMLElement;
  private body: HTMLElement;
  private pts: Float32Array = new Float32Array(0);
  private labels: string[] = [];
  private cats: number[] = [];
  private N = 0;
  private hover = -1;
  private sel = -1;
  private key = '';
  private lastCompute = 0;
  private drawn = '';
  private screen: Float32Array = new Float32Array(0);

  constructor(studio: LLMStudio) {
    this.studio = studio;
    const modeSeg = seg<'tokens' | 'positions'>(
      [
        { value: 'tokens', label: 'Token embeddings', title: 'One vector per token in the vocabulary' },
        { value: 'positions', label: 'Position embeddings', title: 'One vector per position in the context window' },
      ],
      (v) => {
        this.mode = v;
        modeSeg.set(v);
        this.sel = -1;
        this.key = '';
      },
      'sm',
    );
    modeSeg.set('tokens');
    this.canvas = h('canvas', { class: 'llm-scatter' });
    this.wrap = h('div', { class: 'llm-scatter-wrap' }, this.canvas);
    this.info = h('div', { class: 'note' });
    this.legend = h('div', { class: 'llm-legend' });
    this.near = h('div');
    this.msg = h('div', { class: 'empty-note' }, icon('orbit'), h('div', null, h('b', null, 'No model yet.'), h('div', { class: 'dim small' }, 'Train a model to see how it arranges its tokens. At first they are scattered randomly; after training, similar tokens cluster: vowels, digits, punctuation…')));
    this.body = h('div', null, h('div', { class: 'row', style: { marginBottom: '8px' } }, modeSeg.el), this.wrap, this.legend, this.info, section('Nearest neighbours', 'crosshair', [this.near]));
    this.el = h('div', null, this.msg, this.body);
    this.canvas.addEventListener('mousemove', (e) => {
      const i = this.pick(e);
      if (i !== this.hover) {
        this.hover = i;
        this.drawn = '';
      }
    });
    this.canvas.addEventListener('mouseleave', () => {
      this.hover = -1;
      this.drawn = '';
    });
    this.canvas.addEventListener('click', (e) => {
      this.sel = this.pick(e);
      this.drawn = '';
      this.showNeighbours();
    });
    new ResizeObserver(() => (this.drawn = '')).observe(this.wrap);
  }

  render(visible: boolean): void {
    if (!visible) return;
    const tr = this.studio.trained;
    this.msg.classList.toggle('hidden', !!tr);
    this.body.classList.toggle('hidden', !tr);
    if (!tr) return;
    const key = `${this.studio.version}|${this.mode}`;
    const now = performance.now();
    if (key !== this.key && (now - this.lastCompute > 2000 || !this.key.endsWith(this.mode))) {
      this.key = key;
      this.lastCompute = now;
      this.compute();
      this.showNeighbours();
    }
    const want = `${this.key}|${this.hover}|${this.sel}|${this.wrap.clientWidth}`;
    if (want !== this.drawn) {
      this.drawn = want;
      this.draw();
    }
  }

  private rows(): { X: Float32Array; N: number } | null {
    const m = this.studio.trained!.model;
    const { d, ctx, vocab } = m.cfg;
    if (this.mode === 'tokens') return { X: m.wte.w, N: vocab };
    if (m.cfg.pos === 'none') return null;
    const X = new Float32Array(ctx * d);
    for (let t = 0; t < ctx; t++) X.set(m.posVec(t)!, t * d);
    return { X, N: ctx };
  }

  private compute(): void {
    const tr = this.studio.trained!;
    const r = this.rows();
    clear(this.legend);
    if (!r) {
      this.N = 0;
      setText(this.info, 'This model has no position embeddings (positions = none): it cannot tell the order of the tokens, except through the causal mask.');
      return;
    }
    const { d } = tr.model.cfg;
    const { pts, explained } = pca2(r.X, r.N, d);
    this.pts = pts;
    this.N = r.N;
    if (this.mode === 'tokens') {
      this.labels = tr.tok.vocab.map(showToken);
      this.cats = tr.tok.vocab.map(category);
      const used = new Set(this.cats);
      for (const [i, [name, col]] of CATS.entries()) if (used.has(i)) this.legend.append(h('span', null, h('i', { style: { background: col } }), name));
      setText(this.info, `${r.N} tokens × ${d} numbers each, projected onto the 2 directions where they differ most (PCA: ${fmt(explained[0] * 100, 0)}% + ${fmt(explained[1] * 100, 0)}% of the spread). Click a token to see its nearest neighbours in the full ${d}-dimensional space.`);
    } else {
      this.labels = Array.from({ length: r.N }, (_, t) => String(t));
      this.cats = [];
      setText(this.info, tr.model.cfg.pos === 'learned' ? `The ${r.N} learned position vectors (PCA: ${fmt(explained[0] * 100, 0)}% + ${fmt(explained[1] * 100, 0)}%). Neighbouring positions usually end up next to each other: the model learns a smooth "ruler".` : `The ${r.N} fixed sinusoidal position vectors (not learned): sine waves of different frequencies, so every position gets a unique pattern.`);
    }
  }

  private draw(): void {
    const c = this.canvas;
    const W = Math.max(200, this.wrap.clientWidth);
    const H = Math.round(Math.min(520, Math.max(260, W * 0.8)));
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    c.width = Math.round(W * dpr);
    c.height = Math.round(H * dpr);
    c.style.width = `${W}px`;
    c.style.height = `${H}px`;
    const g = c.getContext('2d')!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const N = this.N;
    if (!N) return;
    let x0 = Infinity;
    let x1 = -Infinity;
    let y0 = Infinity;
    let y1 = -Infinity;
    for (let i = 0; i < N; i++) {
      const x = this.pts[i * 2];
      const y = this.pts[i * 2 + 1];
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
    }
    const pad = 22;
    const sx = (W - 2 * pad) / Math.max(1e-9, x1 - x0);
    const sy = (H - 2 * pad) / Math.max(1e-9, y1 - y0);
    this.screen = new Float32Array(N * 2);
    for (let i = 0; i < N; i++) {
      this.screen[i * 2] = pad + (this.pts[i * 2] - x0) * sx;
      this.screen[i * 2 + 1] = H - pad - (this.pts[i * 2 + 1] - y0) * sy;
    }
    // axes
    g.strokeStyle = 'rgba(255,255,255,0.06)';
    g.beginPath();
    g.moveTo(pad + (0 - x0) * sx, 0);
    g.lineTo(pad + (0 - x0) * sx, H);
    g.moveTo(0, H - pad - (0 - y0) * sy);
    g.lineTo(W, H - pad - (0 - y0) * sy);
    g.stroke();
    g.font = '11px "JetBrains Mono", ui-monospace, monospace';
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    if (this.mode === 'positions') {
      g.strokeStyle = 'rgba(110,139,255,0.35)';
      g.lineWidth = 1;
      g.beginPath();
      for (let i = 0; i < N; i++) (i ? g.lineTo : g.moveTo).call(g, this.screen[i * 2], this.screen[i * 2 + 1]);
      g.stroke();
      const every = Math.max(1, Math.ceil(N / 24));
      for (let i = 0; i < N; i++) {
        const t = i / Math.max(1, N - 1);
        g.fillStyle = `hsl(${Math.round(230 - t * 180)}, 80%, 62%)`;
        g.beginPath();
        g.arc(this.screen[i * 2], this.screen[i * 2 + 1], i === this.hover ? 5 : 3, 0, Math.PI * 2);
        g.fill();
        if (i % every === 0 || i === this.hover || i === N - 1) {
          g.fillStyle = '#c9cfdc';
          g.fillText(String(i), this.screen[i * 2], this.screen[i * 2 + 1] - 10);
        }
      }
      return;
    }
    const labelAll = N <= 260;
    for (let i = 0; i < N; i++) {
      const x = this.screen[i * 2];
      const y = this.screen[i * 2 + 1];
      const col = CATS[this.cats[i]]?.[1] ?? '#8b93a7';
      const hi = i === this.hover || i === this.sel;
      if (labelAll && !hi) {
        g.fillStyle = col;
        g.globalAlpha = 0.9;
        g.fillText(this.labels[i], x, y);
        g.globalAlpha = 1;
      } else if (!hi) {
        g.fillStyle = col;
        g.globalAlpha = 0.75;
        g.fillRect(x - 1.5, y - 1.5, 3, 3);
        g.globalAlpha = 1;
      }
    }
    for (const i of [this.sel, this.hover]) {
      if (i < 0) continue;
      const x = this.screen[i * 2];
      const y = this.screen[i * 2 + 1];
      g.font = 'bold 13px "JetBrains Mono", ui-monospace, monospace';
      const tw = g.measureText(this.labels[i]).width + 10;
      g.fillStyle = '#222a3c';
      g.fillRect(x - tw / 2, y - 10, tw, 20);
      g.strokeStyle = i === this.sel ? '#ffffff' : '#6e8bff';
      g.strokeRect(x - tw / 2 + 0.5, y - 9.5, tw - 1, 19);
      g.fillStyle = '#ffffff';
      g.fillText(this.labels[i], x, y + 0.5);
      g.font = '11px "JetBrains Mono", ui-monospace, monospace';
    }
  }

  private pick(e: MouseEvent): number {
    const r = this.canvas.getBoundingClientRect();
    const x = e.clientX - r.left;
    const y = e.clientY - r.top;
    let best = -1;
    let bd = 14 * 14;
    for (let i = 0; i < this.N; i++) {
      const dx = this.screen[i * 2] - x;
      const dy = this.screen[i * 2 + 1] - y;
      const dd = dx * dx + dy * dy;
      if (dd < bd) {
        bd = dd;
        best = i;
      }
    }
    return best;
  }

  private showNeighbours(): void {
    clear(this.near);
    const tr = this.studio.trained;
    const r = tr ? this.rows() : null;
    if (!tr || !r || this.sel < 0 || this.sel >= r.N) {
      this.near.append(h('div', { class: 'muted small' }, `Click a ${this.mode === 'tokens' ? 'token' : 'position'} in the map.`));
      return;
    }
    const d = tr.model.cfg.d;
    const X = r.X;
    const norm = (i: number) => {
      let s = 0;
      for (let k = 0; k < d; k++) s += X[i * d + k] * X[i * d + k];
      return Math.sqrt(s) || 1;
    };
    const ni = norm(this.sel);
    const sims: [number, number][] = [];
    for (let j = 0; j < r.N; j++) {
      if (j === this.sel) continue;
      let dot = 0;
      for (let k = 0; k < d; k++) dot += X[this.sel * d + k] * X[j * d + k];
      sims.push([j, dot / (ni * norm(j))]);
    }
    sims.sort((a, b) => b[1] - a[1]);
    this.near.append(h('div', { class: 'small dim', style: { marginBottom: '6px' } }, 'Most similar to ', h('code', null, this.labels[this.sel]), ' (cosine similarity: 1 = same direction, 0 = unrelated):'));
    for (const [j, s] of sims.slice(0, 8))
      this.near.append(h('div', { class: 'tip-row' }, h('code', null, this.labels[j]), h('span', { class: 'bar' }, h('i', { style: { width: `${Math.max(0, s * 100).toFixed(1)}%` } })), h('span', { class: 'mono small' }, fmt(s, 2))));
  }
}
