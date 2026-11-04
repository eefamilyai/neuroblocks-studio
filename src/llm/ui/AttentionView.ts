/**
 * Attention: for a piece of text, which earlier tokens every token looks at,
 * in every layer and head. Rows = the token that is "reading", columns =
 * the tokens it can look at (only earlier ones: the attention is causal).
 */
import { clear, fmt, h, section, seg, setText } from '../../ui/dom.ts';
import { icon } from '../../ui/icons.ts';
import { softmaxT, topK } from '../decode.ts';
import { showToken } from '../tokenizer.ts';
import { pct } from './Playground.ts';
import type { LLMStudio } from './LLMStudio.ts';

const FONT = '10px "JetBrains Mono", ui-monospace, monospace';

/** Attention weight → colour (dark blue … bright). */
function heat(a: number): string {
  const t = Math.pow(Math.max(0, Math.min(1, a)), 0.65);
  const r = Math.round(26 + t * (150 - 26) + Math.max(0, t - 0.7) * 300);
  const g = Math.round(32 + t * (170 - 32) + Math.max(0, t - 0.7) * 250);
  const b = Math.round(52 + t * (255 - 52));
  return `rgb(${Math.min(255, r)},${Math.min(255, g)},${b})`;
}

export class AttentionView {
  readonly el: HTMLElement;
  private studio: LLMStudio;
  private input: HTMLTextAreaElement;
  private layerHost: HTMLElement;
  private headHost: HTMLElement;
  private canvas: HTMLCanvasElement;
  private grid: HTMLElement;
  private bigWrap: HTMLElement;
  private tip: HTMLElement;
  private focus: HTMLElement;
  private pred: HTMLElement;
  private msg: HTMLElement;
  private body: HTMLElement;
  private layer = 0;
  /** −1 = all heads (small multiples) */
  private head = -1;
  private row = -1;
  private ids: number[] = [];
  private labels: string[] = [];
  private atts: Float32Array[] = [];
  private logits = new Float32Array(0);
  private V = 0;
  private H = 0;
  private key = '';
  private lastCompute = 0;
  private userText = false;
  private geo = { L: 0, T: 0, cell: 0, n: 0 };
  private drawn = '';

  constructor(studio: LLMStudio) {
    this.studio = studio;
    this.input = h('textarea', { class: 'input llm-prompt', rows: '2', spellcheck: 'false', placeholder: 'Text to look at' }) as HTMLTextAreaElement;
    this.input.addEventListener('input', () => {
      this.userText = true;
      this.row = -1;
    });
    this.layerHost = h('div', { class: 'row', style: { gap: '6px' } });
    this.headHost = h('div', { class: 'row', style: { gap: '6px' } });
    this.canvas = h('canvas', { class: 'llm-heat' });
    this.tip = h('div', { class: 'chart-tip', style: { display: 'none' } });
    this.bigWrap = h('div', { class: 'llm-heat-wrap' }, this.canvas, this.tip);
    this.grid = h('div', { class: 'llm-heads' });
    this.focus = h('div', { class: 'llm-focus' });
    this.pred = h('div', { class: 'llm-pred' });
    this.msg = h('div', { class: 'empty-note' }, icon('eye'), h('div', null, h('b', null, 'No model yet.'), h('div', { class: 'dim small' }, 'Train a model, then come back: you will see which earlier tokens every token pays attention to. In the Copy machine preset, heads learn to point exactly at the letter to copy.')));
    this.body = h(
      'div',
      null,
      section('Text', 'type', [this.input, h('div', { class: 'llm-att-controls' }, h('span', { class: 'lbl' }, 'Layer'), this.layerHost, h('span', { class: 'lbl' }, 'Head'), this.headHost)]),
      section('Attention pattern', 'eye', [this.grid, this.bigWrap, h('div', { class: 'note' }, 'Each row is one token deciding what to read. The brighter a square, the more of that earlier token’s information it takes in. Everything above the diagonal is black: a token can never look at the future. Click a row to follow it below.')]),
      section('Follow one token', 'crosshair', [this.focus, this.pred]),
    );
    this.el = h('div', null, this.msg, this.body);
    this.canvas.addEventListener('mousemove', (e) => this.hover(e));
    this.canvas.addEventListener('mouseleave', () => (this.tip.style.display = 'none'));
    this.canvas.addEventListener('click', (e) => {
      const c = this.cellAt(e);
      if (c) {
        this.row = c[0];
        this.drawn = '';
      }
    });
    new ResizeObserver(() => (this.drawn = '')).observe(this.body);
  }

  /** Look at this text (from the playground). row −1 = the last token. */
  setText(text: string, row = -1): void {
    this.input.value = text;
    this.userText = true;
    this.row = row;
    this.key = '';
  }

  private defaultText(): string {
    const st = this.studio;
    const s = st.trainingView.samples;
    if (s.length) return s[s.length - 1].text;
    const c = st.corpus;
    const v = c.valStart ?? Math.floor(c.text.length * 0.9);
    return c.text.slice(v, v + 200);
  }

  render(visible: boolean): void {
    if (!visible) return;
    const tr = this.studio.trained;
    this.msg.classList.toggle('hidden', !!tr);
    this.body.classList.toggle('hidden', !tr);
    if (!tr) return;
    if (!this.userText && !this.input.value) this.input.value = this.defaultText();
    const now = performance.now();
    const key = `${this.studio.version}|${this.input.value}|${tr.model.cfg.ctx}`;
    if (key !== this.key && (now - this.lastCompute > 900 || !this.key.endsWith(this.input.value + '|' + tr.model.cfg.ctx))) {
      this.key = key;
      this.lastCompute = now;
      this.compute();
    }
    const want = `${this.key}|${this.layer}|${this.head}|${this.row}|${this.body.clientWidth}`;
    if (want !== this.drawn) {
      this.drawn = want;
      this.draw();
    }
  }

  private compute(): void {
    const tr = this.studio.trained!;
    const { ctx, layers, heads, vocab } = tr.model.cfg;
    let ids = Array.from(tr.tok.encode(this.input.value));
    if (ids.length > ctx) ids = ids.slice(-ctx);
    this.ids = ids;
    this.labels = ids.map((i) => showToken(tr.tok.vocab[i] ?? '?'));
    this.H = heads;
    this.V = vocab;
    this.atts = [];
    if (!ids.length) return;
    const logits = tr.model.forward(ids, 1, ids.length);
    this.logits = logits.slice(0, ids.length * vocab);
    for (let l = 0; l < layers; l++) this.atts.push(tr.model.attention(l).slice());
    if (this.layer >= layers) this.layer = 0;
    if (this.head >= heads) this.head = -1;
    // layer & head pickers
    clear(this.layerHost);
    const ls = seg<string>(
      Array.from({ length: layers }, (_, l) => ({ value: String(l), label: String(l + 1), title: `Layer ${l + 1}` })),
      (v) => {
        this.layer = Number(v);
        ls.set(v);
      },
      'sm',
    );
    ls.set(String(this.layer));
    this.layerHost.append(ls.el);
    clear(this.headHost);
    const hs = seg<string>(
      [{ value: '-1', label: 'All', title: 'All heads side by side' }, ...Array.from({ length: heads }, (_, i) => ({ value: String(i), label: String(i + 1), title: `Head ${i + 1}` }))],
      (v) => {
        this.head = Number(v);
        hs.set(v);
      },
      'sm',
    );
    hs.set(String(this.head));
    this.headHost.append(hs.el);
  }

  /** attention of head hh, query i, key j in the current layer */
  private a(hh: number, i: number, j: number): number {
    const T = this.ids.length;
    return this.atts[this.layer][(hh * T + i) * T + j];
  }

  private draw(): void {
    const T = this.ids.length;
    if (!T || !this.atts.length) {
      clear(this.grid);
      clear(this.focus);
      clear(this.pred);
      this.canvas.style.display = 'none';
      if (!T) this.focus.append(h('div', { class: 'muted small' }, 'Type some text above (characters the model knows).'));
      return;
    }
    const row = this.row < 0 || this.row >= T ? T - 1 : this.row;
    this.grid.classList.toggle('hidden', this.head >= 0);
    this.bigWrap.classList.toggle('hidden', this.head < 0);
    if (this.head < 0) this.drawGrid(row);
    else this.drawBig(row);
    this.drawFocus(row);
  }

  private drawGrid(row: number): void {
    clear(this.grid);
    const T = this.ids.length;
    // the grid is a CSS grid with square cells; draw at the size they will have
    const cols = this.H <= 2 ? this.H : this.H <= 4 ? 2 : this.H <= 6 ? 3 : 4;
    this.grid.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;
    const W = (this.grid.clientWidth || 360) - 16;
    const size = Math.max(60, Math.floor((W - (cols - 1) * 8) / cols));
    for (let hh = 0; hh < this.H; hh++) {
      const c = h('canvas', { class: 'llm-head-canvas', title: `Head ${hh + 1}: click to enlarge` });
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      c.width = Math.round(size * dpr);
      c.height = Math.round(size * dpr);
      const g = c.getContext('2d')!;
      g.scale(dpr, dpr);
      const cell = size / T;
      g.fillStyle = '#0d1118';
      g.fillRect(0, 0, size, size);
      for (let i = 0; i < T; i++)
        for (let j = 0; j <= i; j++) {
          g.fillStyle = heat(this.a(hh, i, j));
          g.fillRect(j * cell, i * cell, Math.ceil(cell), Math.ceil(cell));
        }
      g.strokeStyle = 'rgba(255,255,255,0.7)';
      g.lineWidth = 1;
      g.strokeRect(0.5, row * cell + 0.5, size - 1, Math.max(1, cell - 1));
      c.addEventListener('click', () => {
        this.head = hh;
        this.key = '';
      });
      this.grid.append(h('div', { class: 'llm-head-cell' }, h('div', { class: 'tiny muted' }, `head ${hh + 1}`), c));
    }
  }

  private drawBig(row: number): void {
    const T = this.ids.length;
    const c = this.canvas;
    c.style.display = 'block';
    const W = Math.max(200, this.bigWrap.clientWidth);
    const g0 = c.getContext('2d')!;
    g0.font = FONT;
    let lw = 0;
    for (const s of this.labels) lw = Math.max(lw, g0.measureText(s.slice(0, 8)).width);
    const L = Math.ceil(lw) + 8;
    const cell = Math.max(3, Math.min(22, Math.floor((W - L - 4) / T)));
    const showCols = cell >= 9;
    const top = showCols ? Math.ceil(lw) + 10 : 6;
    const Hh = top + cell * T + 4;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    c.width = Math.round(W * dpr);
    c.height = Math.round(Hh * dpr);
    c.style.width = `${W}px`;
    c.style.height = `${Hh}px`;
    const g = c.getContext('2d')!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.font = FONT;
    g.clearRect(0, 0, W, Hh);
    this.geo = { L, T: top, cell, n: T };
    // labels
    g.fillStyle = '#8b93a7';
    g.textAlign = 'right';
    g.textBaseline = 'middle';
    const every = cell >= 9 ? 1 : Math.ceil(9 / cell);
    for (let i = 0; i < T; i++) if (i % every === 0 || i === row) g.fillText(this.labels[i].slice(0, 8), L - 5, top + i * cell + cell / 2);
    if (showCols) {
      g.save();
      g.textAlign = 'left';
      for (let j = 0; j < T; j++) {
        g.save();
        g.translate(L + j * cell + cell / 2, top - 4);
        g.rotate(-Math.PI / 2);
        g.fillText(this.labels[j].slice(0, 8), 0, 0);
        g.restore();
      }
      g.restore();
    }
    // cells
    g.fillStyle = '#0d1118';
    g.fillRect(L, top, cell * T, cell * T);
    const gap = cell >= 8 ? 1 : 0;
    for (let i = 0; i < T; i++)
      for (let j = 0; j <= i; j++) {
        g.fillStyle = heat(this.a(this.head, i, j));
        g.fillRect(L + j * cell, top + i * cell, cell - gap, cell - gap);
      }
    g.strokeStyle = '#ffffff';
    g.lineWidth = 1.5;
    g.strokeRect(L - 0.5, top + row * cell - 0.5, cell * T, cell);
  }

  private cellAt(e: MouseEvent): [number, number] | null {
    const r = this.canvas.getBoundingClientRect();
    const { L, T, cell, n } = this.geo;
    const x = e.clientX - r.left - L;
    const y = e.clientY - r.top - T;
    const j = Math.floor(x / cell);
    const i = Math.floor(y / cell);
    if (i < 0 || i >= n || j < 0 || j >= n) return null;
    return [i, j];
  }

  private hover(e: MouseEvent): void {
    const c = this.cellAt(e);
    if (!c || this.head < 0) {
      this.tip.style.display = 'none';
      return;
    }
    const [i, j] = c;
    this.tip.innerHTML = '';
    if (j > i) this.tip.append(`"${this.labels[i]}" cannot look ahead at "${this.labels[j]}"`);
    else this.tip.append(`"${this.labels[i]}" (token ${i + 1}) looks at "${this.labels[j]}" (token ${j + 1}): ${fmt(this.a(this.head, i, j) * 100, 1)}%`);
    this.tip.style.display = 'block';
    const r = this.bigWrap.getBoundingClientRect();
    const x = e.clientX - r.left;
    const tw = this.tip.offsetWidth;
    this.tip.style.left = `${x + 14 + tw > r.width ? x - tw - 10 : x + 14}px`;
    this.tip.style.top = `${e.clientY - r.top + 12}px`;
  }

  private drawFocus(row: number): void {
    const T = this.ids.length;
    clear(this.focus);
    const tr = this.studio.trained!;
    const heads = this.head >= 0 ? [this.head] : Array.from({ length: this.H }, (_, i) => i);
    const w = new Float32Array(T);
    for (const hh of heads) for (let j = 0; j <= row; j++) w[j] += this.a(hh, row, j) / heads.length;
    this.focus.append(h('div', { class: 'small dim', style: { marginBottom: '6px' } }, `Token ${row + 1} `, h('code', null, this.labels[row]), ` (layer ${this.layer + 1}, ${this.head >= 0 ? `head ${this.head + 1}` : 'average of all heads'}) reads:`));
    const txt = h('div', { class: 'llm-out small-out' });
    for (let j = 0; j < T; j++) {
      const piece = tr.tok.vocab[this.ids[j]] ?? '?';
      const s = h('span', { class: `tk${j === row ? ' focus' : ''}${j > row ? ' future' : ''}${piece.includes('\n') ? ' nl' : ''}`, title: j <= row ? `${fmt(w[j] * 100, 1)}%` : 'in the future: invisible', 'data-j': String(j) }, piece);
      if (j <= row) s.style.background = `rgba(110,139,255,${Math.min(0.9, Math.pow(w[j], 0.7)).toFixed(3)})`;
      s.addEventListener('click', () => {
        this.row = j;
        this.drawn = '';
      });
      txt.append(s);
    }
    this.focus.append(txt);
    // what the model predicts after this token
    clear(this.pred);
    const V = this.V;
    const probs = softmaxT(this.logits.subarray(row * V, (row + 1) * V), 1);
    const actual = row + 1 < T ? this.ids[row + 1] : -1;
    const rows = topK(probs, 5);
    this.pred.append(h('div', { class: 'small dim', style: { margin: '10px 0 4px' } }, 'After it, the model predicts (all layers together):'));
    for (const [id, p] of rows) {
      if (p < 0.001 && this.pred.childElementCount > 1) break;
      this.pred.append(h('div', { class: `tip-row${id === actual ? ' on' : ''}` }, h('code', null, showToken(tr.tok.vocab[id] ?? '?')), h('span', { class: 'bar' }, h('i', { style: { width: `${(p * 100).toFixed(1)}%` } })), h('span', { class: 'mono small' }, pct(p))));
    }
    if (actual >= 0) {
      const p = probs[actual];
      this.pred.append(h('div', { class: 'tiny muted', style: { marginTop: '4px' } }, `Really next: `, h('code', null, showToken(tr.tok.vocab[actual] ?? '?')), ` (${pct(p)}, loss ${fmt(-Math.log(Math.max(1e-9, p)), 2)})`));
    }
    setText(this.tip, '');
  }
}
