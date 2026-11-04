/** Training: loss curves, KPIs, live samples and events. */
import { LineChart } from '../../ui/charts.ts';
import { btn, clear, fmt, fmtSteps, h, section, setText } from '../../ui/dom.ts';
import { icon } from '../../ui/icons.ts';
import type { FromLLM } from '../protocol.ts';
import type { SavedHistory } from '../serialize.ts';
import { thin } from '../serialize.ts';
import type { LLMLogLevel, LLMStudio } from './LLMStudio.ts';

type Progress = Extract<FromLLM, { type: 'progress' }>;

const MAX_SAMPLES = 40;

export class TrainingView {
  readonly el: HTMLElement;
  private studio: LLMStudio;
  // history (plain arrays: the charts smooth them incrementally)
  private lx: number[] = [];
  private ly: number[] = [];
  private vx: number[] = [];
  private vy: number[] = [];
  private rx: number[] = [];
  private lr: number[] = [];
  private gn: number[] = [];
  private bx: number[] = [0, 1];
  private by: number[] = [NaN, NaN];
  private ax: number[] = [];
  private ay: number[] = [];
  private probe: { name: string; help: string } | null = null;
  samples: { step: number; text: string }[] = [];
  private lastTokens = 0;
  private tps = 0;
  private threads = 1;
  private kpi: Record<'step' | 'loss' | 'val' | 'ppl' | 'tokens' | 'tps' | 'lr' | 'gn' | 'acc', HTMLElement>;
  private gnBox: HTMLElement;
  private accBox: HTMLElement;
  private accLabel: HTMLElement;
  private accChart = new LineChart('Held-out accuracy (%)', '', { yRange: [0, 100] });
  private lossChart = new LineChart('Loss', 'Cross-entropy: −log(probability the model gave to the real next token), averaged. Dots: every training batch. Line: smoothed. Green: validation text it never trains on. Dashed: what random guessing would score (log of the vocabulary size).', { tall: true });
  private lrChart = new LineChart('Learning rate', 'How big the weight updates are: warm-up first, then (cosine) decay.');
  private gnChart = new LineChart('Gradient norm', 'The size of the gradient before clipping. Spikes = surprising batches; clipping caps the update size.');
  private samplesEl: HTMLElement;
  private logEl: HTMLElement;
  private empty: HTMLElement;
  private dirty = true;
  private samplesDirty = true;

  constructor(studio: LLMStudio) {
    this.studio = studio;
    const k = (label: string, title: string) => {
      const b = h('b', null, '–');
      return [b, h('div', { class: 'kpi', title }, h('small', null, label), b)] as const;
    };
    const [step, eStep] = k('Step', 'Training steps (batches) done');
    const [loss, eLoss] = k('Train loss', 'Smoothed loss on the training batches');
    const [val, eVal] = k('Val loss', 'Loss on the validation text (never trained on). If it goes up while the train loss goes down, the model is memorising (overfitting).');
    const [ppl, ePpl] = k('Perplexity', 'e^(val loss): the model is as unsure as if it picked between this many tokens. 1 = perfect.');
    const [tokens, eTokens] = k('Tokens seen', 'Tokens of text processed so far');
    const [tps, eTps] = k('Tokens/s', 'Training speed');
    const [lr, eLr] = k('Learning rate', 'Current learning rate');
    const [gn, eGn] = k('Grad norm', 'Size of the last gradient');
    const [acc, eAcc] = k('Accuracy', '');
    this.kpi = { step, loss, val, ppl, tokens, tps, lr, gn, acc };
    this.gnBox = eGn;
    this.accBox = eAcc;
    this.accLabel = eAcc.querySelector('small')!;
    this.accBox.classList.add('hidden');
    this.accChart.el.classList.add('hidden');
    this.accChart.el.style.marginTop = '8px';
    this.accChart.setSeries([{ name: '', color: '#3ccf8e', xs: this.ax, ys: this.ay, raw: true, width: 2, fill: true }]);
    this.lossChart.setSeries(this.lossSeries());
    this.lrChart.setSeries([{ name: '', color: '#9b7bff', xs: this.rx, ys: this.lr, raw: true, fill: true }]);
    this.gnChart.setSeries([{ name: '', color: '#f2b84b', xs: this.rx, ys: this.gn, raw: true }]);
    this.samplesEl = h('div', { class: 'llm-samples' });
    this.logEl = h('div', { class: 'llm-log' });
    this.empty = h('div', { class: 'empty-note' }, icon('chart'), h('div', null, h('b', null, 'Nothing trained yet.'), h('div', { class: 'dim small' }, 'Press Train (Ctrl+Enter). The loss starts near "random guessing" and should drop within seconds. Samples of generated text appear below as the model learns.')));
    this.el = h(
      'div',
      null,
      this.empty,
      h('div', { class: 'kpis' }, eStep, eLoss, eVal, ePpl),
      h('div', { class: 'kpis' }, eTokens, eTps, eLr, eGn, eAcc),
      section('Learning curves', 'chart', [this.lossChart.el, this.accChart.el, h('div', { class: 'grid2', style: { marginTop: '8px' } }, this.lrChart.el, this.gnChart.el)]),
      section('Live samples', 'chat', [h('div', { class: 'note', style: { marginTop: 0, marginBottom: '8px' } }, 'Every few steps the model continues the prompt (set it in the builder). Watch it go from random letters to words to sentences.'), this.samplesEl]),
      section('Events', 'terminal', [this.logEl], [btn({ icon: 'trash', cls: 'ghost xs icon', title: 'Clear', onclick: () => clear(this.logEl) })]),
    );
  }

  private lossSeries() {
    return [
      { name: 'random guess', color: '#6b7489', xs: this.bx, ys: this.by, raw: true, dashed: true, width: 1.2 },
      { name: 'train', color: '#6e8bff', xs: this.lx, ys: this.ly, dots: true },
      { name: 'validation', color: '#3ccf8e', xs: this.vx, ys: this.vy, raw: true, width: 2 },
    ];
  }

  get lastLoss(): number {
    // a short mean (the raw per-batch loss is noisy)
    const n = this.ly.length;
    if (!n) return NaN;
    let s = 0;
    const k = Math.min(n, 20);
    for (let i = n - k; i < n; i++) s += this.ly[i];
    return s / k;
  }

  get lastVal(): number {
    return this.vy.length ? this.vy[this.vy.length - 1] : NaN;
  }

  get hasData(): boolean {
    return this.lx.length > 0;
  }

  /** The dataset has test cases with right answers (or not). */
  setProbe(p: { name: string; help: string } | null): void {
    this.probe = p;
    this.accBox.classList.toggle('hidden', !p);
    this.gnBox.classList.toggle('hidden', !!p);
    this.accChart.el.classList.toggle('hidden', !p);
    if (p) {
      this.accLabel.textContent = p.name;
      this.accBox.title = p.help;
      this.accChart.el.querySelector('.chart-title span')!.textContent = `Held-out accuracy: ${p.name} (%)`;
      this.accChart.el.title = p.help;
    }
    this.dirty = true;
  }

  setBaseline(v: number): void {
    this.by[0] = this.by[1] = v;
    this.dirty = true;
  }

  addProgress(m: Progress): void {
    for (const [s, l] of m.losses) {
      this.lx.push(s);
      this.ly.push(l);
    }
    for (const [s, v] of m.vals) {
      // replace a value for the same step (e.g. after a single step)
      if (this.vx.length && this.vx[this.vx.length - 1] === s) this.vy[this.vy.length - 1] = v;
      else {
        this.vx.push(s);
        this.vy.push(v);
      }
    }
    for (const [s, a] of m.accs) {
      if (this.ax.length && this.ax[this.ax.length - 1] === s) this.ay[this.ay.length - 1] = a * 100;
      else {
        this.ax.push(s);
        this.ay.push(a * 100);
      }
    }
    if (m.losses.length) {
      this.rx.push(m.step);
      this.lr.push(m.lr);
      this.gn.push(m.gradNorm);
    }
    this.lastTokens = m.tokens;
    this.tps = m.tokPerSec;
    this.threads = m.threads;
    this.dirty = true;
  }

  addSample(step: number, text: string): void {
    this.samples.push({ step, text });
    if (this.samples.length > MAX_SAMPLES) this.samples.shift();
    this.samplesDirty = true;
  }

  log(msg: string, level: LLMLogLevel = 'info'): void {
    const t = new Date();
    const ts = `${String(t.getHours()).padStart(2, '0')}:${String(t.getMinutes()).padStart(2, '0')}`;
    this.logEl.prepend(h('div', { class: `ln ${level}` }, h('span', { class: 't' }, ts), h('span', null, msg)));
    while (this.logEl.childElementCount > 60) this.logEl.lastElementChild!.remove();
  }

  reset(): void {
    this.lx = [];
    this.ly = [];
    this.vx = [];
    this.vy = [];
    this.rx = [];
    this.lr = [];
    this.gn = [];
    this.bx = [0, 1];
    this.by = [NaN, NaN];
    this.ax = [];
    this.ay = [];
    this.samples = [];
    this.lastTokens = 0;
    this.tps = 0;
    this.lossChart.setSeries(this.lossSeries());
    this.accChart.setSeries([{ name: '', color: '#3ccf8e', xs: this.ax, ys: this.ay, raw: true, width: 2, fill: true }]);
    this.lrChart.setSeries([{ name: '', color: '#9b7bff', xs: this.rx, ys: this.lr, raw: true, fill: true }]);
    this.gnChart.setSeries([{ name: '', color: '#f2b84b', xs: this.rx, ys: this.gn, raw: true }]);
    this.dirty = true;
    this.samplesDirty = true;
  }

  restore(hist: SavedHistory | undefined, samples: { step: number; text: string }[] | undefined): void {
    this.reset();
    if (hist) {
      for (const [s, l] of hist.loss ?? []) {
        this.lx.push(s);
        this.ly.push(l);
      }
      for (const [s, v] of hist.val ?? []) {
        this.vx.push(s);
        this.vy.push(v);
      }
      for (const [s, a] of hist.acc ?? []) {
        this.ax.push(s);
        this.ay.push(a);
      }
    }
    if (samples) this.samples = samples.slice(-MAX_SAMPLES);
    const tr = this.studio.trained;
    if (tr) this.setBaseline(Math.log(tr.tok.size));
  }

  history(maxPoints: number): SavedHistory {
    return { loss: thin(this.lx, this.ly, maxPoints), val: thin(this.vx, this.vy, maxPoints), acc: thin(this.ax, this.ay, maxPoints) };
  }

  csv(): string {
    const val = new Map<number, number>();
    this.vx.forEach((s, i) => val.set(s, this.vy[i]));
    const acc = new Map<number, number>();
    this.ax.forEach((s, i) => acc.set(s, this.ay[i]));
    const rows = [`step,train_loss,val_loss${this.probe ? ',accuracy_percent' : ''}`];
    for (let i = 0; i < this.lx.length; i++) {
      const v = val.get(this.lx[i]);
      const a = acc.get(this.lx[i]);
      rows.push(`${this.lx[i]},${this.ly[i].toFixed(5)},${v !== undefined ? v.toFixed(5) : ''}${this.probe ? `,${a !== undefined ? a.toFixed(1) : ''}` : ''}`);
    }
    return rows.join('\n');
  }

  render(visible: boolean): void {
    if (!visible) return;
    const tr = this.studio.trained;
    if (this.dirty) {
      this.dirty = false;
      const last = this.lx.length ? this.lx[this.lx.length - 1] : 1;
      this.bx[0] = this.lx.length ? this.lx[0] : 0;
      this.bx[1] = Math.max(last, this.bx[0] + 1);
      this.lossChart.invalidate();
      this.accChart.invalidate();
      this.lrChart.invalidate();
      this.gnChart.invalidate();
      const val = this.lastVal;
      setText(this.kpi.step, tr ? fmtSteps(tr.step) : '–');
      setText(this.kpi.loss, fmt(this.lastLoss, 3));
      setText(this.kpi.val, fmt(val, 3));
      setText(this.kpi.ppl, Number.isFinite(val) ? fmt(Math.exp(val), 2) : '–');
      setText(this.kpi.tokens, this.lastTokens || tr ? fmtSteps(this.lastTokens || tr!.tokens) : '–');
      setText(this.kpi.tps, this.tps > 0 && this.studio.state === 'running' ? fmtSteps(this.tps) : '–');
      this.kpi.tps.parentElement!.title = `Training speed: tokens of text per second, on ${this.threads} CPU thread${this.threads > 1 ? 's' : ''}`;
      setText(this.kpi.tps.previousElementSibling!, this.threads > 1 ? `Tok/s · ${this.threads} thr` : 'Tokens/s');
      setText(this.kpi.lr, this.lr.length ? this.lr[this.lr.length - 1].toExponential(1) : '–');
      setText(this.kpi.gn, this.gn.length ? fmt(this.gn[this.gn.length - 1], 2) : '–');
      setText(this.kpi.acc, this.ay.length ? `${fmt(this.ay[this.ay.length - 1], 0)}%` : '–');
      this.empty.classList.toggle('hidden', this.lx.length > 0 || !!tr);
    }
    this.lossChart.draw();
    this.accChart.draw();
    this.lrChart.draw();
    this.gnChart.draw();
    if (this.samplesDirty) {
      this.samplesDirty = false;
      const prompt = this.studio.project.sample.prompt;
      clear(this.samplesEl);
      if (!this.samples.length) this.samplesEl.append(h('div', { class: 'muted small' }, 'No samples yet.'));
      for (let i = this.samples.length - 1; i >= 0; i--) {
        const s = this.samples[i];
        const body = h('div', { class: 'txt' });
        if (prompt && s.text.startsWith(prompt)) body.append(h('span', { class: 'pr' }, prompt), s.text.slice(prompt.length));
        else body.append(s.text);
        this.samplesEl.append(h('div', { class: 'llm-sample' + (i === this.samples.length - 1 ? ' newest' : '') }, h('span', { class: 'badge' }, `step ${fmtSteps(s.step)}`), body));
      }
    }
  }
}
