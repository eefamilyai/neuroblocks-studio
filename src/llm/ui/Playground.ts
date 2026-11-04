/**
 * The playground: generate text with the trained model. Every generated
 * token is coloured by how sure the model was; hover a token to see the
 * other candidates; the "next token" panel shows the probabilities for the
 * token after the text, and you can pick one yourself.
 */
import { RNG } from '../../ml/rng.ts';
import { btn, clear, fmt, h, section, seg, setText, toast } from '../../ui/dom.ts';
import { icon } from '../../ui/icons.ts';
import { Decoder, lineBreakTable, sampleToken, softmaxT, topK, windowStart, withStart } from '../decode.ts';
import { showToken } from '../tokenizer.ts';
import type { LLMStudio } from './LLMStudio.ts';

interface GenTok {
  id: number;
  text: string;
  /** model probability of this token (temperature 1); NaN for prompt tokens */
  p: number;
  /** the most likely candidates at this position: [id, probability] */
  top: [number, number][];
  prompt: boolean;
}

type Colour = 'prob' | 'tokens' | 'plain';

const TOKEN_COLOURS = ['rgba(110,139,255,0.22)', 'rgba(60,207,142,0.2)', 'rgba(242,184,75,0.2)', 'rgba(242,107,125,0.2)', 'rgba(155,123,255,0.22)'];

/** Confidence → colour: red (surprising) … green (sure). */
/** A probability as a short percentage ("<0.1%" for tiny ones). */
export function pct(p: number): string {
  const v = p * 100;
  if (v > 0 && v < 0.1) return '<0.1%';
  return `${v >= 99.95 && v < 100 ? '>99.9' : fmt(v, 1)}%`;
}

export function probColour(p: number, alpha = 0.32): string {
  const hue = Math.round(130 * Math.sqrt(Math.max(0, Math.min(1, p))));
  return `hsla(${hue}, 75%, 50%, ${alpha})`;
}

export class Playground {
  readonly el: HTMLElement;
  private studio: LLMStudio;
  private promptEl: HTMLTextAreaElement;
  private out: HTMLElement;
  private outWrap: HTMLElement;
  private empty: HTMLElement;
  private info: HTMLElement;
  private nextEl: HTMLElement;
  private nextHead: HTMLElement;
  private tip: HTMLElement;
  private genBtn: HTMLButtonElement;
  private tempVal: HTMLElement;
  private temp = 0.8;
  private topk = 0;
  private topp = 1;
  private len = 200;
  private colour: Colour = 'prob';
  private toks: GenTok[] = [];
  private dec: Decoder | null = null;
  private decVersion = -1;
  private generating = false;
  private stopFlag = false;
  private rng = new RNG((Date.now() & 0xffff) + 1);
  private shownVersion = -1;
  private nextDirty = true;

  constructor(studio: LLMStudio) {
    this.studio = studio;
    this.promptEl = h('textarea', { class: 'input llm-prompt', rows: '3', spellcheck: 'false', placeholder: 'Type the start of a text, then press Generate (Ctrl+Enter)' }) as HTMLTextAreaElement;
    this.promptEl.value = studio.project.sample.prompt;
    this.promptEl.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        void this.generate(false, this.len);
      }
    });
    this.promptEl.addEventListener('input', () => (this.nextDirty = !this.toks.length));
    // controls
    const temp = h('input', { type: 'range', min: '0', max: '2', step: '0.05', value: String(this.temp), title: 'Temperature: 0 = always the most likely token; 1 = the model’s own probabilities; higher = wilder' }) as HTMLInputElement;
    this.tempVal = h('b', { class: 'mono small' }, this.temp.toFixed(2));
    temp.addEventListener('input', () => {
      this.temp = Number(temp.value);
      setText(this.tempVal, this.temp.toFixed(2));
      this.nextDirty = true;
    });
    const sel = (opts: [number, string][], v: number, title: string, set: (v: number) => void) => {
      const s = h('select', { class: 'sel sm', title }) as HTMLSelectElement;
      for (const [x, l] of opts) s.append(h('option', { value: String(x) }, l));
      s.value = String(v);
      s.addEventListener('change', () => {
        set(Number(s.value));
        this.nextDirty = true;
      });
      return s;
    };
    const topk = sel([[0, 'off'], [1, '1'], [3, '3'], [5, '5'], [10, '10'], [40, '40']], 0, 'Top-k: only sample from the k most likely tokens', (v) => (this.topk = v));
    const topp = sel([[1, 'off'], [0.95, '0.95'], [0.9, '0.9'], [0.8, '0.8'], [0.5, '0.5']], 1, 'Top-p (nucleus): only sample from the most likely tokens that together have this much probability', (v) => (this.topp = v));
    const len = sel([[20, '20'], [50, '50'], [100, '100'], [200, '200'], [500, '500'], [1000, '1000']], this.len, 'How many tokens to generate', (v) => (this.len = v));
    const colour = seg<Colour>(
      [
        { value: 'prob', label: 'Confidence', title: 'Colour every generated token by how likely the model thought it was (green = sure, red = surprising)' },
        { value: 'tokens', label: 'Tokens', title: 'Show where every token starts and ends' },
        { value: 'plain', label: 'Plain', title: 'Just the text' },
      ],
      (v) => {
        this.colour = v;
        colour.set(v);
        this.redraw();
      },
      'sm',
    );
    colour.set(this.colour);
    this.genBtn = btn({ icon: 'sparkles', label: 'Generate', cls: 'primary sm', title: 'Generate text after the prompt (Ctrl+Enter)', onclick: () => void this.generate(false, this.len) });
    const oneBtn = btn({ icon: 'step', label: '+1 token', cls: 'sm', title: 'Generate ONE more token and look at the probabilities', onclick: () => void this.generate(this.toks.length > 0, 1) });
    const contBtn = btn({ icon: 'play', label: 'Continue', cls: 'ghost sm', title: 'Generate more after the current text', onclick: () => void this.generate(this.toks.length > 0, this.len) });
    const clearBtn = btn({ icon: 'eraser', cls: 'ghost sm icon', title: 'Clear the output', onclick: () => this.reset() });
    const attBtn = btn({ icon: 'eye', label: 'Attention', cls: 'ghost sm', title: 'Look at the attention for this text', onclick: () => this.studio.inspectAttention(this.text() || this.promptEl.value) });

    this.out = h('div', { class: 'llm-out' });
    this.empty = h('div', { class: 'empty-note' }, icon('sparkles'), h('div', null, h('b', null, 'No model yet.'), h('div', { class: 'dim small' }, 'Press Train (Ctrl+Enter). After a minute you can generate text here. Even a model trained for 30 seconds has learned something: try it early and again later!')));
    this.outWrap = h('div', { class: 'llm-out-wrap' }, this.empty, this.out);
    this.info = h('div', { class: 'muted small' });
    this.tip = h('div', { class: 'llm-tip hidden' });
    this.nextHead = h('span', { class: 'dim small' });
    this.nextEl = h('div', { class: 'llm-next' });
    this.el = h(
      'div',
      { class: 'llm-play' },
      section('Prompt', 'pencil', [
        this.promptEl,
        h(
          'div',
          { class: 'llm-controls' },
          h('label', { class: 'row', title: temp.title }, h('span', { class: 'lbl' }, 'Temperature'), temp, this.tempVal),
          h('label', { class: 'row' }, h('span', { class: 'lbl' }, 'Top-k'), topk),
          h('label', { class: 'row' }, h('span', { class: 'lbl' }, 'Top-p'), topp),
          h('label', { class: 'row' }, h('span', { class: 'lbl' }, 'Tokens'), len),
        ),
        h('div', { class: 'row', style: { marginTop: '8px', flexWrap: 'wrap', gap: '6px' } }, this.genBtn, oneBtn, contBtn, clearBtn, h('div', { class: 'spacer' }), attBtn, colour.el),
      ]),
      section('Output', 'chat', [this.outWrap, this.info]),
      section('Next token', 'chart', [this.nextHead, this.nextEl, h('div', { class: 'note' }, 'These are the probabilities for the token that comes next (after temperature, top-k and top-p). Click one to add it yourself: you choose instead of the dice.')]),
      this.tip,
    );
    // token hover / click (event delegation)
    this.out.addEventListener('mouseover', (e) => this.hover(e));
    this.out.addEventListener('mouseleave', () => this.tip.classList.add('hidden'));
    this.out.addEventListener('click', (e) => {
      const t = (e.target as HTMLElement).closest<HTMLElement>('.tk');
      if (!t) return;
      const i = Number(t.dataset.i);
      this.studio.inspectAttention(this.textUpTo(i + 1), -1);
    });
  }

  setPrompt(p: string): void {
    this.promptEl.value = p;
    this.reset();
  }

  /** All text: prompt + generated. */
  text(): string {
    return this.toks.map((t) => t.text).join('');
  }

  private textUpTo(n: number): string {
    return this.toks
      .slice(0, n)
      .map((t) => t.text)
      .join('');
  }

  reset(): void {
    this.stopFlag = this.generating;
    this.toks = [];
    this.dec = null;
    clear(this.out);
    setText(this.info, '');
    this.nextDirty = true;
  }

  /** Make sure the decoder holds the current tokens with the current weights. */
  private syncDecoder(): Decoder | null {
    const tr = this.studio.trained;
    if (!tr) return null;
    if (this.dec && this.decVersion === this.studio.version && this.dec.model === tr.model) return this.dec;
    const dec = this.newDecoder();
    const ids = this.toks.map((t) => t.id);
    dec.feed(ids.length ? this.window(ids) : this.startIds());
    this.dec = dec;
    this.decVersion = this.studio.version;
    return dec;
  }

  /** A decoder for the trained model (sliding to line starts for one-example-per-line data). */
  private newDecoder(): Decoder {
    const tr = this.studio.trained!;
    const dec = new Decoder(tr.model);
    dec.lineBreaks = this.studio.corpus.align ? lineBreakTable(tr.tok.vocab) : null;
    dec.lineMode = this.studio.corpus.align ?? 'start';
    return dec;
  }

  /** The newest tokens that fit in the context window (from a line start for line-by-line data). */
  private window(ids: number[]): number[] {
    const tr = this.studio.trained!;
    const mode = this.studio.corpus.align ?? 'start';
    const breaks = this.studio.corpus.align ? lineBreakTable(tr.tok.vocab) : null;
    const all = withStart(ids, breaks, tr.tok.vocab, mode);
    return all.slice(windowStart(all, Math.max(1, tr.model.cfg.ctx - 1), breaks, mode));
  }

  /** What to condition on when there is no text at all: a new line (or token 0). */
  private startIds(): number[] {
    const tok = this.studio.trained!.tok;
    const nl = tok.encode('\n');
    return nl.length ? [nl[0]] : [0];
  }

  async generate(cont: boolean, count: number): Promise<void> {
    const tr = this.studio.trained;
    if (!tr) {
      toast('Train a model first (press Train)', 2500, 'info');
      return;
    }
    if (this.generating) {
      this.stopFlag = true;
      return;
    }
    if (!cont) {
      this.reset();
      const ids = tr.tok.encode(this.promptEl.value);
      for (const id of ids) this.toks.push({ id, text: tr.tok.vocab[id], p: NaN, top: [], prompt: true });
      this.redraw();
    }
    const dec = this.syncDecoder()!;
    this.generating = true;
    this.stopFlag = false;
    this.genBtn.replaceChildren(icon('stop'), h('span', { class: 'tb-label-txt' }, 'Stop'));
    const t0 = performance.now();
    let n = 0;
    const tok = tr.tok;
    try {
      while (n < count && !this.stopFlag) {
        const ts = performance.now();
        const frag = document.createDocumentFragment();
        while (n < count && performance.now() - ts < 14) {
          if (this.dec !== dec || this.studio.trained?.model !== dec.model) {
            n = count;
            break;
          }
          const probs = softmaxT(dec.logits, 1);
          const id = sampleToken(dec.logits, this.rng, this.temp, this.topk, this.topp);
          const g: GenTok = { id, text: tok.vocab[id], p: probs[id], top: topK(probs, 8), prompt: false };
          this.toks.push(g);
          frag.append(this.span(g, this.toks.length - 1));
          dec.step(id);
          n++;
        }
        const stick = this.outWrap.scrollHeight - this.outWrap.scrollTop - this.outWrap.clientHeight < 40;
        this.out.append(frag);
        if (stick) this.outWrap.scrollTop = this.outWrap.scrollHeight;
        if (n < count && !this.stopFlag) await new Promise((r) => requestAnimationFrame(r));
      }
    } finally {
      this.generating = false;
      this.stopFlag = false;
      this.genBtn.replaceChildren(icon('sparkles'), h('span', { class: 'tb-label-txt' }, 'Generate'));
    }
    const ms = performance.now() - t0;
    const gen = this.toks.filter((t) => !t.prompt);
    const meanLogP = gen.reduce((a, t) => a + Math.log(Math.max(1e-9, t.p)), 0) / Math.max(1, gen.length);
    setText(this.info, `${gen.length} tokens generated${n ? ` · ${fmt((n * 1000) / Math.max(1, ms), 0)} tokens/s` : ''} · average confidence ${fmt(Math.exp(meanLogP) * 100, 0)}% (geometric mean) · model at step ${tr.step.toLocaleString()}`);
    this.nextDirty = true;
  }

  /** Add a token the user picked. */
  private pick(id: number): void {
    const tr = this.studio.trained;
    if (!tr || this.generating) return;
    if (!this.toks.length) {
      for (const pid of tr.tok.encode(this.promptEl.value)) this.toks.push({ id: pid, text: tr.tok.vocab[pid], p: NaN, top: [], prompt: true });
      this.redraw();
    }
    const dec = this.syncDecoder()!;
    const probs = softmaxT(dec.logits, 1);
    const g: GenTok = { id, text: tr.tok.vocab[id], p: probs[id], top: topK(probs, 8), prompt: false };
    this.toks.push(g);
    this.out.append(this.span(g, this.toks.length - 1));
    dec.step(id);
    this.nextDirty = true;
  }

  private span(t: GenTok, i: number): HTMLElement {
    const nl = t.text.includes('\n');
    const s = h('span', { class: `tk${t.prompt ? ' pr' : ''}${nl ? ' nl' : ''}`, 'data-i': String(i) }, t.text);
    if (this.colour === 'prob' && !t.prompt) s.style.background = probColour(t.p);
    else if (this.colour === 'tokens') s.style.background = TOKEN_COLOURS[i % TOKEN_COLOURS.length];
    return s;
  }

  private redraw(): void {
    clear(this.out);
    const frag = document.createDocumentFragment();
    this.toks.forEach((t, i) => frag.append(this.span(t, i)));
    this.out.append(frag);
  }

  private hover(e: MouseEvent): void {
    const el = (e.target as HTMLElement).closest<HTMLElement>('.tk');
    const tr = this.studio.trained;
    if (!el || !tr) {
      this.tip.classList.add('hidden');
      return;
    }
    const t = this.toks[Number(el.dataset.i)];
    if (!t) return;
    clear(this.tip);
    this.tip.append(h('div', { class: 'tip-head' }, h('code', null, showToken(t.text)), h('span', { class: 'muted' }, ` token #${t.id}`)));
    if (t.prompt) this.tip.append(h('div', { class: 'muted small' }, 'Part of your prompt.'));
    else {
      this.tip.append(h('div', { class: 'small' }, `The model gave it ${pct(t.p)}. Its top candidates were:`));
      for (const [id, p] of t.top) {
        if (p < 0.001 && id !== t.id) continue;
        this.tip.append(h('div', { class: `tip-row${id === t.id ? ' on' : ''}` }, h('code', null, showToken(tr.tok.vocab[id] ?? '?')), h('span', { class: 'bar' }, h('i', { style: { width: `${(p * 100).toFixed(1)}%`, background: probColour(p, 0.9) } })), h('span', { class: 'mono' }, pct(p))));
      }
    }
    this.tip.append(h('div', { class: 'muted tiny', style: { marginTop: '4px' } }, 'Click: see what this token paid attention to'));
    this.tip.classList.remove('hidden');
    const host = this.el.getBoundingClientRect();
    const r = el.getBoundingClientRect();
    const tw = this.tip.offsetWidth;
    const left = Math.max(4, Math.min(host.width - tw - 4, r.left - host.left));
    const below = r.bottom - host.top + 6;
    const th = this.tip.offsetHeight;
    this.tip.style.left = `${left}px`;
    this.tip.style.top = `${r.top - host.top - th - 6 > 0 && below + th > host.height ? r.top - host.top - th - 6 : below}px`;
  }

  render(visible: boolean): void {
    if (!visible) return;
    const tr = this.studio.trained;
    this.empty.classList.toggle('hidden', !!tr || this.toks.length > 0);
    if (this.shownVersion !== this.studio.version) {
      this.shownVersion = this.studio.version;
      this.nextDirty = true;
    }
    if (this.nextDirty && !this.generating) {
      this.nextDirty = false;
      this.renderNext();
    }
  }

  /** The probabilities for the next token after the current text (or the prompt). */
  private renderNext(): void {
    clear(this.nextEl);
    const tr = this.studio.trained;
    if (!tr) {
      setText(this.nextHead, 'Train a model to see its predictions.');
      return;
    }
    let logits: Float32Array;
    let after: string;
    if (this.toks.length) {
      const dec = this.syncDecoder()!;
      logits = dec.logits;
      after = this.text();
    } else {
      const dec = this.newDecoder();
      const ids = Array.from(tr.tok.encode(this.promptEl.value));
      logits = dec.feed(ids.length ? this.window(ids) : this.startIds());
      after = this.promptEl.value;
    }
    // what sampling would actually use: temperature, then top-k / top-p
    const raw = softmaxT(logits, 1);
    let p: Float32Array;
    if (this.temp <= 0) {
      p = new Float32Array(raw.length);
      let best = 0;
      for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
      p[best] = 1;
    } else {
      p = softmaxT(logits, this.temp);
      let keep: Set<number> | null = null;
      if (this.topk > 0) keep = new Set(topK(p, this.topk).map(([i]) => i));
      if (this.topp < 1) {
        const s = new Set<number>();
        let acc = 0;
        for (const [i, q] of topK(p, p.length)) {
          if (keep && !keep.has(i)) continue;
          s.add(i);
          acc += q;
          if (acc >= this.topp) break;
        }
        keep = s;
      }
      if (keep) {
        let tot = 0;
        for (let i = 0; i < p.length; i++) if (!keep.has(i)) p[i] = 0;
        for (let i = 0; i < p.length; i++) tot += p[i];
        for (let i = 0; i < p.length; i++) p[i] /= tot || 1;
      }
    }
    const tail = after.slice(-24);
    setText(this.nextHead, `After "…${tail.replace(/\n/g, '↵')}" the model expects:`);
    let entropy = 0;
    for (const q of raw) if (q > 0) entropy -= q * Math.log2(q);
    for (const [id, q] of topK(p, 10)) {
      if (q < 0.001 && this.nextEl.childElementCount > 0) break;
      this.nextEl.append(
        h(
          'button',
          { class: 'llm-next-row', type: 'button', title: `model probability ${pct(raw[id])} → with these settings ${pct(q)}. Click to add this token.`, onclick: () => this.pick(id) },
          h('code', null, showToken(tr.tok.vocab[id] ?? '?')),
          h('span', { class: 'bar' }, h('i', { style: { width: `${Math.max(0.5, q * 100).toFixed(1)}%` } })),
          h('span', { class: 'mono small' }, pct(q)),
        ),
      );
    }
    this.nextEl.append(h('div', { class: 'muted tiny', style: { marginTop: '4px' } }, `Uncertainty (entropy): ${fmt(entropy, 2)} bits, like choosing between ${fmt(Math.pow(2, entropy), 1)} equally likely tokens.`));
  }
}
