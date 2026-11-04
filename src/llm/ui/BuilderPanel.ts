/** The mini-LLM builder: data, tokenizer, architecture and training settings, plus a live diagram. */
import { DATASETS } from '../datasets.ts';
import { countParams, flopsPerToken } from '../gpt.ts';
import type { GPTConfig } from '../gpt.ts';
import type { LLMProject } from '../protocol.ts';
import { btn, clear, fmt, h, section, seg, setText } from '../../ui/dom.ts';
import { icon } from '../../ui/icons.ts';

export type ChangeKind = 'data' | 'tokenizer' | 'model' | 'train' | 'sample';

type Opt<T> = [T, string];

const EXPLAIN: Record<string, string> = {
  text: 'The model never sees letters: it sees numbers. The tokenizer cuts the text into tokens (characters, words or BPE pieces) and gives each one an id.',
  emb: 'Token embedding: a table with one learned vector per token. Token id 7 → row 7. Similar tokens end up with similar vectors (see the Embeddings tab).',
  pos: 'Attention by itself does not know the ORDER of the tokens. Position embeddings add a different vector for position 0, 1, 2… "learned" = a trained table, "sinusoidal" = fixed sine waves (the original Transformer), "none" = no order information at all (try it!).',
  ln: 'LayerNorm rescales each token\'s vector to mean 0 and spread 1 (then a learned gain and bias). It keeps the numbers in a healthy range so deep stacks train well.',
  attn: 'Self-attention: every token makes a query ("what am I looking for?"), a key ("what do I contain?") and a value ("what I pass on"). It mixes the values of earlier tokens, weighted by softmax(query·key / √d). Several heads look for different things in parallel. "Causal": a token can only look back, never ahead.',
  res: 'Residual connection: the block ADDS its result to its input instead of replacing it. Information and gradients can flow straight through, which is why transformers can be stacked deep.',
  mlp: 'The MLP (feed-forward network) processes each token on its own: expand to a wider layer, apply a non-linearity (GELU), project back. Most of the weights live here; it is thought to store much of what the model "knows".',
  head: 'The output head turns the final vector into one score (logit) per token in the vocabulary. With tied weights it reuses the token embedding table (fewer weights, often better for small models).',
  softmax: 'Softmax turns the scores into probabilities that add up to 1. Training pushes up the probability of the token that really came next (cross-entropy loss). Generating = picking from these probabilities, again and again.',
};

export class BuilderPanel {
  readonly el: HTMLElement;
  private p: LLMProject;
  private onChange: (p: LLMProject, kind: ChangeKind) => void;
  private onEditText: () => void;
  private dataInfo: HTMLElement;
  private tokInfo: HTMLElement;
  private modelInfo: HTMLElement;
  private banner: HTMLElement;
  private diagram: HTMLElement;
  private explain: HTMLElement;
  private customBtn: HTMLButtonElement;
  private vocabRow: HTMLElement;
  private headSel!: HTMLSelectElement;
  private vocabCount = 0;
  private controls: { refresh: () => void }[] = [];

  constructor(p: LLMProject, onChange: (p: LLMProject, kind: ChangeKind) => void, onEditText: () => void, machine: { threads: () => number; setThreads: (n: number) => void; auto: number; max: number }) {
    this.p = p;
    this.onChange = onChange;
    this.onEditText = onEditText;
    this.dataInfo = h('div', { class: 'note' });
    this.tokInfo = h('div', { class: 'note' });
    this.modelInfo = h('div', { class: 'note' });
    this.banner = h('div', { class: 'llm-banner hidden' });
    this.diagram = h('div', { class: 'llm-diagram' });
    this.explain = h('div', { class: 'note llm-explain' }, 'Click a part of the model to see what it does.');
    this.customBtn = btn({ icon: 'pencil', label: 'Edit text', cls: 'xs', onclick: () => this.onEditText() });
    this.vocabRow = h('div');

    // ---- data
    const dataSel = this.select<string>(
      [...DATASETS.map((d) => [d.id, `${d.emoji} ${d.name}`] as Opt<string>), ['custom', '✏️ Your own text']],
      () => this.p.data.dataset,
      (v) => {
        this.p.data.dataset = v;
        this.change('data');
        if (v === 'custom' && !this.p.data.text) this.onEditText();
      },
    );
    const blurb = h('div', { class: 'note' });
    const lines = h('input', { type: 'checkbox' }) as HTMLInputElement;
    lines.addEventListener('change', () => ((this.p.data.align = lines.checked), this.change('data')));
    const linesRow = h('label', { class: 'row small dim', style: { marginTop: '8px' }, title: 'For lists (names, words, one sentence or problem per line): every training window starts at the beginning of a line, so the model always knows where an example starts.' }, lines, 'Each line is a separate example');
    const refreshBlurb = () => {
      setText(blurb, DATASETS.find((d) => d.id === this.p.data.dataset)?.blurb ?? 'Paste or load any text: a book you wrote, song lyrics, your own code… The model learns to continue it.');
      const custom = this.p.data.dataset === 'custom';
      this.customBtn.classList.toggle('hidden', !custom);
      linesRow.classList.toggle('hidden', !custom);
      lines.checked = !!this.p.data.align;
    };
    this.controls.push({ refresh: refreshBlurb });

    // ---- tokenizer
    const tokSeg = seg<LLMProject['tokenizer']['kind']>(
      [
        { value: 'char', label: 'Characters', title: 'Every character is a token' },
        { value: 'word', label: 'Words', title: 'Frequent words are tokens; rare ones fall back to characters' },
        { value: 'bpe', label: 'BPE', title: 'Byte-pair encoding: learned sub-word pieces, like GPT' },
      ],
      (v) => {
        this.p.tokenizer.kind = v;
        tokSeg.set(v);
        this.change('tokenizer');
      },
      'sm',
    );
    const vocabSel = this.select<number>(
      [
        [128, '128 tokens'],
        [256, '256 tokens'],
        [384, '384 tokens'],
        [512, '512 tokens'],
        [1024, '1024 tokens'],
        [2048, '2048 tokens'],
      ],
      () => this.p.tokenizer.vocabSize,
      (v) => ((this.p.tokenizer.vocabSize = v), this.change('tokenizer')),
    );
    this.vocabRow.append(this.field('Vocabulary size (target)', vocabSel));
    this.controls.push({
      refresh: () => {
        tokSeg.set(this.p.tokenizer.kind);
        this.vocabRow.classList.toggle('hidden', this.p.tokenizer.kind === 'char');
      },
    });

    // ---- model
    const m = () => this.p.model;
    const ctxSel = this.select<number>([[8, '8'], [16, '16'], [24, '24'], [32, '32'], [48, '48'], [64, '64'], [96, '96'], [128, '128'], [256, '256']], () => m().ctx, (v) => ((m().ctx = v), this.change('model')));
    const dSel = this.select<number>([[16, '16'], [24, '24'], [32, '32'], [48, '48'], [64, '64'], [96, '96'], [128, '128'], [192, '192'], [256, '256']], () => m().d, (v) => {
      m().d = v;
      if (v % m().heads) m().heads = [8, 4, 2, 1].find((hh) => v % hh === 0) ?? 1;
      this.change('model');
    });
    const layersSel = this.select<number>([[1, '1'], [2, '2'], [3, '3'], [4, '4'], [6, '6'], [8, '8']], () => m().layers, (v) => ((m().layers = v), this.change('model')));
    this.headSel = this.select<number>([[1, '1'], [2, '2'], [3, '3'], [4, '4'], [6, '6'], [8, '8']], () => m().heads, (v) => ((m().heads = v), this.change('model')));
    const mlpSel = this.select<number>([[1, '1×'], [2, '2×'], [4, '4×']], () => m().mlp, (v) => ((m().mlp = v), this.change('model')));
    const actSel = this.select<GPTConfig['act']>([['gelu', 'GELU'], ['relu', 'ReLU'], ['swish', 'SiLU (swish)']], () => m().act, (v) => ((m().act = v), this.change('model')));
    const posSel = this.select<GPTConfig['pos']>([['learned', 'learned'], ['sinusoidal', 'sinusoidal'], ['none', 'none']], () => m().pos, (v) => ((m().pos = v), this.change('model')));
    const tie = h('input', { type: 'checkbox' }) as HTMLInputElement;
    tie.addEventListener('change', () => ((m().tie = tie.checked), this.change('model')));
    this.controls.push({ refresh: () => (tie.checked = m().tie) });

    // ---- training
    const t = () => this.p.train;
    const lrSel = this.select<number>([[1e-4, '0.0001'], [3e-4, '0.0003'], [1e-3, '0.001'], [2e-3, '0.002'], [3e-3, '0.003'], [5e-3, '0.005'], [1e-2, '0.01'], [3e-2, '0.03']], () => t().lr, (v) => ((t().lr = v), this.change('train')));
    const batchSel = this.select<number>([[2, '2'], [4, '4'], [8, '8'], [12, '12'], [16, '16'], [32, '32']], () => t().batch, (v) => ((t().batch = v), this.change('train')));
    const stepsSel = this.select<number>([[200, '200'], [500, '500'], [800, '800'], [1500, '1,500'], [3000, '3,000'], [6000, '6,000'], [20000, '20,000'], [0, 'no limit']], () => t().steps, (v) => ((t().steps = v), this.change('train')));
    const warmSel = this.select<number>([[0, 'none'], [50, '50 steps'], [100, '100 steps'], [300, '300 steps']], () => t().warmup, (v) => ((t().warmup = v), this.change('train')));
    const schedSel = this.select<LLMProject['train']['schedule']>([['cosine', 'cosine decay'], ['constant', 'constant']], () => t().schedule, (v) => ((t().schedule = v), this.change('train')));
    const wdSel = this.select<number>([[0, 'off'], [0.01, '0.01'], [0.1, '0.1'], [0.3, '0.3']], () => t().weightDecay, (v) => ((t().weightDecay = v), this.change('train')));
    const clipSel = this.select<number>([[0, 'off'], [0.5, '0.5'], [1, '1'], [5, '5']], () => t().clip, (v) => ((t().clip = v), this.change('train')));
    const evalSel = this.select<number>([[25, '25 steps'], [50, '50 steps'], [100, '100 steps'], [250, '250 steps']], () => t().evalEvery, (v) => ((t().evalEvery = v), this.change('train')));
    const seedSel = this.select<number>([[1, '1'], [2, '2'], [3, '3'], [4, '4'], [42, '42']], () => t().seed, (v) => ((t().seed = v), this.change('model')));
    const threadOpts: Opt<number>[] = [[0, `auto (${machine.auto})`]];
    for (let n = 1; n <= machine.max; n++) threadOpts.push([n, String(n)]);
    const threadSel = this.select<number>(threadOpts, machine.threads, (v) => machine.setThreads(v));

    // ---- samples
    const prompt = h('input', { class: 'input sm', style: { width: '100%' }, title: 'Every few steps the model continues this text, so you can watch it improve' }) as HTMLInputElement;
    prompt.addEventListener('change', () => ((this.p.sample.prompt = prompt.value.replace(/\\n/g, '\n')), this.change('sample')));
    const tempSel = this.select<number>([[0, '0 (greedy)'], [0.5, '0.5'], [0.8, '0.8'], [1, '1.0'], [1.3, '1.3']], () => this.p.sample.temperature, (v) => ((this.p.sample.temperature = v), this.change('sample')));
    const everySel = this.select<number>([[25, 'every 25 steps'], [50, 'every 50 steps'], [100, 'every 100 steps'], [250, 'every 250 steps']], () => this.p.sample.every, (v) => ((this.p.sample.every = v), this.change('sample')));
    this.controls.push({ refresh: () => (prompt.value = this.p.sample.prompt.replace(/\n/g, '\\n')) });

    const grid = (...items: HTMLElement[]) => h('div', { class: 'llm-grid' }, ...items);
    this.el = h(
      'div',
      { class: 'llm-builder' },
      this.banner,
      section('Data', 'scroll', [h('div', { class: 'row' }, dataSel, this.customBtn), blurb, linesRow, this.dataInfo]),
      section('Tokenizer', 'type', [tokSeg.el, this.vocabRow, this.tokInfo]),
      section('Model', 'network', [
        grid(this.field('Context (tokens)', ctxSel, 'How many tokens the model can look back at'), this.field('Embedding size', dSel, 'Width of every token vector'), this.field('Layers', layersSel, 'Transformer blocks stacked'), this.field('Attention heads', this.headSel, 'Must divide the embedding size'), this.field('MLP width', mlpSel, 'Hidden size of the MLP, × the embedding size'), this.field('Activation', actSel), this.field('Positions', posSel, 'How the model knows the order of tokens')),
        h('label', { class: 'row small dim', style: { marginTop: '8px' } }, tie, 'Tie the output head to the token embedding'),
        this.modelInfo,
      ]),
      section('Training', 'sliders', [grid(this.field('Learning rate', lrSel), this.field('Batch (sequences)', batchSel), this.field('Steps', stepsSel), this.field('Warm-up', warmSel), this.field('Schedule', schedSel), this.field('Weight decay', wdSel), this.field('Gradient clip', clipSel), this.field('Validate every', evalSel), this.field('Seed', seedSel), this.field('CPU threads', threadSel, 'Split every batch over several CPU cores (like training on several GPUs). More threads = faster, up to the number of cores you have.'))]),
      section('Live samples', 'chat', [this.field('Prompt (\\n = new line)', prompt), grid(this.field('Temperature', tempSel), this.field('How often', everySel))]),
      section('Architecture', 'shapes', [this.diagram, this.explain]),
    );
    this.refresh();
  }

  setProject(p: LLMProject): void {
    this.p = p;
    this.refresh();
  }

  /** Re-read every control from the project (after loading a preset / file). */
  refresh(): void {
    for (const c of this.controls) c.refresh();
    this.refreshDerived();
  }

  private change(kind: ChangeKind): void {
    this.refresh();
    this.onChange(this.p, kind);
  }

  private select<T extends string | number>(opts: Opt<T>[], get: () => T, set: (v: T) => void): HTMLSelectElement {
    const el = h('select', { class: 'sel sm' }) as HTMLSelectElement;
    for (const [v, l] of opts) el.append(h('option', { value: String(v) }, l));
    const isNum = typeof opts[0][0] === 'number';
    el.addEventListener('change', () => set((isNum ? Number(el.value) : el.value) as T));
    this.controls.push({
      refresh: () => {
        const v = String(get());
        if (![...el.options].some((o) => o.value === v)) el.append(h('option', { value: v }, v));
        el.value = v;
      },
    });
    return el;
  }

  private field(label: string, ctl: HTMLElement, title = ''): HTMLElement {
    return h('label', { class: 'field', title }, h('span', { class: 'lbl' }, label), ctl);
  }

  setDataInfo(text: string): void {
    setText(this.dataInfo, text);
  }

  setTokenizerInfo(text: string, vocab: number): void {
    setText(this.tokInfo, text);
    this.vocabCount = vocab;
    this.refreshDerived();
  }

  setBanner(msg: string | null): void {
    this.banner.classList.toggle('hidden', !msg);
    if (msg) this.banner.replaceChildren(icon('info'), h('span', null, msg));
  }

  private refreshDerived(): void {
    const m = this.p.model;
    // only head counts that divide the embedding size
    for (const o of [...this.headSel.options]) o.disabled = m.d % Number(o.value) !== 0;
    const V = this.vocabCount || 64;
    const cfg: GPTConfig = { ...m, vocab: V };
    const n = countParams(cfg);
    setText(this.modelInfo, `${n.toLocaleString()} weights${this.vocabCount ? '' : ' (about)'} · ${fmt(flopsPerToken(cfg) / 1e6, 2)} MFLOP per token to train`);
    this.drawDiagram(cfg);
  }

  private drawDiagram(c: GPTConfig): void {
    const d = c.d;
    const m = c.mlp * d;
    const dh = d / c.heads;
    const tokName = this.p.tokenizer.kind === 'char' ? 'characters' : this.p.tokenizer.kind === 'word' ? 'words' : 'BPE pieces';
    const box = (key: string, title: string, sub: string, params: number | null, cls = '') =>
      h('button', { class: `llm-box ${cls}`, type: 'button', onclick: () => this.showExplain(key, title) }, h('span', { class: 't' }, title), h('span', { class: 's' }, sub), params !== null ? h('span', { class: 'n' }, params.toLocaleString()) : null);
    const arrow = () => h('div', { class: 'llm-arrow' });
    clear(this.diagram);
    const perAttn = 3 * d * d + 3 * d + d * d + d;
    const perMlp = m * d + m + d * m + d;
    this.diagram.append(
      box('text', 'Text → tokens', `${tokName} · ${this.vocabCount || '?'} different tokens`, null, 'io'),
      arrow(),
      box('emb', 'Token embedding', `${c.vocab} × ${d}`, c.vocab * d),
      c.pos !== 'none' ? box('pos', `+ Position (${c.pos})`, `${c.ctx} × ${d}`, c.pos === 'learned' ? c.ctx * d : 0) : box('pos', 'No positions', 'order is invisible!', null, 'warn'),
      arrow(),
      h(
        'div',
        { class: 'llm-stack' },
        h('div', { class: 'llm-stack-title' }, `Transformer block × ${c.layers}`),
        box('ln', 'LayerNorm', `${d}`, 2 * d, 'thin'),
        box('attn', 'Causal self-attention', `${c.heads} head${c.heads > 1 ? 's' : ''} × ${dh} dims`, perAttn),
        box('res', '⊕ add (residual)', '', null, 'thin'),
        box('ln', 'LayerNorm', `${d}`, 2 * d, 'thin'),
        box('mlp', 'MLP', `${d} → ${m} → ${d} · ${c.act.toUpperCase()}`, perMlp),
        box('res', '⊕ add (residual)', '', null, 'thin'),
      ),
      arrow(),
      box('ln', 'LayerNorm', `${d}`, 2 * d, 'thin'),
      box('head', 'Output head', c.tie ? `${d} → ${c.vocab} (shares the embedding)` : `${d} → ${c.vocab}`, c.tie ? 0 : c.vocab * d),
      arrow(),
      box('softmax', 'Softmax → next-token probabilities', `${c.vocab} probabilities`, null, 'io'),
    );
  }

  private showExplain(key: string, title: string): void {
    this.explain.replaceChildren(h('b', null, title + ': '), EXPLAIN[key] ?? '');
  }
}
