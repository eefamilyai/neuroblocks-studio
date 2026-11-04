/** Model: every weight tensor, its shape, statistics and a histogram; where the weights are. */
import { clear, fmt, h, section } from '../../ui/dom.ts';
import { icon } from '../../ui/icons.ts';
import { countParams, flopsPerToken } from '../gpt.ts';
import type { GPTConfig } from '../gpt.ts';
import type { LLMStudio } from './LLMStudio.ts';

const GROUPS: [string, string, (name: string) => boolean][] = [
  ['Token embedding', '#6e8bff', (n) => n === 'wte'],
  ['Positions', '#5ab4ff', (n) => n === 'wpe'],
  ['Attention', '#9b7bff', (n) => n.includes('.attn.')],
  ['MLP', '#3ccf8e', (n) => n.includes('.mlp.')],
  ['LayerNorm', '#f2b84b', (n) => n.includes('ln')],
  ['Output head', '#f26b7d', (n) => n === 'head.weight'],
];

const DESCRIBE: Record<string, string> = {
  wte: 'one vector per token',
  wpe: 'one vector per position',
  'ln1.weight': 'LayerNorm gain before attention',
  'ln1.bias': 'LayerNorm shift before attention',
  'attn.qkv.weight': 'makes queries, keys and values',
  'attn.qkv.bias': '',
  'attn.proj.weight': 'mixes the heads back together',
  'attn.proj.bias': '',
  'ln2.weight': 'LayerNorm gain before the MLP',
  'ln2.bias': 'LayerNorm shift before the MLP',
  'mlp.fc.weight': 'expands each token vector',
  'mlp.fc.bias': '',
  'mlp.proj.weight': 'projects back to the embedding size',
  'mlp.proj.bias': '',
  'lnf.weight': 'final LayerNorm gain',
  'lnf.bias': 'final LayerNorm shift',
  'head.weight': 'scores every token in the vocabulary',
};

function histogram(w: Float32Array, lo: number, hi: number, bins: number): Uint32Array {
  const out = new Uint32Array(bins);
  const s = bins / Math.max(1e-12, hi - lo);
  for (let i = 0; i < w.length; i++) {
    const b = Math.min(bins - 1, Math.max(0, Math.floor((w[i] - lo) * s)));
    out[b]++;
  }
  return out;
}

export class ModelView {
  readonly el: HTMLElement;
  private studio: LLMStudio;
  private summary: HTMLElement;
  private share: HTMLElement;
  private table: HTMLElement;
  private msg: HTMLElement;
  private key = '';
  private lastCompute = 0;

  constructor(studio: LLMStudio) {
    this.studio = studio;
    this.summary = h('div');
    this.share = h('div');
    this.table = h('div', { class: 'llm-tensors' });
    this.msg = h('div', { class: 'note', style: { marginTop: 0 } });
    this.el = h(
      'div',
      null,
      section('This model', 'layers', [this.msg, this.summary]),
      section('Where the weights are', 'chart', [this.share]),
      section('Every weight matrix', 'sigma', [this.table, h('div', { class: 'note' }, 'mean ± spread (standard deviation) of the numbers in each tensor, and their histogram. At the start the weights are small random numbers (spread 0.02); training slowly shapes them. LayerNorm gains start at exactly 1.')]),
    );
  }

  render(visible: boolean): void {
    if (!visible) return;
    const st = this.studio;
    const tr = st.trained;
    const key = `${st.version}|${tr ? 't' : JSON.stringify(st.project.model) + st.tok?.size}`;
    const now = performance.now();
    if (key === this.key) return;
    if (tr && this.key && now - this.lastCompute < 1500) return; // at most every 1.5 s while training
    this.key = key;
    this.lastCompute = now;
    const V = tr ? tr.model.cfg.vocab : (st.tok?.size ?? 64);
    const cfg: GPTConfig = tr ? tr.model.cfg : { ...st.project.model, vocab: V };
    const n = countParams(cfg);
    clear(this.summary);
    this.msg.textContent = tr ? `Trained for ${tr.step.toLocaleString()} steps (${tr.tokens.toLocaleString()} tokens of text).` : 'Not trained yet: these are the sizes the model will have. The statistics appear after training starts.';
    this.summary.append(
      h(
        'div',
        { class: 'kv' },
        h('span', null, 'Weights'),
        h('b', null, n.toLocaleString()),
        h('span', null, 'Memory (32-bit floats)'),
        h('b', null, `${fmt((n * 4) / 1024, 0)} KB`),
        h('span', null, 'Layers × heads'),
        h('b', null, `${cfg.layers} × ${cfg.heads}`),
        h('span', null, 'Embedding size'),
        h('b', null, String(cfg.d)),
        h('span', null, 'Context window'),
        h('b', null, `${cfg.ctx} tokens`),
        h('span', null, 'Vocabulary'),
        h('b', null, `${V} tokens`),
        h('span', null, 'Training compute'),
        h('b', null, `${fmt(flopsPerToken(cfg) / 1e6, 2)} MFLOP / token`),
        h('span', null, 'Random-guess loss (ln V)'),
        h('b', null, fmt(Math.log(V), 3)),
      ),
      h('div', { class: 'note' }, `For comparison: GPT-2 small has 124 million weights, about ${fmt(124e6 / n, 0)}× more. The ideas are exactly the same.`),
    );
    // share of weights
    const sizes = new Map<string, number>();
    const d = cfg.d;
    const m = cfg.mlp * d;
    sizes.set('Token embedding', V * d);
    sizes.set('Positions', cfg.pos === 'learned' ? cfg.ctx * d : 0);
    sizes.set('Attention', cfg.layers * (4 * d * d + 4 * d));
    sizes.set('MLP', cfg.layers * (2 * m * d + m + d));
    sizes.set('LayerNorm', (cfg.layers * 4 + 2) * d);
    sizes.set('Output head', cfg.tie ? 0 : V * d);
    clear(this.share);
    const bar = h('div', { class: 'llm-sharebar' });
    const legend = h('div', { class: 'llm-legend' });
    for (const [name, col] of GROUPS) {
      const s = sizes.get(name) ?? 0;
      if (!s) continue;
      bar.append(h('i', { style: { width: `${(s / n) * 100}%`, background: col }, title: `${name}: ${s.toLocaleString()} (${fmt((s / n) * 100, 1)}%)` }));
      legend.append(h('span', null, h('i', { style: { background: col } }), `${name} ${fmt((s / n) * 100, 0)}%`));
    }
    this.share.append(bar, legend);
    if (cfg.tie) this.share.append(h('div', { class: 'note' }, 'The output head is tied to the token embedding: the same matrix reads tokens in and scores them out.'));
    // tensors
    clear(this.table);
    if (!tr) {
      this.table.append(h('div', { class: 'muted small' }, 'Train the model to see its weights.'));
      return;
    }
    const head = h('div', { class: 'llm-trow head' }, h('span', null, 'tensor'), h('span', null, 'shape'), h('span', { class: 'r' }, 'weights'), h('span', { class: 'r' }, 'mean ± spread'), h('span', null, 'histogram'));
    this.table.append(head);
    let lastBlock = '';
    for (const p of tr.model.params) {
      const block = p.name.startsWith('h') ? p.name.split('.')[0] : '';
      if (block && block !== lastBlock) this.table.append(h('div', { class: 'llm-tgroup' }, icon('layers', 13), `Transformer block ${Number(block.slice(1)) + 1}`));
      if (!block && lastBlock) this.table.append(h('div', { class: 'llm-tgroup' }, icon('layers', 13), 'Output'));
      lastBlock = block || (lastBlock ? 'out' : '');
      const w = p.w;
      let s = 0;
      let lo = Infinity;
      let hi = -Infinity;
      for (let i = 0; i < w.length; i++) {
        s += w[i];
        if (w[i] < lo) lo = w[i];
        if (w[i] > hi) hi = w[i];
      }
      const mean = s / w.length;
      let v = 0;
      for (let i = 0; i < w.length; i++) v += (w[i] - mean) * (w[i] - mean);
      const sd = Math.sqrt(v / w.length);
      const short = block ? p.name.slice(block.length + 1) : p.name;
      const group = GROUPS.find(([, , f]) => f(p.name));
      const cv = h('canvas', { class: 'llm-hist', width: '120', height: '26' });
      const g = cv.getContext('2d');
      if (g) {
        // the range always includes 0 (marked), so you can see how far from zero the weights are
        const a = Math.min(lo, 0);
        const b0 = Math.max(hi, 0);
        const pad = (b0 - a) * 0.04 || 1;
        const bins = histogram(w, a - pad, b0 + pad, 40);
        let max = 1;
        for (const b of bins) if (b > max) max = b;
        g.fillStyle = group?.[1] ?? '#6e8bff';
        for (let b = 0; b < 40; b++) {
          const hh = bins[b] ? Math.max(1, (bins[b] / max) * 24) : 0;
          g.fillRect(b * 3, 26 - hh, 2.4, hh);
        }
        g.fillStyle = 'rgba(255,255,255,0.3)';
        g.fillRect(Math.round(((0 - (a - pad)) / (b0 - a + 2 * pad)) * 120), 0, 1, 26);
      }
      this.table.append(
        h(
          'div',
          { class: 'llm-trow', title: `${p.name}: ${DESCRIBE[short] ?? DESCRIBE[p.name] ?? ''}\nmin ${fmt(lo, 3)} · max ${fmt(hi, 3)}${p.decay ? '' : '\n(no weight decay on this one)'}` },
          h('span', { class: 'nm' }, h('i', { style: { background: group?.[1] ?? '#888' } }), short),
          h('span', { class: 'mono' }, p.shape.join('×')),
          h('span', { class: 'mono r' }, w.length.toLocaleString()),
          h('span', { class: 'mono r' }, `${Math.abs(mean) < 5e-4 ? '0.000' : mean.toFixed(3)} ± ${sd.toFixed(3)}`),
          cv,
        ),
      );
    }
  }
}
