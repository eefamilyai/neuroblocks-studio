/** Tokens (how text becomes numbers, the vocabulary, BPE merges) and Data (the training text). */
import { btn, clear, fmt, h, section, setText } from '../../ui/dom.ts';
import { icon } from '../../ui/icons.ts';
import { getDataset } from '../datasets.ts';
import { showToken } from '../tokenizer.ts';
import type { Tokenizer } from '../tokenizer.ts';
import type { LLMStudio } from './LLMStudio.ts';

const CHIP_COLOURS = ['#6e8bff', '#3ccf8e', '#f2b84b', '#f26b7d', '#9b7bff', '#5ab4ff'];

export class TokensView {
  readonly el: HTMLElement;
  private studio: LLMStudio;
  private input: HTMLTextAreaElement;
  private chips: HTMLElement;
  private stats: HTMLElement;
  private head: HTMLElement;
  private vocab: HTMLElement;
  private filter: HTMLInputElement;
  private merges: HTMLElement;
  private mergesSec: HTMLElement;
  private dirty = true;
  private shownTok: Tokenizer | null = null;
  private userEdited = false;

  constructor(studio: LLMStudio) {
    this.studio = studio;
    this.input = h('textarea', { class: 'input llm-prompt', rows: '3', spellcheck: 'false', placeholder: 'Type anything to see how the tokenizer cuts it into tokens' }) as HTMLTextAreaElement;
    this.input.addEventListener('input', () => {
      this.userEdited = true;
      this.renderChips();
    });
    this.chips = h('div', { class: 'llm-chips' });
    this.stats = h('div', { class: 'muted small', style: { marginTop: '6px' } });
    this.head = h('div', { class: 'note', style: { marginTop: 0 } });
    this.filter = h('input', { class: 'input sm', placeholder: 'Find a token…', style: { width: '160px' } }) as HTMLInputElement;
    this.filter.addEventListener('input', () => this.renderVocab());
    this.vocab = h('div', { class: 'llm-vocab' });
    this.merges = h('ol', { class: 'llm-merges' });
    this.mergesSec = section('Merges, in the order they were learned', 'puzzle', [h('div', { class: 'note', style: { marginTop: 0, marginBottom: '8px' } }, 'Byte-pair encoding starts with single characters and repeatedly glues together the pair that appears most often in the text. Every merge makes a new token. The first merges are the most common pairs.'), this.merges]);
    this.el = h(
      'div',
      null,
      section('Text → tokens', 'type', [this.head, this.input, this.chips, this.stats]),
      section('Vocabulary', 'book', [this.vocab], [this.filter]),
      this.mergesSec,
    );
  }

  invalidate(): void {
    this.dirty = true;
  }

  render(visible: boolean): void {
    if (!visible) return;
    const tok = this.studio.tok;
    if (!this.dirty && tok === this.shownTok) return;
    this.dirty = false;
    this.shownTok = tok;
    if (!this.userEdited) {
      const text = this.studio.corpus.text;
      const cut = text.lastIndexOf(' ', 150);
      this.input.value = text.slice(0, cut > 60 ? cut : 150);
    }
    if (!tok) {
      setText(this.head, 'Making the tokenizer…');
      clear(this.chips);
      clear(this.vocab);
      this.mergesSec.classList.add('hidden');
      return;
    }
    const kind = tok.kind === 'char' ? 'a character tokenizer: every character is one token' : tok.kind === 'word' ? 'a word tokenizer: common words are single tokens, rare words are spelled out letter by letter' : 'a BPE (byte-pair encoding) tokenizer, the kind GPT uses: common pieces of words become single tokens';
    setText(this.head, `The model reads ${kind}. Every token has an id: the model only ever sees these numbers. ${this.studio.modelMatches ? '' : '(Preview: the model is trained with these settings when you press Train.)'}`);
    this.renderChips();
    this.renderVocab();
    this.mergesSec.classList.toggle('hidden', tok.kind !== 'bpe');
    if (tok.kind === 'bpe') {
      clear(this.merges);
      const show = Math.min(tok.merges.length, 300);
      for (let i = 0; i < show; i++) {
        const [a, b] = tok.merges[i];
        this.merges.append(h('li', null, h('code', null, showToken(a)), ' + ', h('code', null, showToken(b)), ' → ', h('code', { class: 'strong' }, showToken(a + b))));
      }
      if (tok.merges.length > show) this.merges.append(h('li', { class: 'muted' }, `… and ${tok.merges.length - show} more`));
    }
  }

  private renderChips(): void {
    const tok = this.shownTok;
    clear(this.chips);
    if (!tok) return;
    const text = this.input.value;
    const ids = tok.encode(text);
    const frag = document.createDocumentFragment();
    const show = Math.min(ids.length, 600);
    for (let i = 0; i < show; i++) {
      const piece = tok.vocab[ids[i]];
      const c = CHIP_COLOURS[i % CHIP_COLOURS.length];
      frag.append(h('span', { class: 'llm-chip', style: { borderColor: c + '88', background: c + '1c' }, title: `token #${ids[i]}: ${JSON.stringify(piece)}` }, h('span', { class: 't' }, showToken(piece)), h('span', { class: 'id' }, String(ids[i]))));
    }
    if (ids.length > show) frag.append(h('span', { class: 'muted small' }, ` … ${ids.length - show} more`));
    this.chips.append(frag);
    let dropped = 0;
    if (tok.kind === 'char') for (const ch of text) if (!tok.vocab.includes(ch)) dropped++;
    setText(this.stats, `${text.length} characters → ${ids.length} tokens${ids.length ? ` (${fmt(text.length / ids.length, 2)} characters per token)` : ''}${dropped ? ` · ${dropped} characters are not in the vocabulary and are skipped` : ''}`);
  }

  private renderVocab(): void {
    const tok = this.shownTok;
    clear(this.vocab);
    if (!tok) return;
    const q = this.filter.value;
    const frag = document.createDocumentFragment();
    let n = 0;
    tok.vocab.forEach((v, i) => {
      if (q && !v.includes(q)) return;
      if (n++ >= 3000) return;
      frag.append(h('span', { class: 'llm-chip small', title: `token #${i}: ${JSON.stringify(v)}` }, h('span', { class: 't' }, showToken(v)), h('span', { class: 'id' }, String(i))));
    });
    this.vocab.append(h('div', { class: 'muted small', style: { width: '100%', marginBottom: '4px' } }, `${tok.size} tokens${q ? ` · ${n} match` : ''}. The model’s last layer gives one score to each of them.`), frag);
  }
}

export class DataView {
  readonly el: HTMLElement;
  private studio: LLMStudio;
  private info: HTMLElement;
  private actions: HTMLElement;
  private freq: HTMLElement;
  private preview: HTMLElement;
  private dirty = true;

  constructor(studio: LLMStudio) {
    this.studio = studio;
    this.info = h('div');
    this.actions = h('div', { class: 'row', style: { gap: '6px', marginTop: '8px', flexWrap: 'wrap' } });
    this.freq = h('div', { class: 'llm-freq' });
    this.preview = h('pre', { class: 'llm-preview' });
    this.el = h(
      'div',
      null,
      section('Training text', 'scroll', [this.info, this.actions]),
      section('Most common characters', 'chart', [this.freq]),
      section('Preview', 'eye', [this.preview, h('div', { class: 'note' }, 'The model trains on random windows of this text: it sees a window and learns to predict every next token in it. The last part (after the line) is the validation text: the model never trains on it, so its loss there shows how well it generalises.')]),
    );
  }

  invalidate(): void {
    this.dirty = true;
  }

  render(visible: boolean): void {
    if (!visible || !this.dirty) return;
    this.dirty = false;
    const p = this.studio.project;
    const c = this.studio.corpus;
    const text = c.text;
    const ds = getDataset(p.data.dataset);
    clear(this.info);
    clear(this.actions);
    this.info.append(
      h('div', { class: 'row' }, h('span', { style: { fontSize: '20px' } }, ds?.emoji ?? '✏️'), h('b', null, ds?.name ?? 'Your own text')),
      h('div', { class: 'dim small', style: { marginTop: '4px' } }, ds?.blurb ?? 'Text you pasted or loaded. It never leaves your browser.'),
    );
    const valStart = c.valStart ?? Math.floor(text.length * 0.9);
    const lines = text.split('\n').length;
    this.info.append(
      h(
        'div',
        { class: 'kv', style: { marginTop: '10px' } },
        h('span', null, 'Characters'),
        h('b', null, text.length.toLocaleString()),
        h('span', null, 'Lines'),
        h('b', null, lines.toLocaleString()),
        h('span', null, 'Different characters'),
        h('b', null, String(new Set(text).size)),
        h('span', null, 'Training part'),
        h('b', null, `${valStart.toLocaleString()} characters`),
        h('span', null, 'Validation part'),
        h('b', null, `${(text.length - valStart).toLocaleString()} characters`),
      ),
    );
    if (p.data.dataset === 'custom') this.actions.append(btn({ icon: 'pencil', label: 'Edit text', cls: 'sm', onclick: () => this.studio.editText() }));
    else
      this.actions.append(
        btn({
          icon: 'shuffle',
          label: 'Generate different text',
          cls: 'sm',
          title: 'The built-in texts are generated: make a new one with another random seed',
          onclick: () => {
            p.data.seed = (p.data.seed % 97) + 1;
            this.studio.builder.refresh();
            this.studio.settingsChanged('data');
          },
        }),
        btn({ icon: 'pencil', label: 'Use your own text instead', cls: 'ghost sm', onclick: () => this.studio.editText() }),
      );
    // character frequencies
    const counts = new Map<string, number>();
    for (let i = 0; i < Math.min(text.length, 300_000); i++) counts.set(text[i], (counts.get(text[i]) ?? 0) + 1);
    const top = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 24);
    const max = top.length ? top[0][1] : 1;
    clear(this.freq);
    for (const [ch, n] of top) this.freq.append(h('div', { class: 'llm-freq-col', title: `${JSON.stringify(ch)}: ${n.toLocaleString()} times` }, h('div', { class: 'bar' }, h('i', { style: { height: `${Math.max(2, (n / max) * 100)}%` } })), h('code', null, showToken(ch))));
    // preview: the start of the text, and the start of the validation part
    clear(this.preview);
    this.preview.append(text.slice(0, Math.min(valStart, 2500)));
    if (valStart > 2500) this.preview.append(h('span', { class: 'muted' }, `\n… ${(valStart - 2500).toLocaleString()} more characters …\n`));
    this.preview.append(h('span', { class: 'llm-valmark' }, icon('flask', 12), ' validation text starts here '), '\n', h('span', { class: 'dim' }, text.slice(valStart, valStart + 800)));
  }
}
