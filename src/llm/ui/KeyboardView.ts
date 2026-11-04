/**
 * The keyboard: type with your own language model, like on a phone.
 * Suggestions for the next word (tap or Tab), autocorrect when a word ends
 * (Backspace right after a correction undoes it), and the reasoning behind
 * both, under the hood.
 */
import { btn, clear, fmt, h, section, setText } from '../../ui/dom.ts';
import { icon } from '../../ui/icons.ts';
import { TYPO_COST, correctWord, knownWords, splitForSuggest, suggestWords } from '../keyboard.ts';
import type { Correction, KeyboardModel, Suggestion } from '../keyboard.ts';
import { pct } from './Playground.ts';
import type { LLMStudio } from './LLMStudio.ts';

const BOUNDARY = /^[\s.,!?;:]$/;

export class KeyboardView {
  readonly el: HTMLElement;
  private studio: LLMStudio;
  private input: HTMLTextAreaElement;
  private bar: HTMLElement;
  private stats: HTMLElement;
  private why: HTMLElement;
  private fixes: HTMLElement;
  private fixDetail: HTMLElement;
  private empty: HTMLElement;
  private body: HTMLElement;
  private auto: HTMLInputElement;
  private sugg: Suggestion[] = [];
  private known: { key: string; words: Map<string, number> } | null = null;
  private ignore = new Set<string>();
  private lastFix: { start: number; typed: string; word: string; boundary: string; at: number } | null = null;
  private typed = 0;
  private saved = 0;
  private pending = 0;
  private shownVersion = -1;
  private lastRefresh = 0;

  constructor(studio: LLMStudio) {
    this.studio = studio;
    this.input = h('textarea', { class: 'input llm-prompt kb-input', rows: '5', spellcheck: 'false', autocapitalize: 'sentences', placeholder: 'Type a message… (try a typo, e.g. "see you tomorow")' }) as HTMLTextAreaElement;
    this.bar = h('div', { class: 'kb-bar' });
    this.auto = h('input', { type: 'checkbox', checked: true }) as HTMLInputElement;
    this.stats = h('span', { class: 'muted small' });
    this.why = h('div', { class: 'kb-why' });
    this.fixes = h('div', { class: 'kb-fixes' });
    this.fixDetail = h('div');
    this.empty = h('div', { class: 'empty-note' }, icon('keyboard'), h('div', null, h('b', null, 'No model yet.'), h('div', { class: 'dim small' }, 'Train a model first. The ⌨️ Phone keyboard preset (Presets in the top bar) learns everyday text messages in a minute or two; any model works, even one trained on your own writing.')));
    this.body = h(
      'div',
      null,
      section('Keyboard', 'keyboard', [
        h('div', { class: 'note', style: { marginTop: 0, marginBottom: '8px' } }, 'Type below. Your model suggests the next word: tap one or press Tab. When you finish a word, autocorrect fixes typos (press Backspace right after a correction to undo it).'),
        this.bar,
        this.input,
        h('div', { class: 'row', style: { marginTop: '8px', gap: '12px', flexWrap: 'wrap' } }, h('label', { class: 'row small dim' }, this.auto, 'Autocorrect'), btn({ icon: 'eraser', label: 'Clear', cls: 'ghost xs', onclick: () => this.clearText() }), h('div', { class: 'spacer' }), this.stats),
      ]),
      section('How it chose the suggestions', 'sparkles', [
        h('div', { class: 'note', style: { marginTop: 0, marginBottom: '8px' } }, 'The model only predicts the next TOKEN. To suggest whole words, the keyboard follows the few most likely token sequences at once (a beam search) until each one finishes a word, then ranks the words by probability.'),
        this.why,
      ]),
      section('Autocorrect', 'check', [
        h('div', { class: 'note', style: { marginTop: 0, marginBottom: '8px' } }, `When you finish a word it does not know, it looks at known words one or two letters away (a swap, a missing or extra letter…) and scores each: how well it fits here according to the model, minus ${fmt(TYPO_COST, 1)} for every typo. It only changes your word if another one clearly wins. The "known words" are the words of the training text.`),
        this.fixDetail,
        this.fixes,
      ]),
    );
    this.el = h('div', { class: 'kb' }, this.empty, this.body);
    this.input.addEventListener('beforeinput', (e) => this.beforeInput(e as InputEvent));
    this.input.addEventListener('input', (e) => this.onInput(e as InputEvent));
    this.input.addEventListener('keydown', (e) => this.onKey(e));
    this.input.addEventListener('click', () => this.schedule());
    this.input.addEventListener('keyup', (e) => {
      if (e.key.startsWith('Arrow') || e.key === 'Home' || e.key === 'End') this.schedule();
    });
  }

  private km(): KeyboardModel | null {
    const tr = this.studio.trained;
    return tr ? { model: tr.model, tok: tr.tok, align: this.studio.corpus.align } : null;
  }

  private knownWords(): Map<string, number> {
    const t = this.studio.corpus.text;
    const key = `${t.length}:${t.slice(0, 64)}:${t.slice(-64)}`;
    if (this.known?.key !== key) this.known = { key, words: knownWords(this.studio.corpus.text) };
    return this.known.words;
  }

  private schedule(): void {
    cancelAnimationFrame(this.pending);
    this.pending = requestAnimationFrame(() => this.refresh());
  }

  /** Text up to the cursor. */
  private head(): string {
    return this.input.value.slice(0, this.input.selectionStart ?? this.input.value.length);
  }

  private refresh(): void {
    this.lastRefresh = performance.now();
    this.shownVersion = this.studio.version;
    const km = this.km();
    clear(this.bar);
    clear(this.why);
    if (!km) return;
    const text = this.head();
    let all: Suggestion[] = [];
    try {
      all = suggestWords(km, text, 8);
    } catch (e) {
      this.why.append(h('div', { class: 'muted small' }, `Could not predict: ${(e as Error).message}`));
    }
    const top = all[0]?.p ?? 0;
    this.sugg = all.filter((s) => s.p >= Math.max(0.002, top * 0.02)).slice(0, 3);
    const { partial } = splitForSuggest(text);
    if (!this.sugg.length) this.bar.append(h('span', { class: 'muted small' }, partial ? 'No completion for this word.' : 'No suggestion.'));
    this.sugg.forEach((s, i) =>
      this.bar.append(
        h(
          'button',
          { class: 'kb-sugg' + (i === 0 ? ' first' : ''), type: 'button', title: `${pct(s.p)} likely${i === 0 ? ' (Tab)' : ''}`, onmousedown: (e: Event) => e.preventDefault(), onclick: () => this.accept(s) },
          h('span', { class: 'w' }, s.word),
          h('span', { class: 'p' }, pct(s.p)),
        ),
      ),
    );
    // under the hood: the full ranking
    this.why.append(h('div', { class: 'small dim', style: { marginBottom: '4px' } }, partial ? `Words that start with "${partial}", most likely first:` : `Most likely next words after "${text.slice(-30).replace(/\n/g, '↵') || '(start)'}":`));
    for (const s of all) this.why.append(h('div', { class: 'tip-row' }, h('code', null, s.word), h('span', { class: 'bar' }, h('i', { style: { width: `${(s.p / Math.max(top, 1e-9)) * 100}%` } })), h('span', { class: 'mono small' }, pct(s.p))));
    if (!all.length) this.why.append(h('div', { class: 'muted small' }, 'Nothing yet.'));
  }

  /** Insert a suggestion: it replaces the half-typed word and adds a space. */
  private accept(s: Suggestion): void {
    const v = this.input.value;
    const cur = this.input.selectionStart ?? v.length;
    const { partial } = splitForSuggest(v.slice(0, cur));
    const start = cur - partial.length;
    const ins = (s.space ? ' ' : '') + s.word + ' ';
    this.input.value = v.slice(0, start) + ins + v.slice(cur);
    const at = start + ins.length;
    this.input.setSelectionRange(at, at);
    this.saved += Math.max(0, ins.length - partial.length - 1);
    this.lastFix = null;
    this.updateStats();
    this.input.focus();
    this.schedule();
  }

  private beforeInput(e: InputEvent): void {
    if (e.inputType === 'insertText' && e.data) this.typed += e.data.length;
  }

  private onInput(e: InputEvent): void {
    if (e.inputType === 'insertText' && e.data && BOUNDARY.test(e.data) && this.auto.checked) this.autocorrect(e.data);
    else if (e.inputType !== 'insertText') this.lastFix = null;
    this.updateStats();
    this.schedule();
  }

  private onKey(e: KeyboardEvent): void {
    if (e.key === 'Tab' && !e.shiftKey && this.sugg.length) {
      e.preventDefault();
      this.accept(this.sugg[0]);
      return;
    }
    if (e.key === 'Backspace' && this.lastFix && this.input.selectionStart === this.lastFix.at && this.input.selectionEnd === this.lastFix.at) {
      // undo the last autocorrect, like a phone: back to what you typed (and do not correct it again)
      e.preventDefault();
      const f = this.lastFix;
      const v = this.input.value;
      this.input.value = v.slice(0, f.start) + f.typed + v.slice(f.at);
      const at = f.start + f.typed.length;
      this.input.setSelectionRange(at, at);
      this.ignore.add(f.typed.toLowerCase());
      this.lastFix = null;
      this.log(`↩ kept "${f.typed}" (undo)`, 'undo');
      this.schedule();
    }
  }

  /** A word just ended (the user typed `boundary`): correct it if needed. */
  private autocorrect(boundary: string): void {
    const km = this.km();
    if (!km) return;
    const v = this.input.value;
    const end = (this.input.selectionStart ?? v.length) - boundary.length; // where the word ends
    let start = end;
    while (start > 0 && /[A-Za-z]/.test(v[start - 1])) start--;
    const typed = v.slice(start, end);
    if (typed.length < 2 || this.ignore.has(typed.toLowerCase())) return;
    let c: Correction;
    try {
      c = correctWord(km, v.slice(0, start), typed, this.knownWords());
    } catch {
      return;
    }
    if (c.candidates.length) this.showDetail(c);
    if (!c.changed) return;
    this.input.value = v.slice(0, start) + c.word + v.slice(end);
    const at = start + c.word.length + boundary.length;
    this.input.setSelectionRange(at, at);
    this.lastFix = { start, typed, word: c.word, boundary, at };
    const best = c.candidates[0];
    const mine = c.candidates.find((x) => x.word === typed);
    const times = mine ? Math.exp(best.total - mine.total) : NaN;
    const howMuch = !Number.isFinite(times) ? '' : times > 1e6 ? 'over a million times more likely here' : `about ${fmt(times, 0)}× more likely here`;
    this.log(`"${typed}" → "${c.word}"${howMuch ? `: ${howMuch}, ${best.edits} typo${best.edits === 1 ? '' : 's'} away` : ''}`, 'fix');
  }

  private showDetail(c: Correction): void {
    clear(this.fixDetail);
    const rows = c.candidates.slice(0, 6);
    this.fixDetail.append(
      h('div', { class: 'small dim', style: { marginBottom: '4px' } }, `You typed "${c.typed}". Candidates (higher total wins):`),
      h(
        'div',
        { class: 'kb-table' },
        h('span', { class: 'th' }, 'word'),
        h('span', { class: 'th r' }, 'typos'),
        h('span', { class: 'th r' }, 'fits here'),
        h('span', { class: 'th r' }, 'total'),
        ...rows.flatMap((r) => [
          h('code', { class: r.word === c.word ? 'win' : '' }, r.word === c.typed ? `${r.word} (yours)` : r.word),
          h('span', { class: 'r mono' }, String(r.edits)),
          h('span', { class: 'r mono', title: 'log-probability from the model (closer to 0 = fits better)' }, fmt(r.lm, 1)),
          h('span', { class: 'r mono' }, fmt(r.total, 1)),
        ]),
      ),
    );
  }

  private log(msg: string, kind: 'fix' | 'undo'): void {
    this.fixes.prepend(h('div', { class: `kb-fix ${kind}` }, icon(kind === 'fix' ? 'check' : 'undo', 13), h('span', null, msg)));
    while (this.fixes.childElementCount > 8) this.fixes.lastElementChild!.remove();
  }

  private updateStats(): void {
    const total = this.typed + this.saved;
    setText(this.stats, total ? `${this.typed} keys typed · suggestions saved ${this.saved} (${fmt((this.saved / total) * 100, 0)}%)` : '');
  }

  private clearText(): void {
    this.input.value = '';
    this.typed = 0;
    this.saved = 0;
    this.lastFix = null;
    clear(this.fixes);
    clear(this.fixDetail);
    this.updateStats();
    this.input.focus();
    this.schedule();
  }

  render(visible: boolean): void {
    if (!visible) return;
    const tr = this.studio.trained;
    this.empty.classList.toggle('hidden', !!tr);
    this.body.classList.toggle('hidden', !tr);
    if (!tr) return;
    // new weights while training: refresh the suggestions now and then
    if (this.shownVersion !== this.studio.version && performance.now() - this.lastRefresh > 1000) this.refresh();
  }
}
