/**
 * A phone keyboard on top of a mini language model:
 *
 *  - Word prediction: the most likely next words (or completions of the word
 *    you are typing). The model predicts TOKENS, not words, so a beam search
 *    keeps the few most likely token sequences until each one finishes a word.
 *    Works with any tokenizer (characters, words, BPE).
 *
 *  - Autocorrect, the classic "noisy channel" way: when you finish a word,
 *    look at known words a letter or two away from what you typed, and pick
 *    the one with the best   language-model score (does it fit here?)
 *                          + typo score (how many keys would you have missed?).
 */
import { Decoder, lineBreakTable, windowStart, withStart } from './decode.ts';
import type { LineMode } from './decode.ts';
import type { GPT } from './gpt.ts';
import type { Tokenizer } from './tokenizer.ts';

const isLetter = (c: string) => (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');

/** log-softmax of the decoder's current logits */
function logProbs(logits: Float32Array): Float32Array {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  let sum = 0;
  for (let i = 0; i < logits.length; i++) sum += Math.exp(logits[i] - max);
  const lse = max + Math.log(sum);
  const out = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) out[i] = logits[i] - lse;
  return out;
}

/**
 * Split typed text into the context and the start of the word being typed.
 * The leading space belongs to the word (that is how the tokenizers cut text),
 * so it moves from the context into `lead`.
 */
export function splitForSuggest(text: string): { context: string; lead: string; partial: string } {
  let i = text.length;
  while (i > 0 && isLetter(text[i - 1])) i--;
  const partial = text.slice(i);
  let context = text.slice(0, i);
  let lead = partial;
  if (context.endsWith(' ')) {
    context = context.slice(0, -1);
    lead = ' ' + partial;
  }
  return { context, lead, partial };
}

export interface KeyboardModel {
  model: GPT;
  tok: Tokenizer;
  /** one example per line: how the training windows lined up with the lines */
  align?: LineMode;
}

/**
 * A decoder that has read `context` (the newest part that fits the window).
 * With nothing to read it starts like a new line, so its first prediction is
 * "the first word of a text".
 */
export function readContext(km: KeyboardModel, context: string): Decoder {
  const { model, tok } = km;
  const dec = new Decoder(model);
  const breaks = km.align ? lineBreakTable(tok.vocab) : null;
  dec.lineBreaks = breaks;
  dec.lineMode = km.align ?? 'start';
  let ids = withStart(Array.from(tok.encode(context)), breaks, tok.vocab, dec.lineMode);
  if (!ids.length) {
    const nl = tok.encode('\n');
    ids = nl.length ? [nl[0]] : [0];
  }
  dec.feed(ids.slice(windowStart(ids, model.cfg.ctx - 1, breaks, dec.lineMode)));
  return dec;
}

export interface Suggestion {
  word: string;
  /** probability of exactly this word (given the context and what you typed) */
  p: number;
  /** the model put a space before it (the text did not end with one) */
  space: boolean;
}

/**
 * The k most likely words that continue `text`: next words after a space, or
 * completions of a half-typed word. Beam search over tokens until a word ends.
 */
export function suggestWords(km: KeyboardModel, text: string, k = 3, opts: { beam?: number; maxSteps?: number } = {}): Suggestion[] {
  const { tok } = km;
  const { context, lead } = splitForSuggest(text);
  const W = opts.beam ?? 6;
  const maxSteps = opts.maxSteps ?? (tok.kind === 'char' ? 18 : 8);
  const root = readContext(km, context);
  const vocab = tok.vocab;
  interface Beam {
    dec: Decoder;
    gen: string;
    logp: number;
  }
  let beams: Beam[] = [{ dec: root, gen: '', logp: 0 }];
  const done = new Map<string, { lp: number; space: boolean }>();
  const kthBest = () => {
    if (done.size < k) return -Infinity;
    return [...done.values()].map((d) => d.lp).sort((a, b) => b - a)[k - 1];
  };
  for (let step = 0; step < maxSteps && beams.length; step++) {
    const cand: { parent: Beam; id: number; gen: string; logp: number }[] = [];
    for (const b of beams) {
      const lp = logProbs(b.dec.logits);
      // the best W·2 tokens that agree with what was typed
      const best: { id: number; lp: number }[] = [];
      for (let id = 0; id < vocab.length; id++) {
        const s = b.gen + vocab[id];
        const ok = s.length <= lead.length ? lead.startsWith(s) : s.startsWith(lead);
        if (!ok || !vocab[id]) continue;
        const v = lp[id];
        if (best.length < W * 2 || v > best[best.length - 1].lp) {
          let j = best.length < W * 2 ? best.length : best.length - 1;
          best[j] = { id, lp: v };
          while (j > 0 && best[j - 1].lp < best[j].lp) {
            [best[j - 1], best[j]] = [best[j], best[j - 1]];
            j--;
          }
        }
      }
      for (const c of best) cand.push({ parent: b, id: c.id, gen: b.gen + vocab[c.id], logp: b.logp + c.lp });
    }
    cand.sort((a, b) => b.logp - a.logp);
    const next: Beam[] = [];
    for (const c of cand) {
      // finished: past what was typed, a word made of letters, then something that is not a letter
      if (c.gen.length > lead.length) {
        const m = /^( ?)([A-Za-z]+)([^A-Za-z])/.exec(c.gen);
        if (m) {
          const word = m[2];
          const prev = done.get(word);
          done.set(word, { lp: Math.log(Math.exp(prev?.lp ?? -Infinity) + Math.exp(c.logp)), space: m[1] === ' ' && !lead.startsWith(' ') });
          continue;
        }
        // something other than a word (punctuation, digits…): not a suggestion
        if (!/^ ?[A-Za-z]*$/.test(c.gen)) continue;
      }
      if (next.length >= W) continue;
      if (c.logp < kthBest()) continue; // can no longer beat the k-th best finished word
      const dec = c.parent.dec.clone();
      dec.step(c.id);
      next.push({ dec, gen: c.gen, logp: c.logp });
    }
    beams = next;
  }
  return [...done.entries()]
    .sort((a, b) => b[1].lp - a[1].lp)
    .slice(0, k)
    .map(([word, d]) => ({ word, p: Math.exp(d.lp), space: d.space }));
}

/** log P(the model writes exactly `str` next, and the word ends there) after the decoder's state. */
export function scoreWord(dec: Decoder, tok: Tokenizer, str: string): number {
  const d = dec.clone();
  let lp = 0;
  for (const id of tok.encode(str)) {
    lp += logProbs(d.logits)[id];
    d.step(id);
  }
  // the word must END here: the next token starts with something that is not a letter
  const l = logProbs(d.logits);
  let end = 0;
  for (let id = 0; id < tok.vocab.length; id++) {
    const c = tok.vocab[id][0];
    if (c !== undefined && !isLetter(c)) end += Math.exp(l[id]);
  }
  return lp + Math.log(Math.max(end, 1e-12));
}

/** Damerau–Levenshtein distance (with swaps of neighbours), giving up above `max`. */
export function editDistance(a: string, b: string, max = 3): number {
  if (Math.abs(a.length - b.length) > max) return max + 1;
  const n = a.length;
  const m = b.length;
  let prev2 = new Array<number>(m + 1).fill(0);
  let prev = Array.from({ length: m + 1 }, (_, j) => j);
  for (let i = 1; i <= n; i++) {
    const cur = new Array<number>(m + 1);
    cur[0] = i;
    let rowMin = cur[0];
    for (let j = 1; j <= m; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      let v = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost);
      if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) v = Math.min(v, prev2[j - 2] + 1);
      cur[j] = v;
      if (v < rowMin) rowMin = v;
    }
    if (rowMin > max) return max + 1;
    prev2 = prev;
    prev = cur;
  }
  return prev[m];
}

/** Every word in a text, lower-cased, with how often it appears. */
export function knownWords(text: string): Map<string, number> {
  const out = new Map<string, number>();
  for (const m of text.matchAll(/[A-Za-z]+/g)) {
    const w = m[0].toLowerCase();
    out.set(w, (out.get(w) ?? 0) + 1);
  }
  return out;
}

export interface Correction {
  typed: string;
  /** what autocorrect would write (= typed when it keeps your word) */
  word: string;
  changed: boolean;
  /** the candidates it considered, best first: language-model score, typo score, total */
  candidates: { word: string; edits: number; lm: number; total: number }[];
}

/** cost of one typo, in log-probability (≈ "one key in a hundred is a mistake") */
export const TYPO_COST = Math.log(100);

const matchCase = (word: string, like: string) => (like[0] && like[0] === like[0].toUpperCase() && like[0] !== like[0].toLowerCase() ? word[0].toUpperCase() + word.slice(1) : word);

/**
 * Autocorrect the word just typed: `before` is all the text in front of it
 * (including the space that separates them). Keeps the word if it is known,
 * otherwise picks the best close known word if it is clearly better.
 */
export function correctWord(km: KeyboardModel, before: string, typed: string, known: Map<string, number>, opts: { margin?: number; maxCandidates?: number } = {}): Correction {
  const lower = typed.toLowerCase();
  const none: Correction = { typed, word: typed, changed: false, candidates: [] };
  if (!typed || known.has(lower) || !/^[A-Za-z]+$/.test(typed)) return none;
  const maxEdits = typed.length <= 4 ? 1 : 2;
  const close: { word: string; edits: number; count: number }[] = [];
  for (const [w, count] of known) {
    if (Math.abs(w.length - lower.length) > maxEdits) continue;
    const e = editDistance(lower, w, maxEdits);
    if (e <= maxEdits) close.push({ word: matchCase(w, typed), edits: e, count });
  }
  if (!close.length) return none;
  close.sort((a, b) => a.edits - b.edits || b.count - a.count);
  const pool = close.slice(0, opts.maxCandidates ?? 24);
  const { context: ctx, lead } = splitForSuggest(before);
  const space = lead.startsWith(' ') ? ' ' : '';
  const dec = readContext(km, ctx);
  // judge every word in the casing the model knows best (it may only have seen "thank", or only "Thank")
  const forms = (w: string) => [...new Set([w, w.toLowerCase(), w[0].toUpperCase() + w.slice(1).toLowerCase()])];
  const fit = (w: string) => Math.max(...forms(w).map((f) => scoreWord(dec, km.tok, space + f)));
  const scored = pool.map((c) => {
    const lm = fit(c.word);
    return { word: c.word, edits: c.edits, lm, total: lm - c.edits * TYPO_COST };
  });
  scored.sort((a, b) => b.total - a.total);
  const typedScore = fit(typed);
  const best = scored[0];
  const changed = best.total > typedScore + (opts.margin ?? 1);
  return { typed, word: changed ? best.word : typed, changed, candidates: [{ word: typed, edits: 0, lm: typedScore, total: typedScore }, ...scored].sort((a, b) => b.total - a.total) };
}
