/**
 * Tokenizers: how text becomes the numbers a language model reads.
 *
 *  • characters: every character is a token. Tiny vocabulary, long sequences.
 *  • words:      frequent words (with their leading space) are tokens; anything
 *                else falls back to single characters, so nothing is lost.
 *  • BPE:        byte-pair encoding, the method GPT uses. Start from characters
 *                and repeatedly MERGE the most frequent pair of neighbouring
 *                tokens into a new token, until the vocabulary is big enough.
 *
 * Text is first split into "pre-tokens" (a word with its leading space, a
 * number digit, a run of punctuation, whitespace), like GPT-2, so merges never
 * cross word boundaries. The pattern is ASCII-only so the exported Python
 * script can reproduce it with the standard `re` module.
 */

export type TokenizerKind = 'char' | 'word' | 'bpe';

export interface TokenizerJSON {
  kind: TokenizerKind;
  vocab: string[];
  /** BPE merges, in the order they were learned */
  merges?: [string, string][];
}

/** GPT-2-style pre-tokenizer (ASCII version): " word", " 7" (one digit), " ?!", whitespace. */
export const PRETOKEN_RE = / ?[A-Za-z]+| ?[0-9]| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+/g;

export function pretokenize(text: string): string[] {
  return text.match(PRETOKEN_RE) ?? [];
}

/** Printable version of a token: spaces as ·, new lines as ↵, tabs as →. */
export function showToken(s: string): string {
  return s.replace(/ /g, '·').replace(/\n/g, '↵').replace(/\t/g, '→').replace(/\r/g, '␍');
}

export class Tokenizer {
  readonly kind: TokenizerKind;
  readonly vocab: string[];
  readonly merges: [string, string][];
  private index = new Map<string, number>();
  private ranks = new Map<string, number>();
  private cache = new Map<string, number[]>();

  constructor(j: TokenizerJSON) {
    this.kind = j.kind;
    this.vocab = j.vocab.slice();
    this.merges = (j.merges ?? []).map((m) => [m[0], m[1]] as [string, string]);
    this.vocab.forEach((t, i) => this.index.set(t, i));
    this.merges.forEach(([a, b], i) => this.ranks.set(a + '\u0000' + b, i));
  }

  get size(): number {
    return this.vocab.length;
  }

  /** Learn a tokenizer from text. vocabSize is a target for words / BPE (characters always included). */
  static train(kind: TokenizerKind, text: string, vocabSize = 256): Tokenizer {
    const chars = [...new Set(text)].sort();
    if (kind === 'char') return new Tokenizer({ kind, vocab: chars });
    const words = new Map<string, number>();
    for (const w of pretokenize(text)) words.set(w, (words.get(w) ?? 0) + 1);
    if (kind === 'word') {
      const base = new Set(chars);
      const extra = [...words.entries()]
        .filter(([w, c]) => w.length > 1 && c >= 2 && !base.has(w))
        .sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))
        .slice(0, Math.max(0, vocabSize - chars.length))
        .map(([w]) => w);
      return new Tokenizer({ kind, vocab: [...chars, ...extra] });
    }
    // BPE: work on unique pre-tokens with their counts, and keep the pair counts up to date
    // incrementally (only words that contain the merged pair are revisited)
    const seqs = [...words.entries()].map(([w, c]) => ({ syms: [...w], count: c }));
    const vocab = chars.slice();
    const merges: [string, string][] = [];
    const pairs = new Map<string, number>();
    const where = new Map<string, Set<number>>();
    const addPairs = (i: number, sign: number) => {
      const s = seqs[i];
      const y = s.syms;
      for (let j = 0; j + 1 < y.length; j++) {
        const k = y[j] + '\u0000' + y[j + 1];
        pairs.set(k, (pairs.get(k) ?? 0) + sign * s.count);
        if (sign > 0) {
          let set = where.get(k);
          if (!set) where.set(k, (set = new Set()));
          set.add(i);
        }
      }
    };
    for (let i = 0; i < seqs.length; i++) addPairs(i, 1);
    while (vocab.length < vocabSize) {
      let best = '';
      let bc = 1;
      for (const [k, c] of pairs) if (c > bc || (c === bc && best && k < best)) [best, bc] = [k, c];
      if (!best || bc < 2) break;
      const [a, b] = best.split('\u0000');
      const ab = a + b;
      merges.push([a, b]);
      vocab.push(ab);
      for (const i of where.get(best) ?? []) {
        const y = seqs[i].syms;
        let has = false;
        for (let j = 0; j + 1 < y.length; j++)
          if (y[j] === a && y[j + 1] === b) {
            has = true;
            break;
          }
        if (!has) continue;
        addPairs(i, -1);
        const out: string[] = [];
        for (let j = 0; j < y.length; j++) {
          if (j + 1 < y.length && y[j] === a && y[j + 1] === b) {
            out.push(ab);
            j++;
          } else out.push(y[j]);
        }
        seqs[i].syms = out;
        addPairs(i, 1);
      }
      pairs.delete(best);
      where.delete(best);
    }
    return new Tokenizer({ kind, vocab, merges });
  }

  /** Text → token ids. Characters the tokenizer has never seen are skipped. */
  encode(text: string): Int32Array {
    const out: number[] = [];
    if (this.kind === 'char') {
      for (const ch of text) {
        const i = this.index.get(ch);
        if (i !== undefined) out.push(i);
      }
      return Int32Array.from(out);
    }
    for (const w of pretokenize(text)) {
      let ids = this.cache.get(w);
      if (!ids) {
        ids = this.kind === 'word' ? this.encodeWord(w) : this.encodeBpe(w);
        if (this.cache.size < 50000) this.cache.set(w, ids);
      }
      for (const i of ids) out.push(i);
    }
    return Int32Array.from(out);
  }

  private charIds(w: string): number[] {
    const ids: number[] = [];
    for (const ch of w) {
      const i = this.index.get(ch);
      if (i !== undefined) ids.push(i);
    }
    return ids;
  }

  private encodeWord(w: string): number[] {
    const i = this.index.get(w);
    return i !== undefined ? [i] : this.charIds(w);
  }

  /** Apply the learned merges, lowest rank (earliest learned) first. */
  private encodeBpe(w: string): number[] {
    let syms = [...w].filter((ch) => this.index.has(ch));
    while (syms.length > 1) {
      let best = -1;
      let rank = Infinity;
      for (let i = 0; i + 1 < syms.length; i++) {
        const r = this.ranks.get(syms[i] + '\u0000' + syms[i + 1]);
        if (r !== undefined && r < rank) {
          rank = r;
          best = i;
        }
      }
      if (best < 0) break;
      const [a, b] = this.merges[rank];
      const out: string[] = [];
      for (let i = 0; i < syms.length; i++) {
        if (i + 1 < syms.length && syms[i] === a && syms[i + 1] === b) {
          out.push(a + b);
          i++;
        } else out.push(syms[i]);
      }
      syms = out;
    }
    return syms.map((s) => this.index.get(s)!);
  }

  decode(ids: ArrayLike<number>): string {
    let s = '';
    for (let i = 0; i < ids.length; i++) s += this.vocab[ids[i]] ?? '';
    return s;
  }

  /** Split text into its tokens (strings), e.g. for colouring them. */
  pieces(text: string): string[] {
    return Array.from(this.encode(text), (i) => this.vocab[i]);
  }

  toJSON(): TokenizerJSON {
    return this.kind === 'bpe' ? { kind: this.kind, vocab: this.vocab.slice(), merges: this.merges.map((m) => [m[0], m[1]] as [string, string]) } : { kind: this.kind, vocab: this.vocab.slice() };
  }
}
