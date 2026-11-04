/**
 * Text generation with a KV cache.
 *
 * Generating one token at a time with a full forward pass would redo the
 * work for the whole context at every step. Instead, each layer keeps the
 * keys and values of the tokens seen so far, so a new token only needs its
 * own query, key and value: one row of work per layer.
 *
 * When the context window is full, the oldest half is dropped and the rest
 * is fed again (the model only knows positions 0 … ctx−1).
 */
import { denseForward } from '../ml/nn.ts';
import { RNG } from '../ml/rng.ts';
import type { GPT } from './gpt.ts';
import { activate, layerNormForward } from './gpt.ts';

/** Which token ids contain a line break (for windows that should start at a line). */
export function lineBreakTable(vocab: string[]): Uint8Array {
  return Uint8Array.from(vocab, (v) => (v.includes('\n') ? 1 : 0));
}

/**
 * Where to start a window of at most `max` tokens that ends at the end of
 * `tokens`. With `breaks` (one example per line), it starts at the earliest
 * line start that fits, like the training windows; otherwise as early as fits.
 */
export function windowStart(tokens: ArrayLike<number>, max: number, breaks: Uint8Array | null): number {
  const first = Math.max(0, tokens.length - max);
  if (!breaks || first === 0) return first;
  for (let i = first; i < tokens.length; i++) if (breaks[tokens[i - 1]]) return i;
  return first;
}

export class Decoder {
  readonly model: GPT;
  /** set for one-example-per-line data: when the window is full it slides to a line start */
  lineBreaks: Uint8Array | null = null;
  private kc: Float32Array[];
  private vc: Float32Array[];
  pos = 0;
  /** tokens currently in the window */
  tokens: number[] = [];
  /** attention of the newest token, per layer: [heads × ctx] */
  lastAtt: Float32Array[];
  private x: Float32Array;
  private h: Float32Array;
  private qkv: Float32Array;
  private y: Float32Array;
  private a: Float32Array;
  private f: Float32Array;
  private g: Float32Array;
  private xhat: Float32Array;
  private rstd = new Float32Array(1);
  private zeroV: Float32Array;
  readonly logits: Float32Array;

  constructor(model: GPT) {
    this.model = model;
    const { ctx, d, layers, heads, mlp, vocab } = model.cfg;
    this.kc = Array.from({ length: layers }, () => new Float32Array(ctx * d));
    this.vc = Array.from({ length: layers }, () => new Float32Array(ctx * d));
    this.lastAtt = Array.from({ length: layers }, () => new Float32Array(heads * ctx));
    this.x = new Float32Array(d);
    this.h = new Float32Array(d);
    this.qkv = new Float32Array(3 * d);
    this.y = new Float32Array(d);
    this.a = new Float32Array(d);
    this.f = new Float32Array(mlp * d);
    this.g = new Float32Array(mlp * d);
    this.xhat = new Float32Array(Math.max(d, 1));
    this.zeroV = new Float32Array(vocab);
    this.logits = new Float32Array(vocab);
  }

  reset(): void {
    this.pos = 0;
    this.tokens = [];
  }

  /** Feed a list of tokens (e.g. the prompt); returns the logits after the last one. */
  feed(tokens: ArrayLike<number>): Float32Array {
    for (let i = 0; i < tokens.length; i++) this.step(tokens[i]);
    return this.logits;
  }

  /** Feed one token; returns the logits for the NEXT token (an internal buffer, copy it to keep it). */
  step(tok: number): Float32Array {
    const m = this.model;
    const { ctx, d, heads: H, mlp, act } = m.cfg;
    if (this.pos >= ctx) {
      // window full: keep the newest half (or, for line-by-line data, everything from the
      // earliest line start that leaves room), recompute their keys/values from position 0
      let from = this.tokens.length - Math.floor(ctx / 2);
      if (this.lineBreaks) {
        const n = this.tokens.length;
        const ls = windowStart(this.tokens, ctx - 1, this.lineBreaks);
        // the window ends with a line break: the new line starts fresh at position 0
        if (this.lineBreaks[this.tokens[n - 1]]) from = n;
        else if (ls > 0 && this.lineBreaks[this.tokens[ls - 1]]) from = ls;
      }
      const keep = this.tokens.slice(from);
      this.reset();
      for (const t of keep) this.step(t);
    }
    const p = this.pos;
    const x = this.x;
    const e = (tok | 0) * d;
    const pv = m.posVec(p);
    for (let i = 0; i < d; i++) x[i] = m.wte.w[e + i] + (pv ? pv[i] : 0);
    const dh = d / H;
    const scale = 1 / Math.sqrt(dh);
    for (let l = 0; l < m.blocks.length; l++) {
      const blk = m.blocks[l];
      layerNormForward(x, blk.ln1g.w, blk.ln1b.w, this.h, this.xhat, this.rstd, 1, d);
      denseForward(blk.qkvW.w, blk.qkvB.w, this.h, this.qkv, 1, d, 3 * d);
      const K = this.kc[l];
      const Vc = this.vc[l];
      for (let i = 0; i < d; i++) {
        K[p * d + i] = this.qkv[d + i];
        Vc[p * d + i] = this.qkv[2 * d + i];
      }
      const att = this.lastAtt[l];
      for (let hh = 0; hh < H; hh++) {
        const ho = hh * dh;
        const row = hh * ctx;
        let max = -Infinity;
        for (let s = 0; s <= p; s++) {
          let sc = 0;
          for (let j = 0; j < dh; j++) sc += this.qkv[ho + j] * K[s * d + ho + j];
          sc *= scale;
          att[row + s] = sc;
          if (sc > max) max = sc;
        }
        let sum = 0;
        for (let s = 0; s <= p; s++) sum += att[row + s] = Math.exp(att[row + s] - max);
        for (let s = 0; s <= p; s++) att[row + s] /= sum;
        for (let s = p + 1; s < ctx; s++) att[row + s] = 0;
        for (let j = 0; j < dh; j++) {
          let acc = 0;
          for (let s = 0; s <= p; s++) acc += att[row + s] * Vc[s * d + ho + j];
          this.y[ho + j] = acc;
        }
      }
      denseForward(blk.projW.w, blk.projB.w, this.y, this.a, 1, d, d);
      for (let i = 0; i < d; i++) x[i] += this.a[i];
      layerNormForward(x, blk.ln2g.w, blk.ln2b.w, this.h, this.xhat, this.rstd, 1, d);
      denseForward(blk.fcW.w, blk.fcB.w, this.h, this.f, 1, d, mlp * d);
      activate(act, this.f, this.g, mlp * d);
      denseForward(blk.mprojW.w, blk.mprojB.w, this.g, this.a, 1, mlp * d, d);
      for (let i = 0; i < d; i++) x[i] += this.a[i];
    }
    layerNormForward(x, m.lnfg.w, m.lnfb.w, this.h, this.xhat, this.rstd, 1, d);
    denseForward(m.headW, this.zeroV, this.h, this.logits, 1, d, m.cfg.vocab);
    this.pos++;
    this.tokens.push(tok | 0);
    return this.logits;
  }
}

/** Softmax with temperature: probabilities of the next token. */
export function softmaxT(logits: ArrayLike<number>, temperature = 1): Float32Array {
  const n = logits.length;
  const t = Math.max(1e-4, temperature);
  const p = new Float32Array(n);
  let max = -Infinity;
  for (let i = 0; i < n; i++) if (logits[i] / t > max) max = logits[i] / t;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += p[i] = Math.exp(logits[i] / t - max);
  for (let i = 0; i < n; i++) p[i] /= sum;
  return p;
}

/** The k most likely tokens: [id, probability][] (highest first). */
export function topK(probs: ArrayLike<number>, k: number): [number, number][] {
  const n = probs.length;
  k = Math.max(0, Math.min(k, n));
  if (k * 8 >= n) {
    const idx = Array.from({ length: n }, (_, i) => i);
    idx.sort((a, b) => probs[b] - probs[a]);
    return idx.slice(0, k).map((i) => [i, probs[i]]);
  }
  // small k: keep a sorted list of the best k (insertion), one pass over the vocabulary
  const ids: number[] = [];
  const ps: number[] = [];
  for (let i = 0; i < n; i++) {
    const p = probs[i];
    if (ids.length === k && p <= ps[k - 1]) continue;
    let j = ids.length < k ? ids.length : k - 1;
    while (j > 0 && ps[j - 1] < p) {
      if (j < k) {
        ids[j] = ids[j - 1];
        ps[j] = ps[j - 1];
      }
      j--;
    }
    ids[j] = i;
    ps[j] = p;
    if (ids.length > k) {
      ids.length = k;
      ps.length = k;
    }
  }
  return ids.map((id, j) => [id, ps[j]]);
}

/**
 * Pick the next token. temperature 0 = always the most likely one;
 * topK > 0 = only sample among the k most likely; topP < 1 = nucleus sampling.
 */
export function sampleToken(logits: ArrayLike<number>, rng: RNG, temperature = 1, k = 0, topP = 1): number {
  if (temperature <= 0) {
    let best = 0;
    for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
    return best;
  }
  const p = softmaxT(logits, temperature);
  let cand: [number, number][] | null = null;
  if (k > 0 && k < p.length) cand = topK(p, k);
  if (topP < 1) {
    const sorted = cand ?? topK(p, p.length);
    const keep: [number, number][] = [];
    let acc = 0;
    for (const c of sorted) {
      keep.push(c);
      acc += c[1];
      if (acc >= topP) break;
    }
    cand = keep;
  }
  if (cand) {
    const tot = cand.reduce((a, c) => a + c[1], 0);
    let u = rng.next() * tot;
    for (const [i, q] of cand) {
      u -= q;
      if (u <= 0) return i;
    }
    return cand[cand.length - 1][0];
  }
  let u = rng.next();
  for (let i = 0; i < p.length; i++) {
    u -= p[i];
    if (u <= 0) return i;
  }
  return p.length - 1;
}
