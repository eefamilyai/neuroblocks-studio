/**
 * Where windows may start when every line is one example (and at 0), as long
 * as a whole window fits:
 *  - 'start': AT the line break in front of each line. The line break works as
 *    a "start of example" token: the model learns what the first word of a
 *    line looks like, and generation can start from a line break.
 *  - 'line': right AFTER it, so every line starts at position 0. Fixed-format
 *    tasks (sums, puzzles) learn much faster this way.
 */
export function lineStarts(ids: Int32Array, vocab: string[], T: number, mode: LineMode = 'start'): Int32Array {
  const nl = new Uint8Array(vocab.length);
  vocab.forEach((v, i) => (nl[i] = v.includes('\n') ? 1 : 0));
  const out: number[] = [];
  const last = ids.length - T - 1;
  if (mode === 'line') {
    if (last >= 0) out.push(0);
    for (let i = 0; i + 1 <= last; i++) if (nl[ids[i]]) out.push(i + 1);
    return Int32Array.from(out);
  }
  if (last >= 0 && !nl[ids[0]]) out.push(0);
  for (let i = 0; i <= last; i++) if (nl[ids[i]]) out.push(i);
  return Int32Array.from(out);
}

/**
 * Training a mini language model: next-token prediction.
 *
 * Every step takes `batch` random windows of `ctx` tokens from the training
 * text. The input is the window, the target is the same window shifted by
 * one token: at every position the model must predict what comes next.
 *
 *   loss = mean over positions of −log p(true next token)
 *
 * Optimiser: AdamW (Adam with decoupled weight decay), learning-rate warm-up
 * followed by a cosine decay, and gradient clipping, like real LLM training.
 */
import { RNG } from '../ml/rng.ts';
import type { LineMode } from './decode.ts';
import { GPT } from './gpt.ts';

export interface LLMTrainConfig {
  lr: number;
  /** steps of linear learning-rate warm-up */
  warmup: number;
  schedule: 'cosine' | 'constant';
  /** final learning rate, as a fraction of lr */
  minLr: number;
  batch: number;
  weightDecay: number;
  /** gradient clipping (max global norm; 0 = off) */
  clip: number;
  /** stop after this many steps (0 = never) */
  steps: number;
  /** measure the validation loss every N steps */
  evalEvery: number;
  seed: number;
}

export const DEFAULT_TRAIN: LLMTrainConfig = {
  lr: 3e-3,
  warmup: 100,
  schedule: 'cosine',
  minLr: 0.1,
  batch: 12,
  weightDecay: 0.01,
  clip: 1,
  steps: 4000,
  evalEvery: 100,
  seed: 1,
};

export class LLMTrainer {
  readonly model: GPT;
  readonly train: Int32Array;
  readonly val: Int32Array;
  readonly cfg: LLMTrainConfig;
  private rng: RNG;
  private m: Float32Array[];
  private v: Float32Array[];
  private x: Int32Array;
  private y: Int32Array;
  /** where training / validation windows may start (aligned to line starts), or null = anywhere */
  private trainStarts: Int32Array | null;
  private valStarts: Int32Array | null;
  step = 0;
  tokensSeen = 0;
  lastLoss = NaN;
  lastGradNorm = NaN;
  lr = 0;
  private beta1 = 0.9;
  private beta2 = 0.95;

  constructor(model: GPT, train: Int32Array, val: Int32Array, cfg: LLMTrainConfig, starts: { train: Int32Array; val: Int32Array } | null = null) {
    this.model = model;
    this.train = train;
    this.val = val;
    this.cfg = { ...cfg };
    const T = model.cfg.ctx;
    if (train.length < T + 2) throw new Error(`The training text has only ${train.length} tokens: it needs more than the context length (${T}). Use more text or a shorter context.`);
    this.rng = new RNG(cfg.seed * 13 + 5);
    this.m = model.params.map((p) => new Float32Array(p.w.length));
    this.v = model.params.map((p) => new Float32Array(p.w.length));
    this.x = new Int32Array(cfg.batch * T);
    this.y = new Int32Array(cfg.batch * T);
    this.trainStarts = starts && starts.train.length ? starts.train : null;
    this.valStarts = starts && starts.val.length ? starts.val : null;
  }

  /** Change the settings while training (the seed only matters at the start). */
  setConfig(c: LLMTrainConfig): void {
    const T = this.model.cfg.ctx;
    Object.assign(this.cfg, { ...c, seed: this.cfg.seed });
    if (this.x.length !== c.batch * T) {
      this.x = new Int32Array(c.batch * T);
      this.y = new Int32Array(c.batch * T);
    }
  }

  /** Learning rate at a step: warm-up, then cosine decay to minLr·lr (or constant). */
  lrAt(step: number): number {
    const c = this.cfg;
    if (c.warmup > 0 && step < c.warmup) return (c.lr * (step + 1)) / c.warmup;
    if (c.schedule === 'constant' || c.steps <= c.warmup) return c.lr;
    const p = Math.min(1, (step - c.warmup) / Math.max(1, c.steps - c.warmup));
    return c.lr * (c.minLr + (1 - c.minLr) * 0.5 * (1 + Math.cos(Math.PI * p)));
  }

  private fillBatch(data: Int32Array, B: number, offsets: number[] | null): void {
    const T = this.model.cfg.ctx;
    const starts = data === this.train ? this.trainStarts : null;
    for (let b = 0; b < B; b++) {
      const i = offsets ? offsets[b] : starts ? starts[this.rng.int(starts.length)] : this.rng.int(data.length - T - 1);
      for (let t = 0; t < T; t++) {
        this.x[b * T + t] = data[i + t];
        this.y[b * T + t] = data[i + t + 1];
      }
    }
  }

  /** A random batch of training windows (inputs x and targets y, B·T each). */
  sampleBatch(): { x: Int32Array; y: Int32Array; B: number; T: number } {
    const B = this.cfg.batch;
    this.fillBatch(this.train, B, null);
    return { x: this.x, y: this.y, B, T: this.model.cfg.ctx };
  }

  /**
   * One optimiser step on one random batch, the heart of training:
   *   1. pick B random windows of text (x) and the same windows shifted by one token (y = what comes next)
   *   2. forward: the model's scores for the next token at every position
   *   3. loss: how wrong those scores were, and its gradient w.r.t. the scores
   *   4. backward: the gradient for every weight (chain rule, gpt.ts)
   *   5. update: move every weight a small step downhill (applyUpdate)
   */
  trainStep(): { loss: number; gradNorm: number; lr: number } {
    const model = this.model;
    const { x, y, B, T } = this.sampleBatch(); // 1
    model.forward(x, B, T); // 2
    const loss = model.lossAndGrad(y); // 3
    model.zeroGrad();
    model.backward(); // 4
    return this.applyUpdate(loss, B * T); // 5
  }

  /**
   * Update the weights with the gradient in the model's .g arrays (already
   * averaged over the batch): gradient clipping, then AdamW.
   */
  applyUpdate(loss: number, tokens: number): { loss: number; gradNorm: number; lr: number } {
    const model = this.model;
    // gradient clipping by global norm: the "length" of the whole gradient is √(Σ g²);
    // if it is longer than cfg.clip (a surprising batch), shrink every gradient by the same factor
    let sq = 0;
    for (const p of model.params) for (let i = 0; i < p.g.length; i++) sq += p.g[i] * p.g[i];
    const norm = Math.sqrt(sq);
    const clip = this.cfg.clip > 0 && norm > this.cfg.clip ? this.cfg.clip / (norm + 1e-6) : 1;
    // AdamW: Adam (see ../ml/optim.ts for the idea) with decoupled weight decay
    const lr = this.lrAt(this.step); // learning rate for this step (warm-up, then cosine decay)
    const t = this.step + 1;
    const b1 = this.beta1;
    const b2 = this.beta2;
    const bc1 = 1 - Math.pow(b1, t);
    const bc2 = 1 - Math.pow(b2, t);
    const wd = this.cfg.weightDecay;
    model.params.forEach((p, k) => {
      const m = this.m[k]; // running average of the gradient (direction)
      const v = this.v[k]; // running average of the squared gradient (typical size)
      const w = p.w;
      const g = p.g;
      const decay = p.decay ? wd : 0; // weight decay only for matrices, not biases / LayerNorm gains
      for (let i = 0; i < w.length; i++) {
        const gi = g[i] * clip;
        m[i] = b1 * m[i] + (1 - b1) * gi;
        v[i] = b2 * v[i] + (1 - b2) * gi * gi;
        // Adam step + weight decay: also shrink every weight a tiny bit towards 0, which keeps
        // weights small unless the data really needs them big (a mild guard against memorising)
        w[i] -= lr * ((m[i] / bc1) / (Math.sqrt(v[i] / bc2) + 1e-8) + decay * w[i]);
      }
    });
    this.step++;
    this.tokensSeen += tokens;
    this.lastLoss = loss;
    this.lastGradNorm = norm;
    this.lr = lr;
    return { loss, gradNorm: norm, lr };
  }

  /** Mean loss on `batches` fixed windows spread over the validation text (NaN if it is too short). */
  evaluate(batches = 4): number {
    const T = this.model.cfg.ctx;
    const B = this.cfg.batch;
    if (this.val.length < T + 2) return NaN;
    const span = this.val.length - T - 1;
    const vs = this.valStarts;
    let total = 0;
    for (let k = 0; k < batches; k++) {
      const frac = (b: number) => (k * B + b + 0.5) / (batches * B);
      const offs = Array.from({ length: B }, (_, b) => (vs ? vs[Math.floor(frac(b) * vs.length)] : Math.floor(frac(b) * span)));
      this.fillBatch(this.val, B, offs);
      this.model.forward(this.x, B, T);
      total += this.model.loss(this.y);
    }
    return total / batches;
  }
}
