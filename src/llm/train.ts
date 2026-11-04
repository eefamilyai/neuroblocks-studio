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
  step = 0;
  tokensSeen = 0;
  lastLoss = NaN;
  lastGradNorm = NaN;
  lr = 0;
  private beta1 = 0.9;
  private beta2 = 0.95;

  constructor(model: GPT, train: Int32Array, val: Int32Array, cfg: LLMTrainConfig) {
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
    for (let b = 0; b < B; b++) {
      const i = offsets ? offsets[b] : this.rng.int(data.length - T - 1);
      for (let t = 0; t < T; t++) {
        this.x[b * T + t] = data[i + t];
        this.y[b * T + t] = data[i + t + 1];
      }
    }
  }

  /** One optimiser step on one random batch. */
  trainStep(): { loss: number; gradNorm: number; lr: number } {
    const model = this.model;
    const T = model.cfg.ctx;
    const B = this.cfg.batch;
    this.fillBatch(this.train, B, null);
    model.forward(this.x, B, T);
    const loss = model.lossAndGrad(this.y);
    model.zeroGrad();
    model.backward();
    // gradient clipping by global norm
    let sq = 0;
    for (const p of model.params) for (let i = 0; i < p.g.length; i++) sq += p.g[i] * p.g[i];
    const norm = Math.sqrt(sq);
    const clip = this.cfg.clip > 0 && norm > this.cfg.clip ? this.cfg.clip / (norm + 1e-6) : 1;
    // AdamW
    const lr = this.lrAt(this.step);
    const t = this.step + 1;
    const b1 = this.beta1;
    const b2 = this.beta2;
    const bc1 = 1 - Math.pow(b1, t);
    const bc2 = 1 - Math.pow(b2, t);
    const wd = this.cfg.weightDecay;
    model.params.forEach((p, k) => {
      const m = this.m[k];
      const v = this.v[k];
      const w = p.w;
      const g = p.g;
      const decay = p.decay ? wd : 0;
      for (let i = 0; i < w.length; i++) {
        const gi = g[i] * clip;
        m[i] = b1 * m[i] + (1 - b1) * gi;
        v[i] = b2 * v[i] + (1 - b2) * gi * gi;
        w[i] -= lr * ((m[i] / bc1) / (Math.sqrt(v[i] / bc2) + 1e-8) + decay * w[i]);
      }
    });
    this.step++;
    this.tokensSeen += B * T;
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
    let total = 0;
    for (let k = 0; k < batches; k++) {
      const offs = Array.from({ length: B }, (_, b) => Math.floor(((k * B + b + 0.5) / (batches * B)) * span));
      this.fillBatch(this.val, B, offs);
      this.model.forward(this.x, B, T);
      total += this.model.loss(this.y);
    }
    return total / batches;
  }
}
