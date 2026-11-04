/**
 * ===========================================================================
 *  optim.ts: optimisers turn gradients into weight updates
 * ===========================================================================
 *
 * After backward() (nn.ts), every weight w has a gradient g = dLoss/dw:
 * "if this weight grew a little, the loss would grow by g times as much".
 * To make the loss smaller, move each weight a small step the OTHER way.
 * That step is all an optimiser does. They differ in how they size it:
 *
 *  SGD:      w ← w − lr·g
 *            plain "walk downhill". lr = learning rate (step size), e.g. 0.01.
 *  Momentum: v ← μ·v + g ;  w ← w − lr·v
 *            keep some speed from earlier steps (like a ball rolling downhill):
 *            it rolls through small bumps and speeds up on long slopes.
 *  RMSProp:  s ← ρ·s + (1−ρ)·g² ; w ← w − lr·g/√(s+ε)
 *            per-weight step sizes: weights whose gradients are usually big
 *            take smaller steps, weights with tiny gradients take bigger ones.
 *  Adam:     momentum + RMSProp + bias correction (the default almost everywhere)
 *
 * Example with SGD, lr = 0.1: a weight 0.5 with gradient 2 becomes
 * 0.5 − 0.1·2 = 0.3. A gradient of −2 would push it UP to 0.7 instead.
 */
import type { ParamRef } from './nn.ts';

export type OptimizerKind = 'adam' | 'sgd' | 'momentum' | 'rmsprop';

export interface Optimizer {
  lr: number;
  /** Update every weight once, using the gradients currently stored in each ParamRef.g */
  step(): void;
  zeroGrad(): void;
  /** Size of the last update relative to the weights (a health metric). */
  lastUpdateRatio: number;
  params: ParamRef[];
}

/** What all optimisers share: the list of parameters and the learning rate. */
abstract class BaseOpt implements Optimizer {
  lr: number;
  params: ParamRef[];
  // ‖update‖ / ‖weights‖ of the last step. Around 0.001 is healthy; much bigger means
  // the steps are violent (learning rate too high), much smaller means almost frozen.
  lastUpdateRatio = 0;
  constructor(params: ParamRef[], lr: number) {
    this.params = params;
    this.lr = lr;
  }
  abstract step(): void;
  /** Gradients ACCUMULATE in backward(): clear them before computing new ones. */
  zeroGrad(): void {
    for (const p of this.params) p.g.fill(0);
  }
}

/** Stochastic gradient descent: w ← w − lr·g. */
export class SGD extends BaseOpt {
  step(): void {
    let upd = 0; // Σ (size of each change)², for the health metric
    let wsum = 0; // Σ weight², for the health metric
    for (const p of this.params) {
      const { w, g } = p;
      for (let i = 0; i < w.length; i++) {
        const d = this.lr * g[i]; // the step: proportional to the slope
        w[i] -= d; // downhill
        upd += d * d;
        wsum += w[i] * w[i];
      }
    }
    this.lastUpdateRatio = Math.sqrt(upd / (wsum + 1e-12));
  }
}

/** SGD with momentum: steps keep a share μ (e.g. 90%) of the previous step's direction. */
export class Momentum extends BaseOpt {
  mu: number;
  private v: Float32Array[]; // the "velocity" of every weight
  constructor(params: ParamRef[], lr: number, mu = 0.9) {
    super(params, lr);
    this.mu = mu;
    this.v = params.map((p) => new Float32Array(p.w.length));
  }
  step(): void {
    let upd = 0;
    let wsum = 0;
    this.params.forEach((p, k) => {
      const v = this.v[k];
      const { w, g } = p;
      for (let i = 0; i < w.length; i++) {
        v[i] = this.mu * v[i] + g[i]; // keep 90% of the old velocity, add the new slope
        const d = this.lr * v[i];
        w[i] -= d;
        upd += d * d;
        wsum += w[i] * w[i];
      }
    });
    this.lastUpdateRatio = Math.sqrt(upd / (wsum + 1e-12));
  }
}

/** RMSProp: divide each step by the typical size of that weight's recent gradients. */
export class RMSProp extends BaseOpt {
  rho: number;
  eps: number;
  private s: Float32Array[]; // running average of g² for every weight
  constructor(params: ParamRef[], lr: number, rho = 0.99, eps = 1e-8) {
    super(params, lr);
    this.rho = rho;
    this.eps = eps;
    this.s = params.map((p) => new Float32Array(p.w.length));
  }
  step(): void {
    let upd = 0;
    let wsum = 0;
    this.params.forEach((p, k) => {
      const s = this.s[k];
      const { w, g } = p;
      for (let i = 0; i < w.length; i++) {
        s[i] = this.rho * s[i] + (1 - this.rho) * g[i] * g[i]; // how big are this weight's gradients lately?
        const d = (this.lr * g[i]) / (Math.sqrt(s[i]) + this.eps); // normalised step (ε avoids ÷ 0)
        w[i] -= d;
        upd += d * d;
        wsum += w[i] * w[i];
      }
    });
    this.lastUpdateRatio = Math.sqrt(upd / (wsum + 1e-12));
  }
}

/**
 * Adam = momentum (m: running average of the gradient, the direction) +
 * RMSProp (v: running average of the squared gradient, the typical size).
 * Every weight effectively gets its own learning rate. In practice the step
 * of each weight is about lr in size, in the direction gradients usually point.
 */
export class Adam extends BaseOpt {
  beta1: number; // how much of the old direction m to keep each step (0.9)
  beta2: number; // how much of the old size v to keep each step (0.999)
  eps: number;
  t = 0; // number of steps taken so far
  private m: Float32Array[];
  private v: Float32Array[];
  constructor(params: ParamRef[], lr: number, beta1 = 0.9, beta2 = 0.999, eps = 1e-8) {
    super(params, lr);
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.eps = eps;
    this.m = params.map((p) => new Float32Array(p.w.length));
    this.v = params.map((p) => new Float32Array(p.w.length));
  }
  step(): void {
    this.t++;
    const { beta1: b1, beta2: b2, eps, lr } = this;
    // bias correction: m and v start at 0, so for the first few steps they are
    // too small (only one gradient was averaged with a lot of zeros). Dividing by
    // 1 − β^t scales them back up; after a few hundred steps it is ≈ 1 and does nothing.
    const bc1 = 1 - Math.pow(b1, this.t);
    const bc2 = 1 - Math.pow(b2, this.t);
    let upd = 0;
    let wsum = 0;
    this.params.forEach((p, k) => {
      const m = this.m[k];
      const v = this.v[k];
      const { w, g } = p;
      for (let i = 0; i < w.length; i++) {
        const gi = g[i];
        m[i] = b1 * m[i] + (1 - b1) * gi; // average direction
        v[i] = b2 * v[i] + (1 - b2) * gi * gi; // average squared size
        // step = lr × (direction) / (typical size): about ±lr when the gradient is consistent
        const d = (lr * (m[i] / bc1)) / (Math.sqrt(v[i] / bc2) + eps);
        w[i] -= d;
        upd += d * d;
        wsum += w[i] * w[i];
      }
    });
    this.lastUpdateRatio = Math.sqrt(upd / (wsum + 1e-12));
  }
}

export function makeOptimizer(kind: OptimizerKind, params: ParamRef[], lr: number): Optimizer {
  switch (kind) {
    case 'sgd':
      return new SGD(params, lr);
    case 'momentum':
      return new Momentum(params, lr);
    case 'rmsprop':
      return new RMSProp(params, lr);
    default:
      return new Adam(params, lr);
  }
}

/** Global L2 norm of all gradients: √(sum of every gradient²), the "length" of the whole gradient. */
export function gradNorm(params: ParamRef[]): number {
  let s = 0;
  for (const p of params) for (let i = 0; i < p.g.length; i++) s += p.g[i] * p.g[i];
  return Math.sqrt(s);
}

/**
 * Gradient clipping: if the gradient vector is longer than maxNorm, shrink it.
 * Prevents one weird batch from blowing up the network. Returns the norm before clipping.
 * (Shrinking every gradient by the same factor keeps the direction, only the size changes.)
 */
export function clipGradNorm(params: ParamRef[], maxNorm: number): number {
  const norm = gradNorm(params);
  if (maxNorm > 0 && norm > maxNorm) {
    const k = maxNorm / (norm + 1e-6);
    for (const p of params) for (let i = 0; i < p.g.length; i++) p.g[i] *= k;
  }
  return norm;
}
