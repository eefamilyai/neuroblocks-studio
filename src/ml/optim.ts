/**
 * Optimizers turn gradients into weight updates.
 *
 *  SGD:      w ← w − lr·g
 *  Momentum: v ← μ·v + g ;  w ← w − lr·v           (rolls through small bumps)
 *  RMSProp:  s ← ρ·s + (1−ρ)·g² ; w ← w − lr·g/√(s+ε)   (per-weight step sizes)
 *  Adam:     momentum + RMSProp + bias correction (the default almost everywhere)
 */
import type { ParamRef } from './nn.ts';

export type OptimizerKind = 'adam' | 'sgd' | 'momentum' | 'rmsprop';

export interface Optimizer {
  lr: number;
  step(): void;
  zeroGrad(): void;
  /** Size of the last update relative to the weights (a health metric). */
  lastUpdateRatio: number;
  params: ParamRef[];
}

abstract class BaseOpt implements Optimizer {
  lr: number;
  params: ParamRef[];
  lastUpdateRatio = 0;
  constructor(params: ParamRef[], lr: number) {
    this.params = params;
    this.lr = lr;
  }
  abstract step(): void;
  zeroGrad(): void {
    for (const p of this.params) p.g.fill(0);
  }
}

export class SGD extends BaseOpt {
  step(): void {
    let upd = 0;
    let wsum = 0;
    for (const p of this.params) {
      const { w, g } = p;
      for (let i = 0; i < w.length; i++) {
        const d = this.lr * g[i];
        w[i] -= d;
        upd += d * d;
        wsum += w[i] * w[i];
      }
    }
    this.lastUpdateRatio = Math.sqrt(upd / (wsum + 1e-12));
  }
}

export class Momentum extends BaseOpt {
  mu: number;
  private v: Float32Array[];
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
        v[i] = this.mu * v[i] + g[i];
        const d = this.lr * v[i];
        w[i] -= d;
        upd += d * d;
        wsum += w[i] * w[i];
      }
    });
    this.lastUpdateRatio = Math.sqrt(upd / (wsum + 1e-12));
  }
}

export class RMSProp extends BaseOpt {
  rho: number;
  eps: number;
  private s: Float32Array[];
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
        s[i] = this.rho * s[i] + (1 - this.rho) * g[i] * g[i];
        const d = (this.lr * g[i]) / (Math.sqrt(s[i]) + this.eps);
        w[i] -= d;
        upd += d * d;
        wsum += w[i] * w[i];
      }
    });
    this.lastUpdateRatio = Math.sqrt(upd / (wsum + 1e-12));
  }
}

export class Adam extends BaseOpt {
  beta1: number;
  beta2: number;
  eps: number;
  t = 0;
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
        m[i] = b1 * m[i] + (1 - b1) * gi;
        v[i] = b2 * v[i] + (1 - b2) * gi * gi;
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

/** Global L2 norm of all gradients. */
export function gradNorm(params: ParamRef[]): number {
  let s = 0;
  for (const p of params) for (let i = 0; i < p.g.length; i++) s += p.g[i] * p.g[i];
  return Math.sqrt(s);
}

/**
 * Gradient clipping: if the gradient vector is longer than maxNorm, shrink it.
 * Prevents one weird batch from blowing up the network. Returns the norm before clipping.
 */
export function clipGradNorm(params: ParamRef[], maxNorm: number): number {
  const norm = gradNorm(params);
  if (maxNorm > 0 && norm > maxNorm) {
    const k = maxNorm / (norm + 1e-6);
    for (const p of params) for (let i = 0; i < p.g.length; i++) p.g[i] *= k;
  }
  return norm;
}
