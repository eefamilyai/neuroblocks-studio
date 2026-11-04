/**
 * Turning network outputs into actions.
 *
 * DISCRETE ("pick one option per control"): the network outputs one score
 * (a *logit*) per option. Softmax turns scores into probabilities:
 *     p_k = e^{z_k} / Σ_j e^{z_j}
 * With several controls (e.g. "move" and "turn") we use one softmax per
 * control, called a *branch*. That's a multi-discrete action space.
 *
 * CONTINUOUS ("how much"): the network outputs a mean μ per control, and a
 * learnable log standard deviation sets how much noise is added while
 * exploring:  a = μ + σ·ε,  ε ~ N(0, 1).
 */
import { RNG } from './rng.ts';

export interface ActionSpace {
  kind: 'discrete' | 'continuous';
  /** discrete: number of options per branch. continuous: one entry (=1) per dimension. */
  branches: number[];
  /** name of each branch / dimension */
  labels: string[];
  /** discrete: option names per branch */
  optionLabels: string[][];
}

const LOG_2PI = Math.log(2 * Math.PI);

export function policyOutputSize(space: ActionSpace): number {
  return space.kind === 'discrete' ? space.branches.reduce((a, b) => a + b, 0) : space.branches.length;
}

export function actionDims(space: ActionSpace): number {
  return space.branches.length;
}

/** Joint (flattened) number of discrete actions, used by DQN. */
export function jointActionCount(space: ActionSpace): number {
  return space.branches.reduce((a, b) => a * b, 1);
}

/** joint index -> per-branch option indices */
export function decodeJoint(space: ActionSpace, idx: number, out: Float32Array, off = 0): void {
  for (let b = space.branches.length - 1; b >= 0; b--) {
    const n = space.branches[b];
    out[off + b] = idx % n;
    idx = Math.floor(idx / n);
  }
}

export function encodeJoint(space: ActionSpace, act: ArrayLike<number>, off = 0): number {
  let idx = 0;
  for (let b = 0; b < space.branches.length; b++) idx = idx * space.branches[b] + (act[off + b] | 0);
  return idx;
}

/** Numerically stable softmax of x[off..off+n) into out[outOff..]. */
export function softmax(x: ArrayLike<number>, off: number, n: number, out: Float32Array | number[], outOff = 0): void {
  let max = -Infinity;
  for (let i = 0; i < n; i++) if (x[off + i] > max) max = x[off + i];
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const e = Math.exp(x[off + i] - max);
    out[outOff + i] = e;
    sum += e;
  }
  for (let i = 0; i < n; i++) out[outOff + i] /= sum;
}

const scratch = new Float32Array(256);

export class PolicyHead {
  readonly space: ActionSpace;
  readonly outSize: number;
  readonly dims: number;

  constructor(space: ActionSpace) {
    this.space = space;
    this.outSize = policyOutputSize(space);
    this.dims = actionDims(space);
  }

  /**
   * Sample an action for one row of network output. Writes the action into
   * act[actOff..] and returns its log-probability.
   */
  sample(
    out: Float32Array,
    off: number,
    logStd: Float32Array | null,
    rng: RNG,
    act: Float32Array,
    actOff: number,
    deterministic: boolean,
  ): number {
    const sp = this.space;
    if (sp.kind === 'discrete') {
      let lp = 0;
      let o = off;
      for (let b = 0; b < sp.branches.length; b++) {
        const n = sp.branches[b];
        softmax(out, o, n, scratch, 0);
        let k = 0;
        if (deterministic) {
          for (let i = 1; i < n; i++) if (scratch[i] > scratch[k]) k = i;
        } else {
          let r = rng.next();
          k = n - 1;
          for (let i = 0; i < n; i++) {
            r -= scratch[i];
            if (r <= 0) {
              k = i;
              break;
            }
          }
        }
        act[actOff + b] = k;
        lp += Math.log(Math.max(scratch[k], 1e-12));
        o += n;
      }
      return lp;
    }
    let lp = 0;
    for (let d = 0; d < this.dims; d++) {
      const mu = out[off + d];
      const ls = logStd ? logStd[d] : 0;
      const std = Math.exp(ls);
      const eps = deterministic ? 0 : rng.gauss();
      const a = mu + std * eps;
      act[actOff + d] = a;
      lp += -0.5 * eps * eps - ls - 0.5 * LOG_2PI;
    }
    return lp;
  }

  logProb(out: Float32Array, off: number, logStd: Float32Array | null, act: Float32Array, actOff: number): number {
    const sp = this.space;
    if (sp.kind === 'discrete') {
      let lp = 0;
      let o = off;
      for (let b = 0; b < sp.branches.length; b++) {
        const n = sp.branches[b];
        softmax(out, o, n, scratch, 0);
        lp += Math.log(Math.max(scratch[act[actOff + b] | 0], 1e-12));
        o += n;
      }
      return lp;
    }
    let lp = 0;
    for (let d = 0; d < this.dims; d++) {
      const ls = logStd ? logStd[d] : 0;
      const z = (act[actOff + d] - out[off + d]) / Math.exp(ls);
      lp += -0.5 * z * z - ls - 0.5 * LOG_2PI;
    }
    return lp;
  }

  entropy(out: Float32Array, off: number, logStd: Float32Array | null): number {
    const sp = this.space;
    if (sp.kind === 'discrete') {
      let h = 0;
      let o = off;
      for (let b = 0; b < sp.branches.length; b++) {
        const n = sp.branches[b];
        softmax(out, o, n, scratch, 0);
        for (let i = 0; i < n; i++) if (scratch[i] > 0) h -= scratch[i] * Math.log(scratch[i]);
        o += n;
      }
      return h;
    }
    let h = 0;
    for (let d = 0; d < this.dims; d++) h += (logStd ? logStd[d] : 0) + 0.5 * (LOG_2PI + 1);
    return h;
  }

  /**
   * Adds  ∂/∂out [ cLogp·logπ(a|s) + cEnt·H(π(·|s)) ]  into dOut (and dLogStd).
   *
   * Discrete:   ∂logπ/∂z_j = 1[j=a] − p_j      ∂H/∂z_j = −p_j (log p_j + H_branch)
   * Gaussian:   ∂logπ/∂μ = (a−μ)/σ²            ∂logπ/∂logσ = ((a−μ)/σ)² − 1
   *             ∂H/∂logσ = 1
   */
  accumGrad(
    out: Float32Array,
    off: number,
    logStd: Float32Array | null,
    act: Float32Array,
    actOff: number,
    cLogp: number,
    cEnt: number,
    dOut: Float32Array,
    dOff: number,
    dLogStd: Float32Array | null,
  ): void {
    const sp = this.space;
    if (sp.kind === 'discrete') {
      let o = off;
      let d = dOff;
      for (let b = 0; b < sp.branches.length; b++) {
        const n = sp.branches[b];
        softmax(out, o, n, scratch, 0);
        const a = act[actOff + b] | 0;
        let h = 0;
        if (cEnt !== 0) for (let i = 0; i < n; i++) if (scratch[i] > 0) h -= scratch[i] * Math.log(scratch[i]);
        for (let i = 0; i < n; i++) {
          const p = scratch[i];
          let g = cLogp * ((i === a ? 1 : 0) - p);
          if (cEnt !== 0 && p > 0) g += cEnt * (-p * (Math.log(p) + h));
          dOut[d + i] += g;
        }
        o += n;
        d += n;
      }
      return;
    }
    for (let k = 0; k < this.dims; k++) {
      const ls = logStd ? logStd[k] : 0;
      const std = Math.exp(ls);
      const diff = act[actOff + k] - out[off + k];
      dOut[dOff + k] += cLogp * (diff / (std * std));
      if (dLogStd) {
        const z = diff / std;
        dLogStd[k] += cLogp * (z * z - 1) + cEnt;
      }
    }
  }

  /** Probabilities for display (discrete) or means (continuous). */
  probs(out: Float32Array, off: number): number[][] {
    const sp = this.space;
    const res: number[][] = [];
    if (sp.kind === 'discrete') {
      let o = off;
      for (const n of sp.branches) {
        const p = new Array<number>(n);
        softmax(out, o, n, p, 0);
        res.push(p);
        o += n;
      }
    } else {
      for (let d = 0; d < this.dims; d++) res.push([out[off + d]]);
    }
    return res;
  }
}
