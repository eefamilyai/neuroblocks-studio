/**
 * Running mean / variance (the parallel version of Welford's algorithm).
 *
 * Neural nets learn best when inputs are roughly "mean 0, std 1". We don't
 * know the statistics of observations in advance, so we track them as the
 * agent explores and normalise on the fly:   x̂ = clip((x − μ) / σ, ±5)
 *
 * The same object, applied to discounted returns, scales rewards (PPO trick).
 */
export class RunningMeanStd {
  readonly size: number;
  mean: Float64Array;
  var: Float64Array;
  count: number;
  clip: number;

  constructor(size: number, clip = 5) {
    this.size = size;
    this.mean = new Float64Array(size);
    this.var = new Float64Array(size).fill(1);
    this.count = 1e-4;
    this.clip = clip;
  }

  /** Update with a batch of rows x[batch × size]. */
  update(x: Float32Array, batch: number): void {
    const n = this.size;
    const bMean = new Float64Array(n);
    const bVar = new Float64Array(n);
    for (let r = 0; r < batch; r++) for (let i = 0; i < n; i++) bMean[i] += x[r * n + i];
    for (let i = 0; i < n; i++) bMean[i] /= batch;
    for (let r = 0; r < batch; r++)
      for (let i = 0; i < n; i++) {
        const d = x[r * n + i] - bMean[i];
        bVar[i] += d * d;
      }
    for (let i = 0; i < n; i++) bVar[i] /= batch;
    const tot = this.count + batch;
    for (let i = 0; i < n; i++) {
      const delta = bMean[i] - this.mean[i];
      const newMean = this.mean[i] + (delta * batch) / tot;
      const m2 = this.var[i] * this.count + bVar[i] * batch + (delta * delta * this.count * batch) / tot;
      this.mean[i] = newMean;
      this.var[i] = m2 / tot;
    }
    this.count = tot;
  }

  /** out = clip((x − mean) / sqrt(var + ε)) */
  normalize(x: Float32Array, out: Float32Array, batch: number): void {
    const n = this.size;
    const c = this.clip;
    for (let r = 0; r < batch; r++)
      for (let i = 0; i < n; i++) {
        let v = (x[r * n + i] - this.mean[i]) / Math.sqrt(this.var[i] + 1e-8);
        if (v > c) v = c;
        else if (v < -c) v = -c;
        out[r * n + i] = v;
      }
  }

  toJSON(): { mean: number[]; var: number[]; count: number; clip: number } {
    return { mean: Array.from(this.mean), var: Array.from(this.var), count: this.count, clip: this.clip };
  }

  static fromJSON(j: { mean: number[]; var: number[]; count: number; clip?: number }): RunningMeanStd {
    const r = new RunningMeanStd(j.mean.length, j.clip ?? 5);
    r.mean.set(j.mean);
    r.var.set(j.var);
    r.count = j.count;
    return r;
  }
}

/** Identity normaliser with the same interface, for when normalisation is off. */
export function identityNormalizer(size: number): RunningMeanStd {
  const r = new RunningMeanStd(size, 1e9);
  r.count = 1e12;
  return r;
}
