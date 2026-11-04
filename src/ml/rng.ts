/**
 * Seeded pseudo-random number generator (mulberry32).
 *
 * Machine learning uses randomness everywhere: initial weights, exploration,
 * shuffling minibatches, mutations. A *seeded* generator makes experiments
 * repeatable: same seed -> same "random" numbers -> same training run.
 */
export class RNG {
  private s: number;
  private spare: number | null = null;

  constructor(seed = 1234567) {
    this.s = (seed | 0) || 1;
  }

  /** Uniform float in [0, 1). */
  next(): number {
    this.s = (this.s + 0x6d2b79f5) | 0;
    let t = this.s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  uniform(a: number, b: number): number {
    return a + (b - a) * this.next();
  }

  /** Integer in [0, n). */
  int(n: number): number {
    return Math.floor(this.next() * n);
  }

  /** Standard normal sample via the Box-Muller transform. */
  gauss(): number {
    if (this.spare !== null) {
      const v = this.spare;
      this.spare = null;
      return v;
    }
    let u = 0;
    let v = 0;
    while (u === 0) u = this.next();
    while (v === 0) v = this.next();
    const mag = Math.sqrt(-2 * Math.log(u));
    this.spare = mag * Math.sin(2 * Math.PI * v);
    return mag * Math.cos(2 * Math.PI * v);
  }

  normal(mean: number, std: number): number {
    return mean + std * this.gauss();
  }

  /** In-place Fisher-Yates shuffle. */
  shuffle<T>(arr: T[] | Int32Array | Uint32Array): void {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = this.int(i + 1);
      const t = arr[i];
      arr[i] = arr[j];
      arr[j] = t;
    }
  }

  /** Derive an independent child generator. */
  fork(): RNG {
    return new RNG(Math.floor(this.next() * 2147483647) + 1);
  }
}
