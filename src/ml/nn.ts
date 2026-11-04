/**
 * A tiny, readable neural-network library. No magic, no autograd graph:
 * every layer has an explicit `forward` (compute outputs) and `backward`
 * (compute gradients with the chain rule).
 *
 *   z = W·x + b          (pre-activation, one number per neuron)
 *   a = f(z)             (activation function, e.g. ReLU)
 *
 * Backward, given dL/da coming from the layer above:
 *   dL/dz = dL/da ⊙ f'(z)
 *   dL/dW = dL/dz · xᵀ   (accumulated over the batch)
 *   dL/db = dL/dz
 *   dL/dx = Wᵀ · dL/dz   (passed down to the previous layer)
 *
 * Weights are stored row-major as [outputs × inputs], exactly like
 * PyTorch's nn.Linear.weight, which makes exporting trivial.
 */
import { RNG } from './rng.ts';

export type Activation =
  | 'relu'
  | 'leaky_relu'
  | 'tanh'
  | 'sigmoid'
  | 'elu'
  | 'swish'
  | 'softplus'
  | 'linear';

export const ACTIVATIONS: Activation[] = [
  'relu',
  'leaky_relu',
  'tanh',
  'sigmoid',
  'elu',
  'swish',
  'softplus',
  'linear',
];

export const ACTIVATION_INFO: Record<Activation, { formula: string; blurb: string }> = {
  relu: { formula: 'max(0, z)', blurb: 'Passes positives, blocks negatives. Fast and the most popular choice.' },
  leaky_relu: { formula: 'z > 0 ? z : 0.01·z', blurb: 'Like ReLU but lets a trickle of negatives through, so neurons never fully "die".' },
  tanh: { formula: '(eᶻ − e⁻ᶻ)/(eᶻ + e⁻ᶻ)', blurb: 'Squashes to (−1, 1). Smooth and zero-centred; classic for RL policies.' },
  sigmoid: { formula: '1 / (1 + e⁻ᶻ)', blurb: 'Squashes to (0, 1). Good for probabilities, slow to train in deep nets.' },
  elu: { formula: 'z > 0 ? z : eᶻ − 1', blurb: 'Smooth ReLU variant with negative outputs that saturate at −1.' },
  swish: { formula: 'z · sigmoid(z)', blurb: 'Smooth, self-gated ReLU (a.k.a. SiLU), used in many modern nets.' },
  softplus: { formula: 'log(1 + eᶻ)', blurb: 'A smooth version of ReLU.' },
  linear: { formula: 'z', blurb: 'No activation. Used for output layers that predict any real number.' },
};

export interface LayerSpec {
  units: number;
  activation: Activation;
}

export interface ParamRef {
  w: Float32Array; // the parameter values
  g: Float32Array; // the accumulated gradient dL/dw
  name: string;
}

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

export function applyActivation(act: Activation, z: Float32Array, a: Float32Array, n: number): void {
  switch (act) {
    case 'relu':
      for (let i = 0; i < n; i++) a[i] = z[i] > 0 ? z[i] : 0;
      break;
    case 'leaky_relu':
      for (let i = 0; i < n; i++) a[i] = z[i] > 0 ? z[i] : 0.01 * z[i];
      break;
    case 'tanh':
      for (let i = 0; i < n; i++) a[i] = Math.tanh(z[i]);
      break;
    case 'sigmoid':
      for (let i = 0; i < n; i++) a[i] = sigmoid(z[i]);
      break;
    case 'elu':
      for (let i = 0; i < n; i++) a[i] = z[i] > 0 ? z[i] : Math.exp(z[i]) - 1;
      break;
    case 'swish':
      for (let i = 0; i < n; i++) a[i] = z[i] * sigmoid(z[i]);
      break;
    case 'softplus':
      for (let i = 0; i < n; i++) {
        const v = z[i];
        a[i] = v > 20 ? v : Math.log1p(Math.exp(v));
      }
      break;
    default:
      for (let i = 0; i < n; i++) a[i] = z[i];
  }
}

/** dZ = dA ⊙ f'(z). Uses whichever of z (pre) or a (post) gives the cheapest derivative. */
export function activationBackward(
  act: Activation,
  z: Float32Array,
  a: Float32Array,
  dA: Float32Array,
  dZ: Float32Array,
  n: number,
): void {
  switch (act) {
    case 'relu':
      for (let i = 0; i < n; i++) dZ[i] = z[i] > 0 ? dA[i] : 0;
      break;
    case 'leaky_relu':
      for (let i = 0; i < n; i++) dZ[i] = z[i] > 0 ? dA[i] : 0.01 * dA[i];
      break;
    case 'tanh':
      for (let i = 0; i < n; i++) dZ[i] = dA[i] * (1 - a[i] * a[i]);
      break;
    case 'sigmoid':
      for (let i = 0; i < n; i++) dZ[i] = dA[i] * a[i] * (1 - a[i]);
      break;
    case 'elu':
      for (let i = 0; i < n; i++) dZ[i] = z[i] > 0 ? dA[i] : dA[i] * (a[i] + 1);
      break;
    case 'swish':
      for (let i = 0; i < n; i++) {
        const s = sigmoid(z[i]);
        dZ[i] = dA[i] * (s + z[i] * s * (1 - s));
      }
      break;
    case 'softplus':
      for (let i = 0; i < n; i++) dZ[i] = dA[i] * sigmoid(z[i]);
      break;
    default:
      for (let i = 0; i < n; i++) dZ[i] = dA[i];
  }
}

/** A fully-connected ("dense") layer: every input connects to every output. */
export class Dense {
  readonly inSize: number;
  readonly outSize: number;
  act: Activation;
  W: Float32Array;
  b: Float32Array;
  gW: Float32Array;
  gb: Float32Array;

  // Buffers reused between calls (sized for the largest batch seen so far).
  private cap = 0;
  private xRef: Float32Array = new Float32Array(0);
  z: Float32Array = new Float32Array(0);
  a: Float32Array = new Float32Array(0);
  private dZ: Float32Array = new Float32Array(0);
  private dX: Float32Array = new Float32Array(0);

  constructor(inSize: number, outSize: number, act: Activation, rng: RNG, initScale = 1) {
    this.inSize = inSize;
    this.outSize = outSize;
    this.act = act;
    this.W = new Float32Array(inSize * outSize);
    this.b = new Float32Array(outSize);
    this.gW = new Float32Array(inSize * outSize);
    this.gb = new Float32Array(outSize);
    this.initWeights(rng, initScale);
  }

  /**
   * He init for ReLU-family activations (keeps signal variance ≈ constant
   * when half the units are zeroed), Glorot/Xavier for the rest.
   */
  initWeights(rng: RNG, scale = 1): void {
    const reluLike = this.act === 'relu' || this.act === 'leaky_relu' || this.act === 'elu' || this.act === 'swish';
    const std = reluLike ? Math.sqrt(2 / this.inSize) : Math.sqrt(2 / (this.inSize + this.outSize));
    for (let i = 0; i < this.W.length; i++) this.W[i] = rng.gauss() * std * scale;
    this.b.fill(0);
  }

  private ensure(batch: number): void {
    if (batch <= this.cap) return;
    this.cap = batch;
    this.z = new Float32Array(batch * this.outSize);
    this.a = new Float32Array(batch * this.outSize);
    this.dZ = new Float32Array(batch * this.outSize);
    this.dX = new Float32Array(batch * this.inSize);
  }

  forward(x: Float32Array, batch: number): Float32Array {
    this.ensure(batch);
    this.xRef = x;
    const { W, b, z, inSize: I, outSize: O } = this;
    for (let n = 0; n < batch; n++) {
      const xo = n * I;
      const zo = n * O;
      for (let o = 0; o < O; o++) {
        let s = b[o];
        const wo = o * I;
        for (let i = 0; i < I; i++) s += W[wo + i] * x[xo + i];
        z[zo + o] = s;
      }
    }
    applyActivation(this.act, z, this.a, batch * O);
    return this.a;
  }

  /** Accumulates gW/gb. Returns dL/dx (or an empty array when not needed). */
  backward(dA: Float32Array, batch: number, needInputGrad: boolean): Float32Array {
    const { W, gW, gb, z, a, dZ, dX, inSize: I, outSize: O } = this;
    const x = this.xRef;
    activationBackward(this.act, z, a, dA, dZ, batch * O);
    for (let n = 0; n < batch; n++) {
      const xo = n * I;
      const zo = n * O;
      for (let o = 0; o < O; o++) {
        const dz = dZ[zo + o];
        if (dz === 0) continue;
        gb[o] += dz;
        const wo = o * I;
        for (let i = 0; i < I; i++) gW[wo + i] += dz * x[xo + i];
      }
    }
    if (!needInputGrad) return dX;
    dX.fill(0, 0, batch * I);
    for (let n = 0; n < batch; n++) {
      const xo = n * I;
      const zo = n * O;
      for (let o = 0; o < O; o++) {
        const dz = dZ[zo + o];
        if (dz === 0) continue;
        const wo = o * I;
        for (let i = 0; i < I; i++) dX[xo + i] += dz * W[wo + i];
      }
    }
    return dX;
  }

  zeroGrad(): void {
    this.gW.fill(0);
    this.gb.fill(0);
  }
}

export interface MLPJSON {
  inSize: number;
  layers: { inSize: number; outSize: number; activation: Activation; W: number[]; b: number[] }[];
}

/** Multi-layer perceptron: a stack of Dense layers. */
export class MLP {
  layers: Dense[];
  readonly inSize: number;
  readonly outSize: number;

  constructor(inSize: number, hidden: LayerSpec[], outSize: number, outAct: Activation, rng: RNG, outInitScale = 1) {
    this.inSize = inSize;
    this.outSize = outSize;
    this.layers = [];
    let prev = inSize;
    for (const h of hidden) {
      this.layers.push(new Dense(prev, h.units, h.activation, rng));
      prev = h.units;
    }
    this.layers.push(new Dense(prev, outSize, outAct, rng, outInitScale));
  }

  forward(x: Float32Array, batch: number): Float32Array {
    let h = x;
    for (const l of this.layers) h = l.forward(h, batch);
    return h;
  }

  backward(dOut: Float32Array, batch: number, needInputGrad = false): Float32Array {
    let g = dOut;
    for (let i = this.layers.length - 1; i >= 0; i--) {
      g = this.layers[i].backward(g, batch, i > 0 || needInputGrad);
    }
    return g;
  }

  zeroGrad(): void {
    for (const l of this.layers) l.zeroGrad();
  }

  params(prefix = ''): ParamRef[] {
    const out: ParamRef[] = [];
    this.layers.forEach((l, i) => {
      out.push({ w: l.W, g: l.gW, name: `${prefix}layer${i}.weight` });
      out.push({ w: l.b, g: l.gb, name: `${prefix}layer${i}.bias` });
    });
    return out;
  }

  numParams(): number {
    let n = 0;
    for (const l of this.layers) n += l.W.length + l.b.length;
    return n;
  }

  copyFrom(other: MLP): void {
    for (let i = 0; i < this.layers.length; i++) {
      this.layers[i].W.set(other.layers[i].W);
      this.layers[i].b.set(other.layers[i].b);
    }
  }

  clone(): MLP {
    const m = MLP.fromJSON(this.toJSON());
    return m;
  }

  /** All parameters as one flat vector (used by neuroevolution). */
  getFlat(out?: Float32Array): Float32Array {
    const f = out ?? new Float32Array(this.numParams());
    let k = 0;
    for (const l of this.layers) {
      f.set(l.W, k);
      k += l.W.length;
      f.set(l.b, k);
      k += l.b.length;
    }
    return f;
  }

  setFlat(f: Float32Array): void {
    let k = 0;
    for (const l of this.layers) {
      l.W.set(f.subarray(k, k + l.W.length));
      k += l.W.length;
      l.b.set(f.subarray(k, k + l.b.length));
      k += l.b.length;
    }
  }

  /**
   * Single-sample forward pass that records every layer's pre- and
   * post-activation values. Used by the live brain visualiser.
   */
  trace(x: Float32Array): { pre: Float32Array[]; post: Float32Array[] } {
    const pre: Float32Array[] = [];
    const post: Float32Array[] = [Float32Array.from(x)];
    let h = x;
    for (const l of this.layers) {
      h = l.forward(h, 1);
      pre.push(Float32Array.from(l.z.subarray(0, l.outSize)));
      post.push(Float32Array.from(h.subarray(0, l.outSize)));
    }
    return { pre, post };
  }

  architecture(): { inSize: number; layers: LayerSpec[] } {
    return { inSize: this.inSize, layers: this.layers.map((l) => ({ units: l.outSize, activation: l.act })) };
  }

  toJSON(): MLPJSON {
    return {
      inSize: this.inSize,
      layers: this.layers.map((l) => ({
        inSize: l.inSize,
        outSize: l.outSize,
        activation: l.act,
        W: Array.from(l.W),
        b: Array.from(l.b),
      })),
    };
  }

  static fromJSON(j: MLPJSON): MLP {
    const rng = new RNG(1);
    const hidden = j.layers.slice(0, -1).map((l) => ({ units: l.outSize, activation: l.activation }));
    const last = j.layers[j.layers.length - 1];
    const m = new MLP(j.inSize, hidden, last.outSize, last.activation, rng);
    j.layers.forEach((l, i) => {
      m.layers[i].W.set(l.W);
      m.layers[i].b.set(l.b);
    });
    return m;
  }

  /** Same shapes as another network? (needed before copying weights) */
  sameShape(j: MLPJSON): boolean {
    if (j.inSize !== this.inSize || j.layers.length !== this.layers.length) return false;
    return j.layers.every((l, i) => l.outSize === this.layers[i].outSize && l.inSize === this.layers[i].inSize);
  }
}
