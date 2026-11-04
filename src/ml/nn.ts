/**
 * ===========================================================================
 *  nn.ts: the neural network that is the robot's brain
 * ===========================================================================
 *
 * A tiny, readable neural-network library. No magic, no autograd graph:
 * every layer has an explicit `forward` (compute outputs) and `backward`
 * (compute gradients with the chain rule).
 *
 * THE BIG PICTURE (how numbers flow through the weights)
 * ------------------------------------------------------
 * The robot's senses are a list of numbers, e.g. [distance to flag, angle to
 * flag, my speed, …]. The brain turns that list into a few output numbers
 * (scores for "go left / straight / right", …). It does it in LAYERS:
 *
 *      senses ──► layer 1 ──► layer 2 ──► … ──► outputs
 *     (inputs)   (neurons)   (neurons)          (one per choice)
 *
 * Every neuron in a layer does the same two tiny steps:
 *
 *   1. weighted sum:  z = w₁·x₁ + w₂·x₂ + … + wₙ·xₙ + b
 *        x = the numbers coming in (from the senses, or from the previous layer)
 *        w = this neuron's WEIGHTS: how much it cares about each input
 *            (big positive = "more of this input makes me fire more",
 *             negative = "this input calms me down", ~0 = "I ignore it")
 *        b = its BIAS: a constant nudge up or down
 *   2. activation:    a = f(z)
 *        a simple bend such as ReLU: f(z) = max(0, z). Without these bends the
 *        whole network would collapse into one straight-line formula.
 *
 * Example: a neuron with weights [0.5, −1] and bias 0.1, inputs [2, 3]:
 *   z = 0.5·2 + (−1)·3 + 0.1 = −1.9   →   ReLU(−1.9) = 0   (this neuron stays quiet)
 *
 * The weights and biases are the ONLY things that change during learning.
 * "Training" = nudging thousands of these numbers so the outputs get better.
 *
 * HOW LEARNING NUDGES THE WEIGHTS (backpropagation)
 * -------------------------------------------------
 * After a forward pass, the learning algorithm (PPO, DQN, … in src/ml/algos)
 * says how the outputs should have been different: the gradient of the loss
 * with respect to each output, written dL/da. "If this output had been a bit
 * bigger, would the loss go up or down, and how fast?"
 *
 * `backward` walks the layers in REVERSE and uses the chain rule to turn that
 * into a gradient for every weight:
 *
 *   dL/dz = dL/da ⊙ f'(z)     how much each neuron's weighted sum mattered
 *                             (f' = slope of the activation; ⊙ = multiply element by element)
 *   dL/dW = dL/dz · xᵀ        weight from input i to neuron o: (how much o mattered) × (input i)
 *   dL/db = dL/dz             bias: just how much the neuron mattered
 *   dL/dx = Wᵀ · dL/dz        how much each INPUT mattered: passed down to the previous layer
 *
 * Then an optimiser (src/ml/optim.ts: SGD, Adam…) moves every weight a small
 * step against its gradient: w ← w − learning_rate · dL/dw. Repeat thousands of times.
 *
 * HOW THE NUMBERS ARE STORED
 * --------------------------
 * Everything is a flat Float32Array (a plain list of 32-bit numbers):
 *   - W, the weights of a layer with I inputs and O neurons: O rows of I
 *     numbers ("row-major"). The weight from input i to neuron o is W[o·I + i].
 *     This is exactly PyTorch's nn.Linear.weight layout, which makes
 *     exporting trivial.
 *   - b: O numbers, one bias per neuron.
 *   - A BATCH of B samples is processed at once: x is B rows of I numbers
 *     (sample n's input i is x[n·I + i]), and the outputs z are B rows of O
 *     numbers (z[n·O + o]). Training uses batches (e.g. 256 samples); driving
 *     the robot uses a batch of 1.
 */
import { RNG } from './rng.ts';

/** The "bend" applied after each weighted sum. */
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

/** Shown in the UI next to each activation (formula + one-line explanation). */
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

/** One hidden layer as written in the program: how many neurons, which activation. */
export interface LayerSpec {
  units: number;
  activation: Activation;
}

/** A parameter (weights or biases) together with its gradient, for the optimisers. */
export interface ParamRef {
  w: Float32Array; // the parameter values
  g: Float32Array; // the accumulated gradient dL/dw
  name: string;
}

/** sigmoid(z) = 1 / (1 + e^(−z)): squashes any number into (0, 1). sigmoid(0) = 0.5. */
const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

// ===========================================================================
// ACTIVATIONS: a = f(z), applied to every number of a layer's output
// ===========================================================================

/**
 * a[i] = f(z[i]) for the first n numbers. `z` holds the weighted sums, `a`
 * receives the activated values (both laid out [batch × neurons]).
 */
export function applyActivation(act: Activation, z: Float32Array, a: Float32Array, n: number): void {
  switch (act) {
    case 'relu':
      // ReLU: negatives become 0, positives pass unchanged.  relu(−2) = 0, relu(3) = 3
      for (let i = 0; i < n; i++) a[i] = z[i] > 0 ? z[i] : 0;
      break;
    case 'leaky_relu':
      // like ReLU, but negatives are scaled down instead of zeroed:  −2 → −0.02
      for (let i = 0; i < n; i++) a[i] = z[i] > 0 ? z[i] : 0.01 * z[i];
      break;
    case 'tanh':
      // tanh squashes into (−1, 1): tanh(0) = 0, tanh(1) ≈ 0.76, tanh(5) ≈ 0.9999
      // tanh(z) = (e²ᶻ − 1)/(e²ᶻ + 1). Same float32 result as Math.tanh, ~1.5× faster.
      for (let i = 0; i < n; i++) {
        const v = z[i];
        if (v > 9) a[i] = 1; // beyond ±9 tanh is 1 or −1 to float32 precision (and e^18 would be huge)
        else if (v < -9) a[i] = -1;
        else {
          const e = Math.exp(2 * v);
          a[i] = (e - 1) / (e + 1);
        }
      }
      break;
    case 'sigmoid':
      // squashes into (0, 1): sigmoid(−3) ≈ 0.05, sigmoid(0) = 0.5, sigmoid(3) ≈ 0.95
      for (let i = 0; i < n; i++) a[i] = sigmoid(z[i]);
      break;
    case 'elu':
      // positives pass; negatives curve smoothly down towards −1:  −2 → e⁻² − 1 ≈ −0.86
      for (let i = 0; i < n; i++) a[i] = z[i] > 0 ? z[i] : Math.exp(z[i]) - 1;
      break;
    case 'swish':
      // z · sigmoid(z): like ReLU but smooth, with a small dip below 0 for negatives
      for (let i = 0; i < n; i++) a[i] = z[i] * sigmoid(z[i]);
      break;
    case 'softplus':
      // log(1 + eᶻ): a smooth ReLU. For big z it is just z (and e^z would overflow), so use z
      for (let i = 0; i < n; i++) {
        const v = z[i];
        a[i] = v > 20 ? v : Math.log1p(Math.exp(v));
      }
      break;
    default:
      // linear: no bend at all (used by output layers that must produce any real number)
      for (let i = 0; i < n; i++) a[i] = z[i];
  }
}

/**
 * The activation's part of the chain rule:  dZ = dA ⊙ f'(z).
 *
 * dA = how the loss changes when each activated output changes (comes from
 * the layer above). f'(z) = the SLOPE of the activation at the value it got.
 * Multiplying them says how the loss changes when the weighted sum z changes.
 *
 * Each case uses whichever of z (before the bend) or a (after) gives the
 * cheapest formula for the slope.
 */
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
      // slope of ReLU: 1 where it let the number through, 0 where it blocked it.
      // So a neuron that was "off" (z ≤ 0) gets no learning signal for this sample.
      for (let i = 0; i < n; i++) dZ[i] = z[i] > 0 ? dA[i] : 0;
      break;
    case 'leaky_relu':
      // slope 1 for positives, 0.01 for negatives
      for (let i = 0; i < n; i++) dZ[i] = z[i] > 0 ? dA[i] : 0.01 * dA[i];
      break;
    case 'tanh':
      // slope of tanh = 1 − tanh² = 1 − a²: 1 in the middle, ~0 when squashed near ±1
      for (let i = 0; i < n; i++) dZ[i] = dA[i] * (1 - a[i] * a[i]);
      break;
    case 'sigmoid':
      // slope of sigmoid = a · (1 − a): at most 0.25 (at z = 0)
      for (let i = 0; i < n; i++) dZ[i] = dA[i] * a[i] * (1 - a[i]);
      break;
    case 'elu':
      // slope 1 for positives; for negatives the slope of eᶻ − 1 is eᶻ = a + 1
      for (let i = 0; i < n; i++) dZ[i] = z[i] > 0 ? dA[i] : dA[i] * (a[i] + 1);
      break;
    case 'swish':
      // product rule on z · s(z): s + z · s · (1 − s)
      for (let i = 0; i < n; i++) {
        const s = sigmoid(z[i]);
        dZ[i] = dA[i] * (s + z[i] * s * (1 - s));
      }
      break;
    case 'softplus':
      // the slope of log(1 + eᶻ) is exactly sigmoid(z)
      for (let i = 0; i < n; i++) dZ[i] = dA[i] * sigmoid(z[i]);
      break;
    default:
      // linear: slope 1, the gradient passes through unchanged
      for (let i = 0; i < n; i++) dZ[i] = dA[i];
  }
}

// ===========================================================================
// THE THREE MATRIX LOOPS OF A DENSE LAYER (where the inputs meet the weights)
// ===========================================================================
//
// Almost all the computing time of a neural network is spent in these three
// loops. In their plain, textbook form they are one line each:
//
//   forward:      z[n,o]  = b[o] + Σ_i W[o,i]·x[n,i]
//                 "for every sample n and neuron o: bias + sum of weight × input"
//
//   weight grad:  gW[o,i] += Σ_n dZ[n,o]·x[n,i]      gb[o] += Σ_n dZ[n,o]
//                 "a weight's gradient = how much its neuron mattered × how big
//                  its input was, added up over the batch"
//
//   input grad:   dX[n,i] = Σ_o dZ[n,o]·W[o,i]
//                 "an input's gradient = how much each neuron mattered × the
//                  weight connecting them, added up over the neurons"
//
// Below they are "register tiled": instead of computing one result at a time,
// each pass over the inputs computes a small block of results at once (2
// samples × 4 neurons, or 2 neurons × 4 samples). Every number loaded from
// memory is then used several times while it sits in a fast CPU register.
// Same maths, same results (up to float rounding), about 2× faster in
// JavaScript engines. tests/ml.test.ts checks them against the plain loops.
//
// Each tiled loop is followed by a "leftover" loop that handles the last
// samples or neurons when the batch size / neuron count is not a multiple of
// the block size (e.g. 7 neurons = one block of 4 + 3 leftovers).
// ===========================================================================

/**
 * FORWARD: z = x·Wᵀ + b for a batch of rows.
 * W is [O × I] row-major, x is [B × I], z is [B × O].
 * In words: for every sample n and every neuron o, z = b[o] + Σ W[o,i]·x[n,i].
 */
export function denseForward(W: Float32Array, b: Float32Array, x: Float32Array, z: Float32Array, B: number, I: number, O: number): void {
  let n = 0;
  // ---- main part: two samples (n and n+1) at a time
  for (; n + 1 < B; n += 2) {
    const xa = n * I; // where sample n's inputs start in x
    const xb = xa + I; // where sample n+1's inputs start
    const za = n * O; // where sample n's outputs go in z
    const zb = za + O; // where sample n+1's outputs go
    let o = 0;
    // four neurons (o … o+3) at a time
    for (; o + 3 < O; o += 4) {
      // where each of the four neurons' weight rows starts in W
      const w0 = o * I;
      const w1 = w0 + I;
      const w2 = w1 + I;
      const w3 = w2 + I;
      // 8 running sums, all starting from the neuron's bias:
      //   a0…a3 = sample n's weighted sums for neurons o…o+3
      //   c0…c3 = sample n+1's weighted sums for the same four neurons
      let a0 = b[o];
      let a1 = b[o + 1];
      let a2 = b[o + 2];
      let a3 = b[o + 3];
      let c0 = a0;
      let c1 = a1;
      let c2 = a2;
      let c3 = a3;
      // walk along the inputs once; each input value and each weight is
      // loaded ONCE and used for several of the 8 sums
      for (let i = 0; i < I; i++) {
        const u = x[xa + i]; // input i of sample n
        const v = x[xb + i]; // input i of sample n+1
        const k0 = W[w0 + i]; // weight from input i to neuron o
        const k1 = W[w1 + i]; //                         neuron o+1
        const k2 = W[w2 + i]; //                         neuron o+2
        const k3 = W[w3 + i]; //                         neuron o+3
        a0 += k0 * u;
        a1 += k1 * u;
        a2 += k2 * u;
        a3 += k3 * u;
        c0 += k0 * v;
        c1 += k1 * v;
        c2 += k2 * v;
        c3 += k3 * v;
      }
      // store the 8 finished weighted sums
      z[za + o] = a0;
      z[za + o + 1] = a1;
      z[za + o + 2] = a2;
      z[za + o + 3] = a3;
      z[zb + o] = c0;
      z[zb + o + 1] = c1;
      z[zb + o + 2] = c2;
      z[zb + o + 3] = c3;
    }
    // leftover neurons (when O is not a multiple of 4): one neuron, two samples
    for (; o < O; o++) {
      const wo = o * I;
      let a = b[o];
      let c = b[o];
      for (let i = 0; i < I; i++) {
        const k = W[wo + i];
        a += k * x[xa + i];
        c += k * x[xb + i];
      }
      z[za + o] = a;
      z[zb + o] = c;
    }
  }
  // ---- leftover sample (when B is odd): the plain textbook loop
  for (; n < B; n++) {
    const xo = n * I;
    const zo = n * O;
    for (let o = 0; o < O; o++) {
      let s = b[o];
      const wo = o * I;
      for (let i = 0; i < I; i++) s += W[wo + i] * x[xo + i];
      z[zo + o] = s;
    }
  }
}

/**
 * WEIGHT GRADIENTS: gW += dZᵀ·x and gb += column sums of dZ.
 *
 * dZ[n,o] = how much neuron o's weighted sum mattered for sample n (from the
 * chain rule). The weight from input i to neuron o gets the gradient
 *   Σ over samples n of  dZ[n,o] × x[n,i]
 * ("the neuron mattered, AND this input was big, so this weight is responsible").
 * The bias gets Σ_n dZ[n,o] (a bias is like a weight whose input is always 1).
 *
 * It ACCUMULATES (+=): gradients from several calls add up until zeroGrad().
 * Tiling here: 2 neurons × 4 samples per pass over the inputs.
 */
export function denseGradWeights(gW: Float32Array, gb: Float32Array, x: Float32Array, dZ: Float32Array, B: number, I: number, O: number): void {
  let o = 0;
  // ---- two neurons (o and o+1) at a time
  for (; o + 1 < O; o += 2) {
    const wa = o * I; // neuron o's weight row (in gW)
    const wb = wa + I; // neuron o+1's weight row
    let sa = 0; // bias gradient of neuron o (sum of its dZ over the batch)
    let sb = 0; // bias gradient of neuron o+1
    let n = 0;
    // four samples (n … n+3) at a time
    for (; n + 3 < B; n += 4) {
      // dZ values: a0…a3 = neuron o for the 4 samples, b0…b3 = neuron o+1
      const z0 = n * O + o;
      const z1 = z0 + O;
      const z2 = z1 + O;
      const z3 = z2 + O;
      const a0 = dZ[z0];
      const a1 = dZ[z1];
      const a2 = dZ[z2];
      const a3 = dZ[z3];
      const b0 = dZ[z0 + 1];
      const b1 = dZ[z1 + 1];
      const b2 = dZ[z2 + 1];
      const b3 = dZ[z3 + 1];
      sa += a0 + a1 + a2 + a3;
      sb += b0 + b1 + b2 + b3;
      // shortcut: if both neurons were "off" for all four samples (common with
      // ReLU), these samples add nothing to the weight gradients
      if (a0 === 0 && a1 === 0 && a2 === 0 && a3 === 0 && b0 === 0 && b1 === 0 && b2 === 0 && b3 === 0) continue;
      const x0 = n * I; // the 4 samples' input rows
      const x1 = x0 + I;
      const x2 = x1 + I;
      const x3 = x2 + I;
      for (let i = 0; i < I; i++) {
        // input i of the 4 samples
        const p = x[x0 + i];
        const q = x[x1 + i];
        const r = x[x2 + i];
        const s = x[x3 + i];
        // weight (o, i) and weight (o+1, i): add (how much the neuron mattered) × (input) for each sample
        gW[wa + i] += a0 * p + a1 * q + a2 * r + a3 * s;
        gW[wb + i] += b0 * p + b1 * q + b2 * r + b3 * s;
      }
    }
    // leftover samples (when B is not a multiple of 4)
    for (; n < B; n++) {
      const a = dZ[n * O + o];
      const c = dZ[n * O + o + 1];
      sa += a;
      sb += c;
      const xo = n * I;
      for (let i = 0; i < I; i++) {
        const p = x[xo + i];
        gW[wa + i] += a * p;
        gW[wb + i] += c * p;
      }
    }
    gb[o] += sa;
    gb[o + 1] += sb;
  }
  // ---- leftover neuron (when O is odd): the plain loop, skipping samples where it was off
  for (; o < O; o++) {
    const wo = o * I;
    let s = 0;
    for (let n = 0; n < B; n++) {
      const d = dZ[n * O + o];
      if (d === 0) continue;
      s += d;
      const xo = n * I;
      for (let i = 0; i < I; i++) gW[wo + i] += d * x[xo + i];
    }
    gb[o] += s;
  }
}

/**
 * INPUT GRADIENTS: dX = dZ·W (overwrites dX): how the loss changes with each input.
 *
 * An input feeds every neuron of the layer, so its gradient adds up, over the
 * neurons, (how much that neuron mattered) × (the weight connecting them):
 *   dX[n,i] = Σ_o dZ[n,o] × W[o,i]
 * This is what lets the error flow back to the PREVIOUS layer ("backpropagation").
 * Tiling here: 2 samples × 4 neurons per pass over the inputs.
 */
export function denseGradInputs(W: Float32Array, dZ: Float32Array, dX: Float32Array, B: number, I: number, O: number): void {
  let n = 0;
  // ---- two samples (n and n+1) at a time
  for (; n + 1 < B; n += 2) {
    const xa = n * I; // sample n's input gradients
    const xb = xa + I; // sample n+1's
    const za = n * O; // sample n's dZ
    const zb = za + O; // sample n+1's dZ
    for (let i = 0; i < I; i++) {
      dX[xa + i] = 0; // start from zero: we add up the contributions of all neurons
      dX[xb + i] = 0;
    }
    let o = 0;
    // four neurons (o … o+3) at a time
    for (; o + 3 < O; o += 4) {
      // how much each of the 4 neurons mattered, for the 2 samples
      const a0 = dZ[za + o];
      const a1 = dZ[za + o + 1];
      const a2 = dZ[za + o + 2];
      const a3 = dZ[za + o + 3];
      const b0 = dZ[zb + o];
      const b1 = dZ[zb + o + 1];
      const b2 = dZ[zb + o + 2];
      const b3 = dZ[zb + o + 3];
      // all four neurons were off for both samples: nothing to pass back
      if (a0 === 0 && a1 === 0 && a2 === 0 && a3 === 0 && b0 === 0 && b1 === 0 && b2 === 0 && b3 === 0) continue;
      const w0 = o * I;
      const w1 = w0 + I;
      const w2 = w1 + I;
      const w3 = w2 + I;
      for (let i = 0; i < I; i++) {
        // the weights from input i to the 4 neurons
        const k0 = W[w0 + i];
        const k1 = W[w1 + i];
        const k2 = W[w2 + i];
        const k3 = W[w3 + i];
        dX[xa + i] += a0 * k0 + a1 * k1 + a2 * k2 + a3 * k3;
        dX[xb + i] += b0 * k0 + b1 * k1 + b2 * k2 + b3 * k3;
      }
    }
    // leftover neurons
    for (; o < O; o++) {
      const a = dZ[za + o];
      const c = dZ[zb + o];
      const wo = o * I;
      for (let i = 0; i < I; i++) {
        const k = W[wo + i];
        dX[xa + i] += a * k;
        dX[xb + i] += c * k;
      }
    }
  }
  // ---- leftover sample (when B is odd)
  for (; n < B; n++) {
    const xo = n * I;
    const zo = n * O;
    for (let i = 0; i < I; i++) dX[xo + i] = 0;
    for (let o = 0; o < O; o++) {
      const d = dZ[zo + o];
      if (d === 0) continue;
      const wo = o * I;
      for (let i = 0; i < I; i++) dX[xo + i] += d * W[wo + i];
    }
  }
}

// ===========================================================================
// A DENSE LAYER: weights + biases + activation, with forward and backward
// ===========================================================================

/** A fully-connected ("dense") layer: every input connects to every output. */
export class Dense {
  readonly inSize: number; // I: how many numbers come in
  readonly outSize: number; // O: how many neurons (= numbers going out)
  act: Activation;
  W: Float32Array; // the weights, [O × I]: W[o·I + i] connects input i to neuron o
  b: Float32Array; // the biases, one per neuron
  gW: Float32Array; // gradient of the loss for every weight (same layout as W), filled by backward()
  gb: Float32Array; // gradient for every bias

  // Buffers reused between calls (sized for the largest batch seen so far),
  // so that training does not create garbage for the memory manager to clean up.
  private cap = 0;
  private xRef: Float32Array = new Float32Array(0); // the inputs of the last forward pass (backward needs them)
  z: Float32Array = new Float32Array(0); // weighted sums of the last forward pass  [batch × O]
  a: Float32Array = new Float32Array(0); // activated outputs of the last forward pass [batch × O]
  private dZ: Float32Array = new Float32Array(0); // gradient w.r.t. z [batch × O]
  private dX: Float32Array = new Float32Array(0); // gradient w.r.t. the inputs [batch × I]

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
   * Starting weights: small random numbers. They must not all be equal (every
   * neuron would then compute the same thing forever) and must have the right
   * size: too big and the signals explode layer after layer, too small and
   * they fade away.
   *   - He init for ReLU-family activations: std = √(2 / inputs). ReLU zeroes
   *     about half the neurons, so the weights are a bit bigger to compensate.
   *   - Glorot/Xavier for the rest: std = √(2 / (inputs + outputs)).
   * Biases start at 0. `scale` lets an output layer start smaller (e.g. a
   * policy that starts almost uniform: all choices equally likely).
   */
  initWeights(rng: RNG, scale = 1): void {
    const reluLike = this.act === 'relu' || this.act === 'leaky_relu' || this.act === 'elu' || this.act === 'swish';
    const std = reluLike ? Math.sqrt(2 / this.inSize) : Math.sqrt(2 / (this.inSize + this.outSize));
    for (let i = 0; i < this.W.length; i++) this.W[i] = rng.gauss() * std * scale;
    this.b.fill(0);
  }

  /** Make the buffers big enough for `batch` samples (only grows, never shrinks). */
  private ensure(batch: number): void {
    if (batch <= this.cap) return;
    this.cap = batch;
    this.z = new Float32Array(batch * this.outSize);
    this.a = new Float32Array(batch * this.outSize);
    this.dZ = new Float32Array(batch * this.outSize);
    this.dX = new Float32Array(batch * this.inSize);
  }

  /**
   * FORWARD: inputs → outputs.
   *   x: [batch × inSize] numbers coming in
   *   returns a: [batch × outSize] = f(x·Wᵀ + b)
   * Remembers x, z and a: backward() needs them to compute the gradients.
   */
  forward(x: Float32Array, batch: number): Float32Array {
    this.ensure(batch);
    this.xRef = x;
    denseForward(this.W, this.b, x, this.z, batch, this.inSize, this.outSize); // step 1: weighted sums
    applyActivation(this.act, this.z, this.a, batch * this.outSize); //            step 2: the bend
    return this.a;
  }

  /**
   * BACKWARD: the chain rule for this layer, in three steps.
   *   dA: [batch × outSize] how the loss changes with each output (from the layer above)
   * Accumulates gW/gb. Returns dL/dx, the gradient for the layer BELOW (or an
   * empty array when not needed, e.g. for the first layer: nobody needs the
   * gradient of the senses).
   */
  backward(dA: Float32Array, batch: number, needInputGrad: boolean): Float32Array {
    const { W, gW, gb, z, a, dZ, dX, inSize: I, outSize: O } = this;
    activationBackward(this.act, z, a, dA, dZ, batch * O); // 1. through the bend:  dZ = dA ⊙ f'(z)
    denseGradWeights(gW, gb, this.xRef, dZ, batch, I, O); //  2. weight gradients:  gW += dZᵀ·x,  gb += Σ dZ
    if (!needInputGrad) return dX;
    denseGradInputs(W, dZ, dX, batch, I, O); //                3. pass it down:      dX = dZ·W
    return dX;
  }

  /** Forget the accumulated gradients (called before every new update). */
  zeroGrad(): void {
    this.gW.fill(0);
    this.gb.fill(0);
  }
}

/** A network as plain JSON (for saving, exporting and copying between threads). */
export interface MLPJSON {
  inSize: number;
  layers: { inSize: number; outSize: number; activation: Activation; W: number[]; b: number[] }[];
}

// ===========================================================================
// THE WHOLE NETWORK: a stack of dense layers (a "multi-layer perceptron")
// ===========================================================================

/**
 * Multi-layer perceptron: a stack of Dense layers.
 *
 *   new MLP(6, [{units: 64, activation: 'tanh'}, {units: 64, activation: 'tanh'}], 3, 'linear', rng)
 *
 * makes  6 senses → 64 neurons → 64 neurons → 3 outputs, with
 * 6·64+64 + 64·64+64 + 64·3+3 = 4,739 weights and biases.
 */
export class MLP {
  layers: Dense[];
  readonly inSize: number;
  readonly outSize: number;

  constructor(inSize: number, hidden: LayerSpec[], outSize: number, outAct: Activation, rng: RNG, outInitScale = 1) {
    this.inSize = inSize;
    this.outSize = outSize;
    this.layers = [];
    let prev = inSize; // each layer's input size is the previous layer's neuron count
    for (const h of hidden) {
      this.layers.push(new Dense(prev, h.units, h.activation, rng));
      prev = h.units;
    }
    // the output layer (usually 'linear': scores, values or action means can be any number)
    this.layers.push(new Dense(prev, outSize, outAct, rng, outInitScale));
  }

  /** FORWARD through every layer: the output of one is the input of the next. */
  forward(x: Float32Array, batch: number): Float32Array {
    let h = x;
    for (const l of this.layers) h = l.forward(h, batch);
    return h;
  }

  /**
   * BACKWARD through every layer, LAST to FIRST: each layer turns "how the
   * loss changes with my outputs" into gradients for its weights and "how the
   * loss changes with my inputs", which is the output gradient of the layer below.
   */
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

  /** Every weight and bias array with its gradient: what the optimiser updates. */
  params(prefix = ''): ParamRef[] {
    const out: ParamRef[] = [];
    this.layers.forEach((l, i) => {
      out.push({ w: l.W, g: l.gW, name: `${prefix}layer${i}.weight` });
      out.push({ w: l.b, g: l.gb, name: `${prefix}layer${i}.bias` });
    });
    return out;
  }

  /** Total number of weights and biases. */
  numParams(): number {
    let n = 0;
    for (const l of this.layers) n += l.W.length + l.b.length;
    return n;
  }

  /** Copy another network's weights into this one (same shapes), e.g. DQN's target network. */
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

  /** All parameters as one flat vector (used by neuroevolution, which mutates them directly). */
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

  /** The reverse of getFlat: load all parameters from one flat vector. */
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
   * post-activation values (z and a). Used by the live brain visualiser: this
   * is what you see light up in the Brain tab.
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
