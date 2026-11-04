/**
 * A tiny GPT: a decoder-only transformer language model, written from
 * scratch with explicit forward AND backward passes (no autograd).
 *
 *   tokens ─► token embedding + position embedding
 *          ─► N × [ x += Attention(LayerNorm(x)) ;  x += MLP(LayerNorm(x)) ]
 *          ─► LayerNorm ─► linear "head" ─► one score (logit) per vocabulary entry
 *
 * Attention lets every position look at the earlier ones ("causal"):
 *
 *   q, k, v = x·Wq, x·Wk, x·Wv            (one set per head)
 *   score(t, s) = q_t · k_s / √d_head     for s ≤ t
 *   att(t, ·) = softmax(score(t, ·))
 *   out_t = Σ_s att(t, s) · v_s
 *
 * Training minimises cross-entropy: −log p(next token). The backward pass
 * applies the chain rule to every step above in reverse order; tests/llm.test.ts
 * checks it against finite differences.
 *
 * Weights use PyTorch's nn.Linear layout ([out × in], row-major), and the
 * matrix maths reuses the tiled dense kernels from ../ml/nn.ts.
 */
import { denseForward, denseGradInputs, denseGradWeights } from '../ml/nn.ts';
import { RNG } from '../ml/rng.ts';

export type LLMAct = 'gelu' | 'relu' | 'swish';
export type PosKind = 'learned' | 'sinusoidal' | 'none';

export interface GPTConfig {
  /** number of different tokens */
  vocab: number;
  /** context length: how many tokens the model can look back at */
  ctx: number;
  /** embedding size (width of the residual stream) */
  d: number;
  layers: number;
  heads: number;
  /** MLP hidden size = mlp × d */
  mlp: number;
  act: LLMAct;
  pos: PosKind;
  /** share the token embedding with the output head (fewer weights) */
  tie: boolean;
}

export interface Tensor {
  name: string;
  shape: number[];
  w: Float32Array;
  g: Float32Array;
  /** weight decay applies to matrices, not to biases / layer-norm gains */
  decay: boolean;
}

interface Block {
  ln1g: Tensor;
  ln1b: Tensor;
  qkvW: Tensor;
  qkvB: Tensor;
  projW: Tensor;
  projB: Tensor;
  ln2g: Tensor;
  ln2b: Tensor;
  fcW: Tensor;
  fcB: Tensor;
  mprojW: Tensor;
  mprojB: Tensor;
}

/** Activations kept from the forward pass for the backward pass (sized for N = B·T tokens). */
interface LayerCache {
  ln1xhat: Float32Array;
  ln1rstd: Float32Array;
  h1: Float32Array;
  qkv: Float32Array;
  att: Float32Array;
  y: Float32Array;
  ln2xhat: Float32Array;
  ln2rstd: Float32Array;
  h2: Float32Array;
  f1: Float32Array;
  g1: Float32Array;
}

const GELU_K = Math.sqrt(2 / Math.PI);

export function validateConfig(c: GPTConfig): string | null {
  if (!(c.vocab >= 2)) return 'the vocabulary needs at least 2 tokens';
  if (!(c.ctx >= 2 && c.ctx <= 1024)) return 'context length must be 2…1024';
  if (!(c.d >= 4 && c.d <= 1024)) return 'embedding size must be 4…1024';
  if (c.d % c.heads !== 0) return `the embedding size (${c.d}) must be divisible by the number of heads (${c.heads})`;
  if (!(c.layers >= 1 && c.layers <= 16)) return 'layers must be 1…16';
  if (!(c.mlp >= 1 && c.mlp <= 8)) return 'MLP width must be 1…8 × the embedding size';
  return null;
}

/** Number of weights a configuration has (without building it). */
export function countParams(c: GPTConfig): number {
  const d = c.d;
  const m = c.mlp * d;
  const perLayer = 2 * d + (3 * d * d + 3 * d) + (d * d + d) + 2 * d + (m * d + m) + (d * m + d);
  return c.vocab * d + (c.pos === 'learned' ? c.ctx * d : 0) + c.layers * perLayer + 2 * d + (c.tie ? 0 : c.vocab * d);
}

/** Approximate training FLOPs per token (forward + backward ≈ 3 × forward). */
export function flopsPerToken(c: GPTConfig): number {
  const nonEmb = countParams(c) - c.vocab * c.d - (c.pos === 'learned' ? c.ctx * c.d : 0) + (c.tie ? c.vocab * c.d : 0);
  return 6 * nonEmb + 6 * c.layers * c.ctx * c.d;
}

export class GPT {
  readonly cfg: GPTConfig;
  readonly params: Tensor[] = [];
  readonly wte: Tensor;
  readonly wpe: Tensor | null;
  readonly blocks: Block[] = [];
  readonly lnfg: Tensor;
  readonly lnfb: Tensor;
  readonly head: Tensor | null;
  /** fixed sinusoidal position table [ctx × d] (pos = 'sinusoidal') */
  private sinPos: Float32Array | null = null;
  // ---- caches (forward → backward), sized for the largest batch seen
  private capN = 0;
  private B = 0;
  private T = 0;
  private idx: Int32Array = new Int32Array(0);
  private caches: LayerCache[] = [];
  private lnfxhat = new Float32Array(0);
  private lnfrstd = new Float32Array(0);
  private hf = new Float32Array(0);
  logits = new Float32Array(0);
  // scratch for backward
  private dx = new Float32Array(0);
  private dh = new Float32Array(0);
  private dqkv = new Float32Array(0);
  private dy = new Float32Array(0);
  private dg = new Float32Array(0);
  private dlogits = new Float32Array(0);
  private dattRow = new Float32Array(0);
  private zeroV: Float32Array;
  private scratchV: Float32Array;

  constructor(cfg: GPTConfig, seed = 1) {
    const err = validateConfig(cfg);
    if (err) throw new Error(err);
    this.cfg = { ...cfg };
    const { vocab: V, ctx: T, d, layers: L, mlp } = cfg;
    const m = mlp * d;
    const rng = new RNG(seed);
    const add = (name: string, shape: number[], decay: boolean, init: (i: number) => number): Tensor => {
      const n = shape.reduce((a, b) => a * b, 1);
      const t: Tensor = { name, shape, w: new Float32Array(n), g: new Float32Array(n), decay };
      for (let i = 0; i < n; i++) t.w[i] = init(i);
      this.params.push(t);
      return t;
    };
    const normal = (std: number) => () => rng.gauss() * std;
    const zero = () => 0;
    const one = () => 1;
    // GPT-2 style init: N(0, 0.02); projections into the residual stream scaled by 1/√(2·layers)
    const resid = 0.02 / Math.sqrt(2 * L);
    this.wte = add('wte', [V, d], true, normal(0.02));
    this.wpe = cfg.pos === 'learned' ? add('wpe', [T, d], true, normal(0.01)) : null;
    if (cfg.pos === 'sinusoidal') {
      const p = new Float32Array(T * d);
      for (let t = 0; t < T; t++)
        for (let i = 0; i < d; i += 2) {
          const f = t / Math.pow(10000, i / d);
          p[t * d + i] = Math.sin(f);
          if (i + 1 < d) p[t * d + i + 1] = Math.cos(f);
        }
      this.sinPos = p;
    }
    for (let l = 0; l < L; l++) {
      const pre = `h${l}.`;
      this.blocks.push({
        ln1g: add(pre + 'ln1.weight', [d], false, one),
        ln1b: add(pre + 'ln1.bias', [d], false, zero),
        qkvW: add(pre + 'attn.qkv.weight', [3 * d, d], true, normal(0.02)),
        qkvB: add(pre + 'attn.qkv.bias', [3 * d], false, zero),
        projW: add(pre + 'attn.proj.weight', [d, d], true, normal(resid)),
        projB: add(pre + 'attn.proj.bias', [d], false, zero),
        ln2g: add(pre + 'ln2.weight', [d], false, one),
        ln2b: add(pre + 'ln2.bias', [d], false, zero),
        fcW: add(pre + 'mlp.fc.weight', [m, d], true, normal(0.02)),
        fcB: add(pre + 'mlp.fc.bias', [m], false, zero),
        mprojW: add(pre + 'mlp.proj.weight', [d, m], true, normal(resid)),
        mprojB: add(pre + 'mlp.proj.bias', [d], false, zero),
      });
    }
    this.lnfg = add('lnf.weight', [d], false, one);
    this.lnfb = add('lnf.bias', [d], false, zero);
    this.head = cfg.tie ? null : add('head.weight', [V, d], true, normal(0.02));
    this.zeroV = new Float32Array(V);
    this.scratchV = new Float32Array(V);
  }

  get numParams(): number {
    let n = 0;
    for (const p of this.params) n += p.w.length;
    return n;
  }

  zeroGrad(): void {
    for (const p of this.params) p.g.fill(0);
  }

  /** The weights of the output head (the token embedding when tied). */
  get headW(): Float32Array {
    return this.head ? this.head.w : this.wte.w;
  }

  /** Position vector for position t (length d), or null. */
  posVec(t: number): Float32Array | null {
    const d = this.cfg.d;
    if (this.wpe) return this.wpe.w.subarray(t * d, t * d + d);
    if (this.sinPos) return this.sinPos.subarray(t * d, t * d + d);
    return null;
  }

  private ensure(B: number, T: number): void {
    const N = B * T;
    this.B = B;
    this.T = T;
    if (N <= this.capN && this.caches.length && this.caches[0].att.length >= B * this.cfg.heads * T * T) return;
    this.capN = N;
    const { d, heads: H, mlp, vocab: V } = this.cfg;
    const m = mlp * d;
    this.idx = new Int32Array(N);
    this.caches = this.blocks.map(() => ({
      ln1xhat: new Float32Array(N * d),
      ln1rstd: new Float32Array(N),
      h1: new Float32Array(N * d),
      qkv: new Float32Array(N * 3 * d),
      att: new Float32Array(B * H * T * T),
      y: new Float32Array(N * d),
      ln2xhat: new Float32Array(N * d),
      ln2rstd: new Float32Array(N),
      h2: new Float32Array(N * d),
      f1: new Float32Array(N * m),
      g1: new Float32Array(N * m),
    }));
    this.lnfxhat = new Float32Array(N * d);
    this.lnfrstd = new Float32Array(N);
    this.hf = new Float32Array(N * d);
    this.logits = new Float32Array(N * V);
    this.dx = new Float32Array(N * d);
    this.dh = new Float32Array(N * d);
    this.dqkv = new Float32Array(N * 3 * d);
    this.dy = new Float32Array(N * d);
    this.dg = new Float32Array(N * m);
    this.dlogits = new Float32Array(N * V);
    this.dattRow = new Float32Array(T);
  }

  /**
   * Forward pass for B sequences of T tokens (row-major [B × T]).
   * Returns the logits [B·T × V] (a view into an internal buffer).
   */
  forward(tokens: ArrayLike<number>, B: number, T: number): Float32Array {
    const { d, vocab: V } = this.cfg;
    if (T > this.cfg.ctx) throw new Error(`sequence of ${T} tokens is longer than the context (${this.cfg.ctx})`);
    this.ensure(B, T);
    const N = B * T;
    const x = new Float32Array(N * d);
    // 1. embeddings
    for (let n = 0; n < N; n++) {
      const tok = tokens[n] | 0;
      if (tok < 0 || tok >= V) throw new Error(`token id ${tok} is outside the vocabulary (${V})`);
      this.idx[n] = tok;
      const pv = this.posVec(n % T);
      const e = tok * d;
      for (let i = 0; i < d; i++) x[n * d + i] = this.wte.w[e + i] + (pv ? pv[i] : 0);
    }
    // 2. transformer blocks
    for (let l = 0; l < this.blocks.length; l++) this.blockForward(l, x, B, T);
    // 3. final layer norm + head
    layerNormForward(x, this.lnfg.w, this.lnfb.w, this.hf, this.lnfxhat, this.lnfrstd, N, d);
    denseForward(this.headW, this.zeroV, this.hf, this.logits, N, d, V);
    return this.logits.subarray(0, N * V);
  }

  private blockForward(l: number, x: Float32Array, B: number, T: number): void {
    const blk = this.blocks[l];
    const c = this.caches[l];
    const { d, heads: H, mlp } = this.cfg;
    const m = mlp * d;
    const N = B * T;
    // attention sub-block
    layerNormForward(x, blk.ln1g.w, blk.ln1b.w, c.h1, c.ln1xhat, c.ln1rstd, N, d);
    denseForward(blk.qkvW.w, blk.qkvB.w, c.h1, c.qkv, N, d, 3 * d);
    attentionForward(c.qkv, c.att, c.y, B, T, H, d);
    const a = this.dh; // reuse as scratch: projected attention output
    denseForward(blk.projW.w, blk.projB.w, c.y, a, N, d, d);
    for (let i = 0; i < N * d; i++) x[i] += a[i];
    // MLP sub-block
    layerNormForward(x, blk.ln2g.w, blk.ln2b.w, c.h2, c.ln2xhat, c.ln2rstd, N, d);
    denseForward(blk.fcW.w, blk.fcB.w, c.h2, c.f1, N, d, m);
    activate(this.cfg.act, c.f1, c.g1, N * m);
    denseForward(blk.mprojW.w, blk.mprojB.w, c.g1, a, N, m, d);
    for (let i = 0; i < N * d; i++) x[i] += a[i];
  }

  /**
   * Cross-entropy loss of the last forward pass against `targets` [B·T];
   * fills the gradient of the loss w.r.t. the logits. Returns the mean loss.
   */
  lossAndGrad(targets: ArrayLike<number>): number {
    const N = this.B * this.T;
    const V = this.cfg.vocab;
    const L = this.logits;
    const dL = this.dlogits;
    let loss = 0;
    for (let n = 0; n < N; n++) {
      const o = n * V;
      let max = -Infinity;
      for (let i = 0; i < V; i++) if (L[o + i] > max) max = L[o + i];
      let sum = 0;
      for (let i = 0; i < V; i++) sum += dL[o + i] = Math.exp(L[o + i] - max);
      const tgt = targets[n] | 0;
      loss += Math.log(sum) - (L[o + tgt] - max);
      const inv = 1 / (sum * N);
      for (let i = 0; i < V; i++) dL[o + i] *= inv;
      dL[o + tgt] -= 1 / N;
    }
    return loss / N;
  }

  /** Mean cross-entropy of the last forward pass (no gradient). */
  loss(targets: ArrayLike<number>): number {
    const N = this.B * this.T;
    const V = this.cfg.vocab;
    const L = this.logits;
    let loss = 0;
    for (let n = 0; n < N; n++) {
      const o = n * V;
      let max = -Infinity;
      for (let i = 0; i < V; i++) if (L[o + i] > max) max = L[o + i];
      let sum = 0;
      for (let i = 0; i < V; i++) sum += Math.exp(L[o + i] - max);
      loss += Math.log(sum) - (L[o + (targets[n] | 0)] - max);
    }
    return loss / N;
  }

  /** Backward pass from the gradient filled by lossAndGrad(); ACCUMULATES into every tensor's .g */
  backward(): void {
    const { d, vocab: V } = this.cfg;
    const N = this.B * this.T;
    const T = this.T;
    // head (and the tied embedding)
    const headG = this.head ? this.head.g : this.wte.g;
    denseGradWeights(headG, this.scratchV, this.hf, this.dlogits, N, d, V);
    denseGradInputs(this.headW, this.dlogits, this.dh, N, d, V);
    // final layer norm
    this.dx.fill(0, 0, N * d);
    layerNormBackward(this.dh, this.lnfxhat, this.lnfrstd, this.lnfg.w, this.lnfg.g, this.lnfb.g, this.dx, N, d);
    for (let l = this.blocks.length - 1; l >= 0; l--) this.blockBackward(l);
    // embeddings
    const dx = this.dx;
    for (let n = 0; n < N; n++) {
      const e = this.idx[n] * d;
      for (let i = 0; i < d; i++) this.wte.g[e + i] += dx[n * d + i];
      if (this.wpe) {
        const p = (n % T) * d;
        for (let i = 0; i < d; i++) this.wpe.g[p + i] += dx[n * d + i];
      }
    }
  }

  private blockBackward(l: number): void {
    const blk = this.blocks[l];
    const c = this.caches[l];
    const { d, heads: H, mlp } = this.cfg;
    const m = mlp * d;
    const B = this.B;
    const T = this.T;
    const N = B * T;
    const dx = this.dx; // gradient flowing along the residual stream
    // MLP: x += mproj(act(fc(LN2(x))))
    denseGradWeights(blk.mprojW.g, blk.mprojB.g, c.g1, dx, N, m, d);
    denseGradInputs(blk.mprojW.w, dx, this.dg, N, m, d);
    activateBackward(this.cfg.act, c.f1, this.dg, N * m);
    denseGradWeights(blk.fcW.g, blk.fcB.g, c.h2, this.dg, N, d, m);
    denseGradInputs(blk.fcW.w, this.dg, this.dh, N, d, m);
    layerNormBackward(this.dh, c.ln2xhat, c.ln2rstd, blk.ln2g.w, blk.ln2g.g, blk.ln2b.g, dx, N, d);
    // attention: x += proj(attn(qkv(LN1(x))))
    denseGradWeights(blk.projW.g, blk.projB.g, c.y, dx, N, d, d);
    denseGradInputs(blk.projW.w, dx, this.dy, N, d, d);
    this.dqkv.fill(0, 0, N * 3 * d);
    attentionBackward(c.qkv, c.att, this.dy, this.dqkv, this.dattRow, B, T, H, d);
    denseGradWeights(blk.qkvW.g, blk.qkvB.g, c.h1, this.dqkv, N, d, 3 * d);
    denseGradInputs(blk.qkvW.w, this.dqkv, this.dh, N, d, 3 * d);
    layerNormBackward(this.dh, c.ln1xhat, c.ln1rstd, blk.ln1g.w, blk.ln1g.g, blk.ln1b.g, dx, N, d);
  }

  /** Attention probabilities of layer l from the last forward pass: [B × H × T × T]. */
  attention(l: number): Float32Array {
    return this.caches[l].att.subarray(0, this.B * this.cfg.heads * this.T * this.T);
  }

  /** Copy of all weights (in params order), e.g. to send to another thread. */
  getWeights(): Float32Array[] {
    return this.params.map((p) => p.w.slice());
  }

  setWeights(ws: ArrayLike<number>[]): void {
    if (ws.length !== this.params.length) throw new Error('weights do not match this model');
    this.params.forEach((p, i) => {
      if (ws[i].length !== p.w.length) throw new Error(`weights for ${p.name} have the wrong size`);
      p.w.set(ws[i]);
    });
  }
}

// ---------------------------------------------------------------------------
// The individual operations (exported for the decoder and for tests)
// ---------------------------------------------------------------------------

/** LayerNorm over the last dimension: out = (x − mean)/√(var + ε) · g + b. */
export function layerNormForward(x: Float32Array, g: Float32Array, b: Float32Array, out: Float32Array, xhat: Float32Array, rstd: Float32Array, N: number, d: number): void {
  for (let n = 0; n < N; n++) {
    const o = n * d;
    let mean = 0;
    for (let i = 0; i < d; i++) mean += x[o + i];
    mean /= d;
    let v = 0;
    for (let i = 0; i < d; i++) {
      const t = x[o + i] - mean;
      v += t * t;
    }
    const r = 1 / Math.sqrt(v / d + 1e-5);
    rstd[n] = r;
    for (let i = 0; i < d; i++) {
      const xh = (x[o + i] - mean) * r;
      xhat[o + i] = xh;
      out[o + i] = xh * g[i] + b[i];
    }
  }
}

/** dx += LayerNorm backward; accumulates dg, db. */
export function layerNormBackward(dout: Float32Array, xhat: Float32Array, rstd: Float32Array, g: Float32Array, dg: Float32Array, db: Float32Array, dx: Float32Array, N: number, d: number): void {
  for (let n = 0; n < N; n++) {
    const o = n * d;
    let m1 = 0;
    let m2 = 0;
    for (let i = 0; i < d; i++) {
      const dy = dout[o + i];
      const dxh = dy * g[i];
      dg[i] += dy * xhat[o + i];
      db[i] += dy;
      m1 += dxh;
      m2 += dxh * xhat[o + i];
    }
    m1 /= d;
    m2 /= d;
    const r = rstd[n];
    for (let i = 0; i < d; i++) dx[o + i] += r * (dout[o + i] * g[i] - m1 - xhat[o + i] * m2);
  }
}

/** Causal multi-head self-attention. qkv rows are [q | k | v] (each d wide, heads side by side). */
export function attentionForward(qkv: Float32Array, att: Float32Array, y: Float32Array, B: number, T: number, H: number, d: number): void {
  const dh = d / H;
  const scale = 1 / Math.sqrt(dh);
  const D3 = 3 * d;
  for (let b = 0; b < B; b++)
    for (let h = 0; h < H; h++) {
      const ab = (b * H + h) * T * T;
      for (let t = 0; t < T; t++) {
        const qo = (b * T + t) * D3 + h * dh;
        const row = ab + t * T;
        let max = -Infinity;
        for (let s = 0; s <= t; s++) {
          const ko = (b * T + s) * D3 + d + h * dh;
          let sc = 0;
          for (let j = 0; j < dh; j++) sc += qkv[qo + j] * qkv[ko + j];
          sc *= scale;
          att[row + s] = sc;
          if (sc > max) max = sc;
        }
        let sum = 0;
        for (let s = 0; s <= t; s++) sum += att[row + s] = Math.exp(att[row + s] - max);
        const inv = 1 / sum;
        for (let s = 0; s <= t; s++) att[row + s] *= inv;
        for (let s = t + 1; s < T; s++) att[row + s] = 0;
        const yo = (b * T + t) * d + h * dh;
        for (let j = 0; j < dh; j++) y[yo + j] = 0;
        for (let s = 0; s <= t; s++) {
          const p = att[row + s];
          const vo = (b * T + s) * D3 + 2 * d + h * dh;
          for (let j = 0; j < dh; j++) y[yo + j] += p * qkv[vo + j];
        }
      }
    }
}

/** Backward of attentionForward: accumulates into dqkv (zero it first). */
export function attentionBackward(qkv: Float32Array, att: Float32Array, dy: Float32Array, dqkv: Float32Array, datt: Float32Array, B: number, T: number, H: number, d: number): void {
  const dh = d / H;
  const scale = 1 / Math.sqrt(dh);
  const D3 = 3 * d;
  for (let b = 0; b < B; b++)
    for (let h = 0; h < H; h++) {
      const ab = (b * H + h) * T * T;
      for (let t = 0; t < T; t++) {
        const row = ab + t * T;
        const yo = (b * T + t) * d + h * dh;
        // through the weighted sum: dv_s += att·dy,  datt_s = dy·v_s
        let dot = 0;
        for (let s = 0; s <= t; s++) {
          const vo = (b * T + s) * D3 + 2 * d + h * dh;
          const p = att[row + s];
          let da = 0;
          for (let j = 0; j < dh; j++) {
            da += dy[yo + j] * qkv[vo + j];
            dqkv[vo + j] += p * dy[yo + j];
          }
          datt[s] = da;
          dot += p * da;
        }
        // through the softmax, then the scaled dot product
        const qo = (b * T + t) * D3 + h * dh;
        for (let s = 0; s <= t; s++) {
          const ds = att[row + s] * (datt[s] - dot) * scale;
          if (ds === 0) continue;
          const ko = (b * T + s) * D3 + d + h * dh;
          for (let j = 0; j < dh; j++) {
            dqkv[qo + j] += ds * qkv[ko + j];
            dqkv[ko + j] += ds * qkv[qo + j];
          }
        }
      }
    }
}

export function activate(act: LLMAct, x: Float32Array, out: Float32Array, n: number): void {
  if (act === 'relu') for (let i = 0; i < n; i++) out[i] = x[i] > 0 ? x[i] : 0;
  else if (act === 'swish')
    for (let i = 0; i < n; i++) {
      const v = x[i];
      out[i] = v / (1 + Math.exp(-v));
    }
  else
    for (let i = 0; i < n; i++) {
      // GELU, tanh approximation (as in GPT-2)
      const v = x[i];
      const u = GELU_K * (v + 0.044715 * v * v * v);
      const th = u > 9 ? 1 : u < -9 ? -1 : (Math.exp(2 * u) - 1) / (Math.exp(2 * u) + 1);
      out[i] = 0.5 * v * (1 + th);
    }
}

/** In place: grad ← grad · act'(x). */
export function activateBackward(act: LLMAct, x: Float32Array, grad: Float32Array, n: number): void {
  if (act === 'relu') for (let i = 0; i < n; i++) grad[i] = x[i] > 0 ? grad[i] : 0;
  else if (act === 'swish')
    for (let i = 0; i < n; i++) {
      const s = 1 / (1 + Math.exp(-x[i]));
      grad[i] *= s + x[i] * s * (1 - s);
    }
  else
    for (let i = 0; i < n; i++) {
      const v = x[i];
      const u = GELU_K * (v + 0.044715 * v * v * v);
      const th = u > 9 ? 1 : u < -9 ? -1 : (Math.exp(2 * u) - 1) / (Math.exp(2 * u) + 1);
      grad[i] *= 0.5 * (1 + th) + 0.5 * v * (1 - th * th) * GELU_K * (1 + 3 * 0.044715 * v * v);
    }
}
