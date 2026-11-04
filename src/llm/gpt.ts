/**
 * ===========================================================================
 *  gpt.ts: a tiny GPT (the language model), forward and backward, from scratch
 * ===========================================================================
 *
 * A decoder-only transformer language model, written with explicit forward
 * AND backward passes (no autograd library).
 *
 * WHAT IT DOES
 * ------------
 * Input: a list of token ids, e.g. "the cat" → [12, 7, 33, 2, ...] (one id per
 * character, word or word piece; see tokenizer.ts). Output: for EVERY position,
 * one score (a "logit") per token of the vocabulary: how likely each token is
 * to come NEXT. Softmax turns the scores into probabilities.
 *
 *   tokens ─► token embedding + position embedding        (each token becomes a vector of d numbers)
 *          ─► N × [ x += Attention(LayerNorm(x)) ;         (tokens look at earlier tokens)
 *                   x += MLP(LayerNorm(x)) ]               (each token "thinks" on its own)
 *          ─► LayerNorm ─► linear "head" ─► one score (logit) per vocabulary entry
 *
 * THE RESIDUAL STREAM
 * -------------------
 * Every token carries a vector x of d numbers (d = "embedding size", e.g. 64)
 * through the whole network. The blocks never replace x: they ADD to it
 * (x += …). Think of x as a notepad each block writes extra notes on. This
 * "residual" design lets information and gradients flow straight through,
 * which is why transformers can be stacked deep.
 *
 * ATTENTION, the part that makes it a transformer
 * ------------------------------------------------
 * Every position makes three vectors from its x (by multiplying with weights):
 *   q = query  "what am I looking for?"
 *   k = key    "what do I contain?"
 *   v = value  "what I pass on if someone looks at me"
 * Then, for position t:
 *
 *   score(t, s) = q_t · k_s / √d_head     for every EARLIER (or same) position s ≤ t
 *   att(t, ·) = softmax(score(t, ·))       scores → percentages that add up to 100%
 *   out_t = Σ_s att(t, s) · v_s            a blend of the values, weighted by those percentages
 *
 * "Causal": s ≤ t only. A token can never look at the future, otherwise
 * predicting the next token would be cheating. Several "heads" do this in
 * parallel on different slices of the vector, each looking for something else.
 *
 * HOW IT LEARNS
 * -------------
 * Training minimises the cross-entropy loss: −log(probability given to the
 * real next token), averaged over all positions. The backward pass applies the
 * chain rule to every step above in REVERSE order, giving the gradient of the
 * loss for every weight (stored in each tensor's .g). train.ts then updates
 * the weights with AdamW. tests/llm.test.ts checks every gradient here against
 * finite differences (nudge a weight a tiny bit, see how the loss moves).
 *
 * HOW THE NUMBERS ARE STORED
 * --------------------------
 * Flat Float32Arrays. B sequences of T tokens are processed together:
 * N = B·T token positions, and position n = b·T + t (sequence b, place t).
 * A vector of d numbers per position is stored as N rows of d: x[n·d + i].
 * Weights use PyTorch's nn.Linear layout ([out × in], row-major), and the
 * matrix maths reuses the fast tiled dense kernels from ../ml/nn.ts.
 */
import { denseForward, denseGradInputs, denseGradWeights } from '../ml/nn.ts';
import { RNG } from '../ml/rng.ts';

/** The non-linearity inside each MLP. */
export type LLMAct = 'gelu' | 'relu' | 'swish';
/** How the model knows the ORDER of tokens: a learned table, fixed sine waves, or not at all. */
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

/** One learnable array of numbers (a weight matrix, a bias, …) and its gradient. */
export interface Tensor {
  name: string; // e.g. "h0.attn.qkv.weight" (same names as the PyTorch export)
  shape: number[]; // e.g. [192, 64]
  w: Float32Array; // the values
  g: Float32Array; // the gradient dLoss/dw, filled by backward()
  /** weight decay applies to matrices, not to biases / layer-norm gains */
  decay: boolean;
}

/** The weights of one transformer block. */
interface Block {
  ln1g: Tensor; // LayerNorm before attention: gain (starts at 1)
  ln1b: Tensor; //                               bias (starts at 0)
  qkvW: Tensor; // [3d × d]: makes q, k and v in ONE matrix multiply (rows: all of q, then k, then v)
  qkvB: Tensor;
  projW: Tensor; // [d × d]: mixes the heads' outputs back into one vector
  projB: Tensor;
  ln2g: Tensor; // LayerNorm before the MLP
  ln2b: Tensor;
  fcW: Tensor; // [m × d]: MLP expand (m = mlp × d, e.g. 4 × 64 = 256)
  fcB: Tensor;
  mprojW: Tensor; // [d × m]: MLP project back to d
  mprojB: Tensor;
}

/**
 * Values kept from the forward pass because the backward pass needs them
 * (sized for N = B·T tokens). "xhat" = the normalised values inside LayerNorm,
 * "rstd" = 1/standard deviation per token, h = LayerNorm outputs, f1/g1 = the
 * MLP's hidden layer before/after the activation.
 */
interface LayerCache {
  ln1xhat: Float32Array;
  ln1rstd: Float32Array;
  h1: Float32Array; // LayerNorm-1 output, the input of qkv   [N × d]
  qkv: Float32Array; // queries, keys, values                   [N × 3d]
  att: Float32Array; // attention percentages                    [B × H × T × T]
  y: Float32Array; // attention output, before proj           [N × d]
  ln2xhat: Float32Array;
  ln2rstd: Float32Array;
  h2: Float32Array; // LayerNorm-2 output, the MLP input       [N × d]
  f1: Float32Array; // MLP hidden layer, before activation      [N × m]
  g1: Float32Array; // MLP hidden layer, after activation       [N × m]
}

/** √(2/π), a constant of the GELU formula */
const GELU_K = Math.sqrt(2 / Math.PI);

/** Returns a readable error if the settings cannot make a model, else null. */
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
  // per block: LN1 (2d) + qkv (3d·d + 3d) + proj (d·d + d) + LN2 (2d) + fc (m·d + m) + mproj (d·m + d)
  const perLayer = 2 * d + (3 * d * d + 3 * d) + (d * d + d) + 2 * d + (m * d + m) + (d * m + d);
  // + token embedding (vocab·d) + learned positions (ctx·d) + final LayerNorm (2d) + head (unless tied)
  return c.vocab * d + (c.pos === 'learned' ? c.ctx * d : 0) + c.layers * perLayer + 2 * d + (c.tie ? 0 : c.vocab * d);
}

/** Approximate training FLOPs per token (forward + backward ≈ 3 × forward). */
export function flopsPerToken(c: GPTConfig): number {
  // each weight is used in one multiply-add per token (2 FLOPs) in the forward pass and twice
  // as much in the backward pass: 6 per weight. Embedding lookups cost nothing. Attention adds
  // its own work, growing with the context.
  const nonEmb = countParams(c) - c.vocab * c.d - (c.pos === 'learned' ? c.ctx * c.d : 0) + (c.tie ? c.vocab * c.d : 0);
  return 6 * nonEmb + 6 * c.layers * c.ctx * c.d;
}

export class GPT {
  readonly cfg: GPTConfig;
  /** every learnable tensor, in a fixed order (saving, exporting and the optimiser use this list) */
  readonly params: Tensor[] = [];
  /** token embedding [vocab × d]: row k is the vector of token k */
  readonly wte: Tensor;
  /** learned position embedding [ctx × d] (pos = 'learned'), else null */
  readonly wpe: Tensor | null;
  readonly blocks: Block[] = [];
  /** final LayerNorm gain and bias */
  readonly lnfg: Tensor;
  readonly lnfb: Tensor;
  /** output head [vocab × d]; null when tied to the token embedding */
  readonly head: Tensor | null;
  /** fixed sinusoidal position table [ctx × d] (pos = 'sinusoidal') */
  private sinPos: Float32Array | null = null;
  // ---- caches (forward → backward), sized for the largest batch seen
  private capN = 0;
  private B = 0;
  private T = 0;
  private idx: Int32Array = new Int32Array(0); // the token ids of the last forward pass
  private caches: LayerCache[] = [];
  private lnfxhat = new Float32Array(0);
  private lnfrstd = new Float32Array(0);
  private hf = new Float32Array(0); // final LayerNorm output, the input of the head [N × d]
  logits = new Float32Array(0); // the output scores [N × vocab]
  // scratch for backward
  private dx = new Float32Array(0); // gradient along the residual stream [N × d]
  private dh = new Float32Array(0);
  private dqkv = new Float32Array(0);
  private dy = new Float32Array(0);
  private dg = new Float32Array(0);
  private dlogits = new Float32Array(0); // gradient of the loss for every score [N × vocab]
  private dattRow = new Float32Array(0);
  private zeroV: Float32Array; // zeros, used as the "bias" of the head (it has none)
  private scratchV: Float32Array; // receives the head's (non-existent) bias gradient

  constructor(cfg: GPTConfig, seed = 1) {
    const err = validateConfig(cfg);
    if (err) throw new Error(err);
    this.cfg = { ...cfg };
    const { vocab: V, ctx: T, d, layers: L, mlp } = cfg;
    const m = mlp * d;
    const rng = new RNG(seed);
    // create a tensor, fill it with init(i), and register it in this.params
    const add = (name: string, shape: number[], decay: boolean, init: (i: number) => number): Tensor => {
      const n = shape.reduce((a, b) => a * b, 1);
      const t: Tensor = { name, shape, w: new Float32Array(n), g: new Float32Array(n), decay };
      for (let i = 0; i < n; i++) t.w[i] = init(i);
      this.params.push(t);
      return t;
    };
    const normal = (std: number) => () => rng.gauss() * std; // small random numbers
    const zero = () => 0;
    const one = () => 1;
    // GPT-2 style init: N(0, 0.02); projections into the residual stream scaled by 1/√(2·layers)
    // (every block ADDS to the stream twice, so smaller additions keep its size steady when stacked)
    const resid = 0.02 / Math.sqrt(2 * L);
    this.wte = add('wte', [V, d], true, normal(0.02));
    this.wpe = cfg.pos === 'learned' ? add('wpe', [T, d], true, normal(0.01)) : null;
    if (cfg.pos === 'sinusoidal') {
      // the original Transformer's fixed positions: pairs of sine/cosine waves, fast to slow,
      // so every position gets a unique pattern and nearby positions look alike
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
    // weight tying: the head reuses the token embedding (score of token k = x · wte[k])
    this.head = cfg.tie ? null : add('head.weight', [V, d], true, normal(0.02));
    this.zeroV = new Float32Array(V);
    this.scratchV = new Float32Array(V);
  }

  get numParams(): number {
    let n = 0;
    for (const p of this.params) n += p.w.length;
    return n;
  }

  /** Forget all gradients (they ACCUMULATE: call this before every backward pass). */
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

  /** Make the caches big enough for B sequences of T tokens (only grows). */
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
   * FORWARD PASS for B sequences of T tokens (row-major [B × T]).
   * Returns the logits [B·T × V]: for every position, a score for every
   * possible next token (a view into an internal buffer).
   */
  forward(tokens: ArrayLike<number>, B: number, T: number): Float32Array {
    const { d, vocab: V } = this.cfg;
    if (T > this.cfg.ctx) throw new Error(`sequence of ${T} tokens is longer than the context (${this.cfg.ctx})`);
    this.ensure(B, T);
    const N = B * T;
    const x = new Float32Array(N * d); // the residual stream: one vector of d numbers per position
    // 1. embeddings: look up each token's vector and add its position's vector
    //    x[n] = wte[token] + wpe[position]
    for (let n = 0; n < N; n++) {
      const tok = tokens[n] | 0;
      if (tok < 0 || tok >= V) throw new Error(`token id ${tok} is outside the vocabulary (${V})`);
      this.idx[n] = tok; // remembered: backward() sends the gradient back to these rows
      const pv = this.posVec(n % T); // n % T = the position inside its sequence
      const e = tok * d; // where this token's row starts in wte
      for (let i = 0; i < d; i++) x[n * d + i] = this.wte.w[e + i] + (pv ? pv[i] : 0);
    }
    // 2. transformer blocks: each one adds attention and MLP results to x
    for (let l = 0; l < this.blocks.length; l++) this.blockForward(l, x, B, T);
    // 3. final layer norm + head: one score per vocabulary token, for every position
    //    logits[n, k] = hf[n] · headW[k]   (no bias)
    layerNormForward(x, this.lnfg.w, this.lnfb.w, this.hf, this.lnfxhat, this.lnfrstd, N, d);
    denseForward(this.headW, this.zeroV, this.hf, this.logits, N, d, V);
    return this.logits.subarray(0, N * V);
  }

  /** One transformer block, forward: x += Attention(LN1(x)); x += MLP(LN2(x)). */
  private blockForward(l: number, x: Float32Array, B: number, T: number): void {
    const blk = this.blocks[l];
    const c = this.caches[l];
    const { d, heads: H, mlp } = this.cfg;
    const m = mlp * d;
    const N = B * T;
    // ---- attention sub-block
    layerNormForward(x, blk.ln1g.w, blk.ln1b.w, c.h1, c.ln1xhat, c.ln1rstd, N, d); // h1 = LN1(x)
    denseForward(blk.qkvW.w, blk.qkvB.w, c.h1, c.qkv, N, d, 3 * d); //                qkv = h1·Wqkvᵀ + b
    attentionForward(c.qkv, c.att, c.y, B, T, H, d); //                                y = causal attention
    const a = this.dh; // reuse as scratch: projected attention output
    denseForward(blk.projW.w, blk.projB.w, c.y, a, N, d, d); //                       a = y·Wprojᵀ + b
    for (let i = 0; i < N * d; i++) x[i] += a[i]; //                                  residual: x += a
    // ---- MLP sub-block: works on every position separately
    layerNormForward(x, blk.ln2g.w, blk.ln2b.w, c.h2, c.ln2xhat, c.ln2rstd, N, d); // h2 = LN2(x)
    denseForward(blk.fcW.w, blk.fcB.w, c.h2, c.f1, N, d, m); //                       f1 = h2·Wfcᵀ + b   (expand to m)
    activate(this.cfg.act, c.f1, c.g1, N * m); //                                      g1 = GELU(f1)
    denseForward(blk.mprojW.w, blk.mprojB.w, c.g1, a, N, m, d); //                    a = g1·Wprojᵀ + b  (back to d)
    for (let i = 0; i < N * d; i++) x[i] += a[i]; //                                  residual: x += a
  }

  /**
   * CROSS-ENTROPY LOSS of the last forward pass against `targets` [B·T]
   * (the real next token at every position), and its gradient.
   *
   * For one position with scores L (one per token) and the real next token tgt:
   *   p = softmax(L)                    probabilities: p_k = e^(L_k) / Σ_j e^(L_j)
   *   loss = −log p_tgt                 0 when the model was 100% sure of the right token,
   *                                     large when it gave it a tiny probability
   * Its gradient is surprisingly simple:
   *   dloss/dL_k = p_k − (1 if k = tgt else 0)
   * "Push every score down by its probability, and the right one up by 1."
   * Everything is divided by N because the loss is the MEAN over positions.
   * Returns the mean loss.
   */
  lossAndGrad(targets: ArrayLike<number>): number {
    const N = this.B * this.T;
    const V = this.cfg.vocab;
    const L = this.logits;
    const dL = this.dlogits;
    let loss = 0;
    for (let n = 0; n < N; n++) {
      const o = n * V; // this position's scores start here
      // softmax, safely: subtract the largest score first so e^(…) never overflows
      // (it does not change the result: the shift cancels out in the division)
      let max = -Infinity;
      for (let i = 0; i < V; i++) if (L[o + i] > max) max = L[o + i];
      let sum = 0;
      for (let i = 0; i < V; i++) sum += dL[o + i] = Math.exp(L[o + i] - max); // dL temporarily holds e^(L − max)
      const tgt = targets[n] | 0;
      // −log p_tgt = −log(e^(L_tgt − max) / sum) = log(sum) − (L_tgt − max)
      loss += Math.log(sum) - (L[o + tgt] - max);
      // gradient: p_k / N for every token, minus 1/N for the real one
      const inv = 1 / (sum * N);
      for (let i = 0; i < V; i++) dL[o + i] *= inv;
      dL[o + tgt] -= 1 / N;
    }
    return loss / N;
  }

  /** Mean cross-entropy of the last forward pass (no gradient): used for the validation loss. */
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

  /**
   * BACKWARD PASS from the gradient filled by lossAndGrad(): the forward pass
   * in reverse, applying the chain rule at every step. ACCUMULATES into every
   * tensor's .g (call zeroGrad() first).
   */
  backward(): void {
    const { d, vocab: V } = this.cfg;
    const N = this.B * this.T;
    const T = this.T;
    // ---- head (and the tied embedding): logits = hf · headWᵀ
    const headG = this.head ? this.head.g : this.wte.g;
    denseGradWeights(headG, this.scratchV, this.hf, this.dlogits, N, d, V); // head weight gradient
    denseGradInputs(this.headW, this.dlogits, this.dh, N, d, V); //            dh = gradient w.r.t. hf
    // ---- final layer norm: gives the gradient for the residual stream, dx
    this.dx.fill(0, 0, N * d);
    layerNormBackward(this.dh, this.lnfxhat, this.lnfrstd, this.lnfg.w, this.lnfg.g, this.lnfb.g, this.dx, N, d);
    // ---- the blocks, last to first
    for (let l = this.blocks.length - 1; l >= 0; l--) this.blockBackward(l);
    // ---- embeddings: x[n] = wte[token] + wpe[position], so each position's gradient is
    //      simply ADDED to the row of the token it came from (and of its position)
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

  /**
   * One block, backward. `dx` holds the gradient of the residual stream
   * (dLoss/dx). Because the forward pass did x += f(x), the gradient flows
   * BOTH straight through (unchanged) and through f: each sub-block adds its
   * share into dx (the LayerNorm backward functions use +=).
   */
  private blockBackward(l: number): void {
    const blk = this.blocks[l];
    const c = this.caches[l];
    const { d, heads: H, mlp } = this.cfg;
    const m = mlp * d;
    const B = this.B;
    const T = this.T;
    const N = B * T;
    const dx = this.dx; // gradient flowing along the residual stream
    // ---- MLP: x += mproj(act(fc(LN2(x)))), undone step by step from the end
    denseGradWeights(blk.mprojW.g, blk.mprojB.g, c.g1, dx, N, m, d); // mproj weights
    denseGradInputs(blk.mprojW.w, dx, this.dg, N, m, d); //              dg = gradient w.r.t. g1
    activateBackward(this.cfg.act, c.f1, this.dg, N * m); //              through the GELU: dg = gradient w.r.t. f1
    denseGradWeights(blk.fcW.g, blk.fcB.g, c.h2, this.dg, N, d, m); //   fc weights
    denseGradInputs(blk.fcW.w, this.dg, this.dh, N, d, m); //             dh = gradient w.r.t. h2
    layerNormBackward(this.dh, c.ln2xhat, c.ln2rstd, blk.ln2g.w, blk.ln2g.g, blk.ln2b.g, dx, N, d); // dx += …
    // ---- attention: x += proj(attn(qkv(LN1(x))))
    denseGradWeights(blk.projW.g, blk.projB.g, c.y, dx, N, d, d); //     proj weights
    denseGradInputs(blk.projW.w, dx, this.dy, N, d, d); //               dy = gradient w.r.t. the attention output
    this.dqkv.fill(0, 0, N * 3 * d);
    attentionBackward(c.qkv, c.att, this.dy, this.dqkv, this.dattRow, B, T, H, d); // gradient w.r.t. q, k, v
    denseGradWeights(blk.qkvW.g, blk.qkvB.g, c.h1, this.dqkv, N, d, 3 * d); //       qkv weights
    denseGradInputs(blk.qkvW.w, this.dqkv, this.dh, N, d, 3 * d); //                  dh = gradient w.r.t. h1
    layerNormBackward(this.dh, c.ln1xhat, c.ln1rstd, blk.ln1g.w, blk.ln1g.g, blk.ln1b.g, dx, N, d); // dx += …
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

/**
 * LAYERNORM over the last dimension: out = (x − mean)/√(var + ε) · g + b.
 *
 * For each position separately: shift its d numbers so they average 0, scale
 * them so their spread (standard deviation) is 1, then apply a learned gain g
 * and bias b per channel. It keeps the numbers in a healthy range however
 * many blocks are stacked. ε = 0.00001 avoids dividing by zero.
 * Keeps xhat (the normalised values) and rstd (1/std) for the backward pass.
 */
export function layerNormForward(x: Float32Array, g: Float32Array, b: Float32Array, out: Float32Array, xhat: Float32Array, rstd: Float32Array, N: number, d: number): void {
  for (let n = 0; n < N; n++) {
    const o = n * d; // this position's numbers start here
    let mean = 0;
    for (let i = 0; i < d; i++) mean += x[o + i];
    mean /= d;
    let v = 0; // variance: the average squared distance from the mean
    for (let i = 0; i < d; i++) {
      const t = x[o + i] - mean;
      v += t * t;
    }
    const r = 1 / Math.sqrt(v / d + 1e-5); // 1 / standard deviation
    rstd[n] = r;
    for (let i = 0; i < d; i++) {
      const xh = (x[o + i] - mean) * r; // normalised: mean 0, spread 1
      xhat[o + i] = xh;
      out[o + i] = xh * g[i] + b[i];
    }
  }
}

/**
 * LayerNorm backward: dx += …; accumulates dg, db.
 *
 * The gains and biases are easy: dg += dout · xhat, db += dout.
 * The input is trickier because every output depends on ALL d inputs of the
 * position (through the mean and the spread). The chain rule works out to
 *   dx_i = rstd · (dxhat_i − mean(dxhat) − xhat_i · mean(dxhat · xhat))
 * where dxhat = dout · g. The two means (m1, m2 below) are the corrections for
 * "moving one input also moves the mean and the spread".
 */
export function layerNormBackward(dout: Float32Array, xhat: Float32Array, rstd: Float32Array, g: Float32Array, dg: Float32Array, db: Float32Array, dx: Float32Array, N: number, d: number): void {
  for (let n = 0; n < N; n++) {
    const o = n * d;
    let m1 = 0; // will be mean(dxhat)
    let m2 = 0; // will be mean(dxhat · xhat)
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

/**
 * CAUSAL MULTI-HEAD SELF-ATTENTION, forward.
 *
 * qkv rows are [q | k | v] (each d wide, heads side by side): for position n,
 *   q starts at n·3d, k at n·3d + d, v at n·3d + 2d,
 * and head h uses the slice [h·dh, (h+1)·dh) of each (dh = d / H numbers per head).
 * Writes att [B × H × T × T] (the percentages, kept for backward and for the
 * Attention tab) and y [N × d] (the blended values).
 */
export function attentionForward(qkv: Float32Array, att: Float32Array, y: Float32Array, B: number, T: number, H: number, d: number): void {
  const dh = d / H; // numbers per head
  const scale = 1 / Math.sqrt(dh); // dividing by √dh keeps the scores in a reasonable range
  const D3 = 3 * d; // row length of qkv
  for (let b = 0; b < B; b++) // every sequence
    for (let h = 0; h < H; h++) {
      // every head
      const ab = (b * H + h) * T * T; // this (sequence, head)'s T×T attention table
      for (let t = 0; t < T; t++) {
        // position t is "reading"
        const qo = (b * T + t) * D3 + h * dh; // its query
        const row = ab + t * T; // its row of percentages
        // 1. scores: query of t · key of every earlier position s (s ≤ t: causal)
        let max = -Infinity;
        for (let s = 0; s <= t; s++) {
          const ko = (b * T + s) * D3 + d + h * dh; // key of position s
          let sc = 0;
          for (let j = 0; j < dh; j++) sc += qkv[qo + j] * qkv[ko + j]; // dot product = "how well do they match"
          sc *= scale;
          att[row + s] = sc;
          if (sc > max) max = sc;
        }
        // 2. softmax over the scores (minus the max, for safety): percentages that add up to 1
        let sum = 0;
        for (let s = 0; s <= t; s++) sum += att[row + s] = Math.exp(att[row + s] - max);
        const inv = 1 / sum;
        for (let s = 0; s <= t; s++) att[row + s] *= inv;
        for (let s = t + 1; s < T; s++) att[row + s] = 0; // the future: never looked at
        // 3. output = the values of the earlier positions, blended by those percentages
        const yo = (b * T + t) * d + h * dh;
        for (let j = 0; j < dh; j++) y[yo + j] = 0;
        for (let s = 0; s <= t; s++) {
          const p = att[row + s];
          const vo = (b * T + s) * D3 + 2 * d + h * dh; // value of position s
          for (let j = 0; j < dh; j++) y[yo + j] += p * qkv[vo + j];
        }
      }
    }
}

/**
 * Backward of attentionForward: accumulates into dqkv (zero it first).
 * dy = gradient of the loss w.r.t. the attention output y. The three steps of
 * the forward pass are undone in reverse order: blend → softmax → dot products.
 */
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
        // 3 ← through the weighted sum y_t = Σ_s att(t,s)·v_s:
        //     each value v_s gets att(t,s)·dy_t;  each percentage gets datt_s = dy_t · v_s
        let dot = 0; // Σ_s att(t,s)·datt_s, needed by the softmax step
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
        // 2 ← through the softmax: dscore_s = att_s · (datt_s − Σ att·datt)
        // 1 ← through the scaled dot product score = q·k·scale:
        //     dq_t += dscore_s · k_s · scale,   dk_s += dscore_s · q_t · scale
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

/**
 * The MLP's non-linearity, out[i] = act(x[i]).
 * GELU is the smooth "ReLU" used by GPT: ≈ x for big positive x, ≈ 0 for big
 * negative x, with a smooth curve (and a small dip) in between.
 */
export function activate(act: LLMAct, x: Float32Array, out: Float32Array, n: number): void {
  if (act === 'relu') for (let i = 0; i < n; i++) out[i] = x[i] > 0 ? x[i] : 0;
  else if (act === 'swish')
    for (let i = 0; i < n; i++) {
      const v = x[i];
      out[i] = v / (1 + Math.exp(-v)); // x · sigmoid(x)
    }
  else
    for (let i = 0; i < n; i++) {
      // GELU, tanh approximation (as in GPT-2): 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
      const v = x[i];
      const u = GELU_K * (v + 0.044715 * v * v * v);
      const th = u > 9 ? 1 : u < -9 ? -1 : (Math.exp(2 * u) - 1) / (Math.exp(2 * u) + 1); // tanh(u)
      out[i] = 0.5 * v * (1 + th);
    }
}

/** In place: grad ← grad · act'(x) (the slope of the activation at x). */
export function activateBackward(act: LLMAct, x: Float32Array, grad: Float32Array, n: number): void {
  if (act === 'relu') for (let i = 0; i < n; i++) grad[i] = x[i] > 0 ? grad[i] : 0;
  else if (act === 'swish')
    for (let i = 0; i < n; i++) {
      const s = 1 / (1 + Math.exp(-x[i]));
      grad[i] *= s + x[i] * s * (1 - s);
    }
  else
    for (let i = 0; i < n; i++) {
      // derivative of the GELU formula above (product rule + chain rule through tanh)
      const v = x[i];
      const u = GELU_K * (v + 0.044715 * v * v * v);
      const th = u > 9 ? 1 : u < -9 ? -1 : (Math.exp(2 * u) - 1) / (Math.exp(2 * u) + 1);
      grad[i] *= 0.5 * (1 + th) + 0.5 * v * (1 - th * th) * GELU_K * (1 + 3 * 0.044715 * v * v);
    }
}
