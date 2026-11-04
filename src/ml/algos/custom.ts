/**
 * Your own learning algorithm (code mode: `train.custom({...})`).
 *
 * The program supplies four functions; this class runs them in the training
 * loop and gives them a small toolkit (`ctx`) built on the same from-scratch
 * library the built-in algorithms use:
 *
 *   setup(ctx)            build networks and optimizers (once)
 *   act(obs, ctx)         choose an action for one world → one choice per action
 *   observe(t, ctx)       see what happened: { obs, action, reward, done, next, world, success }
 *   update(ctx)           learn whenever you like; return true when you changed the brain
 *
 * The network you store in `this.policy` is "the brain": it is what the 3D
 * robot uses, what is saved and what gets exported. It must output one
 * logit per choice (softmax per action), like the built-in policies.
 */
import type { AgentSpec } from '../../sim/types.ts';
import type { ScriptFactory, ScriptHandlers } from '../../sim/env.ts';
import { describeScriptError } from '../../sim/scriptErrors.ts';
import { ACTIVATIONS, MLP } from '../nn.ts';
import type { Activation, ParamRef } from '../nn.ts';
import { clipGradNorm, makeOptimizer } from '../optim.ts';
import type { Optimizer, OptimizerKind } from '../optim.ts';
import type { ActionSpace } from '../policy.ts';
import { RNG } from '../rng.ts';
import { StatsLog, VecEnv } from './common.ts';
import type { Algorithm, BrainSnapshot } from './common.ts';

export interface CustomDef {
  setup?: (ctx: CustomCtx) => void;
  act?: (obs: Float32Array, ctx: CustomCtx) => ArrayLike<number> | number;
  observe?: (t: Transition, ctx: CustomCtx) => void;
  update?: (ctx: CustomCtx) => boolean | void;
  policy?: Net;
  [key: string]: unknown;
}

export interface Transition {
  obs: Float32Array;
  action: number[];
  reward: number;
  done: boolean;
  /** true if the episode ended because time ran out (not because of success/failure) */
  timeout: boolean;
  success: boolean;
  next: Float32Array;
  world: number;
}

/** A neural network handle for user code: forward/backward on plain arrays. */
export class Net {
  readonly mlp: MLP;
  readonly inSize: number;
  readonly outSize: number;
  private x = new Float32Array(0);
  private d = new Float32Array(0);
  private lastBatch = 0;

  constructor(mlp: MLP) {
    this.mlp = mlp;
    this.inSize = mlp.inSize;
    this.outSize = mlp.outSize;
  }

  /** Outputs for `batch` inputs laid out one after another (a copy you may keep). */
  forward(x: ArrayLike<number>, batch = 1): Float32Array {
    const n = this.inSize * batch;
    if (x.length < n) throw new Error(`forward(): expected ${batch} × ${this.inSize} numbers, got ${x.length}`);
    if (this.x.length < n) this.x = new Float32Array(n);
    for (let i = 0; i < n; i++) this.x[i] = x[i];
    this.lastBatch = batch;
    return this.mlp.forward(this.x, batch).slice(0, this.outSize * batch);
  }

  /** Backpropagate dLoss/dOutputs for the batch of the LAST forward() call; adds to the gradients. */
  backward(dOut: ArrayLike<number>, batch = this.lastBatch): void {
    if (batch !== this.lastBatch) throw new Error(`backward(): call forward() with the same batch (${batch}) first`);
    const n = this.outSize * batch;
    if (dOut.length < n) throw new Error(`backward(): expected ${n} gradient numbers, got ${dOut.length}`);
    if (this.d.length < n) this.d = new Float32Array(n);
    for (let i = 0; i < n; i++) this.d[i] = Number.isFinite(dOut[i]) ? dOut[i] : 0;
    this.mlp.backward(this.d, batch);
  }

  zeroGrad(): void {
    this.mlp.zeroGrad();
  }

  get params(): ParamRef[] {
    return this.mlp.params();
  }

  get numParams(): number {
    return this.mlp.numParams();
  }

  /** Copy the weights of another network with the same shape (e.g. a target network). */
  copyFrom(other: Net): void {
    this.mlp.copyFrom(other.mlp);
  }

  clone(): Net {
    return new Net(this.mlp.clone());
  }
}

export interface CustomCtx {
  obsSize: number;
  numEnvs: number;
  /** number of logits the policy must output (sum of the choices of every action) */
  outputs: number;
  /** choices per action, e.g. [3, 3] for move + turn */
  branches: number[];
  /** names of the actions and of their choices */
  actions: { name: string; choices: string[] }[];
  /** total environment steps so far */
  steps: number;
  mlp(hidden?: number[], activation?: Activation, outSize?: number, inSize?: number): Net;
  optimizer(net: Net, lr?: number, kind?: OptimizerKind): { step(): void; lr: number };
  adam(net: Net, lr?: number): { step(): void; lr: number };
  sgd(net: Net, lr?: number): { step(): void; lr: number };
  clipGrad(net: Net, maxNorm: number): number;
  probs(logits: ArrayLike<number>, row?: number): number[][];
  sample(logits: ArrayLike<number>, row?: number): number[];
  best(logits: ArrayLike<number>, row?: number): number[];
  logProb(logits: ArrayLike<number>, action: ArrayLike<number>, row?: number): number;
  entropy(logits: ArrayLike<number>, row?: number): number;
  addLogProbGrad(logits: ArrayLike<number>, row: number, action: ArrayLike<number>, scale: number, out: Float32Array): void;
  addEntropyGrad(logits: ArrayLike<number>, row: number, scale: number, out: Float32Array): void;
  batch(rows: ArrayLike<number>[]): Float32Array;
  worlds<T>(make: () => T): T[];
  random(): number;
  gauss(): number;
  randint(n: number): number;
  log(metrics: Record<string, number>): void;
}

export class CustomAlgo implements Algorithm {
  readonly kind = 'custom' as const;
  phase = 'starting';
  stats = new StatsLog();
  updates = 0;
  private vec: VecEnv;
  private spec: AgentSpec;
  private def: CustomDef;
  private ctx: CustomCtx;
  private rng: RNG;
  private branches: number[];
  private O: number;
  private act: Float32Array;
  private prevObs: Float32Array;
  private lastMetrics: Record<string, number> = {};
  private firstNet: Net | null = null;
  private optimizers: Optimizer[] = [];
  private actChoices: number[][];

  constructor(vec: VecEnv, spec: AgentSpec, space: ActionSpace, factory: ScriptFactory, init: BrainSnapshot | null) {
    if (space.kind !== 'discrete') throw new Error('train.custom works with discrete actions: use actions.discrete().');
    this.vec = vec;
    this.spec = spec;
    this.rng = new RNG(spec.train.seed * 7 + 11);
    this.branches = space.branches.slice();
    this.O = this.branches.reduce((a, b) => a + b, 0);
    this.act = new Float32Array(vec.n * vec.actDims);
    this.prevObs = new Float32Array(vec.n * vec.obsSize);
    this.actChoices = Array.from({ length: vec.n }, () => new Array(this.branches.length).fill(0));
    // the program's own copy of its algorithm object
    let S: ScriptHandlers & { algo?: CustomDef };
    try {
      S = factory(new Proxy({}, { get: () => () => 0 }) as never, Math) as ScriptHandlers & { algo?: CustomDef };
    } catch (e) {
      throw new Error(`your program failed to start: ${describeScriptError(e, spec)}`);
    }
    if (!S.algo || typeof S.algo !== 'object') throw new Error('train.custom({...}) was not found in the program.');
    this.def = S.algo;
    for (const k of ['act', 'update'] as const) if (typeof this.def[k] !== 'function') throw new Error(`train.custom needs a ${k}() function.`);
    this.ctx = this.makeCtx(space);
    this.call('setup', () => this.def.setup?.call(this.def, this.ctx));
    const policy = this.policy();
    if (!policy) throw new Error('setup() must create the brain: this.policy = ctx.mlp([32, 32], "tanh")');
    if (policy.inSize !== vec.obsSize || policy.outSize !== this.O) throw new Error(`this.policy must take ${vec.obsSize} inputs and give ${this.O} outputs (it has ${policy.inSize} → ${policy.outSize}). Use ctx.mlp(hidden) to get the right sizes.`);
    if (init && init.kind === 'policy' && policy.mlp.sameShape(init.net)) {
      policy.mlp.layers.forEach((l, i) => {
        l.W.set(init.net.layers[i].W);
        l.b.set(init.net.layers[i].b);
      });
      this.updates = init.updates;
    }
    vec.onEpisode = (r) => this.stats.addEpisode(r);
    vec.reset();
  }

  private policy(): Net | null {
    const p = this.def.policy;
    return p instanceof Net ? p : this.firstNet;
  }

  /** Run user code; errors carry the line number of the program. */
  private call<T>(what: string, f: () => T): T {
    try {
      return f();
    } catch (e) {
      throw new Error(`${what}(): ${describeScriptError(e, this.spec)}`);
    }
  }

  private makeCtx(space: ActionSpace): CustomCtx {
    const B = this.branches;
    const O = this.O;
    const rng = this.rng;
    const vec = this.vec;
    const self = this;
    const softmaxRow = (logits: ArrayLike<number>, base: number, n: number): number[] => {
      let m = -Infinity;
      for (let i = 0; i < n; i++) m = Math.max(m, logits[base + i]);
      const p = new Array<number>(n);
      let s = 0;
      for (let i = 0; i < n; i++) s += p[i] = Math.exp(logits[base + i] - m);
      for (let i = 0; i < n; i++) p[i] /= s;
      return p;
    };
    const probs = (logits: ArrayLike<number>, row = 0): number[][] => {
      const out: number[][] = [];
      let o = row * O;
      for (const n of B) {
        out.push(softmaxRow(logits, o, n));
        o += n;
      }
      return out;
    };
    const opt = (net: Net, lr = 3e-4, kind: OptimizerKind = 'adam') => {
      const o = makeOptimizer(kind, net.params, lr);
      self.optimizers.push(o);
      return {
        step: () => o.step(),
        get lr() {
          return o.lr;
        },
        set lr(v: number) {
          o.lr = v;
        },
      };
    };
    return {
      obsSize: vec.obsSize,
      numEnvs: vec.n,
      outputs: O,
      branches: B.slice(),
      actions: space.labels.map((name, k) => ({ name, choices: space.optionLabels[k].slice() })),
      get steps() {
        return vec.totalSteps;
      },
      mlp(hidden = [32, 32], activation: Activation = 'tanh', outSize = O, inSize = vec.obsSize) {
        if (!(ACTIVATIONS as string[]).includes(activation)) throw new Error(`unknown activation '${activation}'`);
        const layers = (Array.isArray(hidden) ? hidden : [hidden]).map((u) => ({ units: Math.max(1, Math.min(512, Math.round(Number(u) || 32))), activation }));
        const net = new Net(new MLP(inSize, layers, outSize, 'linear', rng, outSize === O ? 0.01 : 1));
        if (!self.firstNet && outSize === O && inSize === vec.obsSize) self.firstNet = net;
        return net;
      },
      optimizer: opt,
      adam: (net, lr = 3e-4) => opt(net, lr, 'adam'),
      sgd: (net, lr = 1e-2) => opt(net, lr, 'sgd'),
      clipGrad: (net, maxNorm) => clipGradNorm(net.params, maxNorm),
      probs,
      sample(logits, row = 0) {
        return probs(logits, row).map((p) => {
          let u = rng.next();
          for (let i = 0; i < p.length; i++) {
            u -= p[i];
            if (u <= 0) return i;
          }
          return p.length - 1;
        });
      },
      best(logits, row = 0) {
        return probs(logits, row).map((p) => p.indexOf(Math.max(...p)));
      },
      logProb(logits, action, row = 0) {
        return probs(logits, row).reduce((s, p, k) => s + Math.log(Math.max(1e-12, p[action[k] | 0] ?? 1e-12)), 0);
      },
      entropy(logits, row = 0) {
        return probs(logits, row).reduce((s, p) => s - p.reduce((a, q) => a + (q > 0 ? q * Math.log(q) : 0), 0), 0);
      },
      // d/dlogits of scale · log π(a|s) = scale · (onehot(a) − p), per action branch
      addLogProbGrad(logits, row, action, scale, out) {
        let o = row * O;
        probs(logits, row).forEach((p, k) => {
          const a = action[k] | 0;
          for (let i = 0; i < p.length; i++) out[o + i] += scale * ((i === a ? 1 : 0) - p[i]);
          o += p.length;
        });
      },
      // d/dlogits of scale · H(π), H = −Σ p log p
      addEntropyGrad(logits, row, scale, out) {
        let o = row * O;
        probs(logits, row).forEach((p) => {
          let h = 0;
          for (const q of p) h -= q > 0 ? q * Math.log(q) : 0;
          for (let i = 0; i < p.length; i++) out[o + i] += scale * -p[i] * (Math.log(Math.max(p[i], 1e-12)) + h);
          o += p.length;
        });
      },
      batch(rows) {
        if (!rows.length) return new Float32Array(0);
        const n = rows[0].length;
        const out = new Float32Array(rows.length * n);
        rows.forEach((r, i) => {
          for (let j = 0; j < n; j++) out[i * n + j] = r[j];
        });
        return out;
      },
      worlds<T>(make: () => T): T[] {
        return Array.from({ length: vec.n }, make);
      },
      random: () => rng.next(),
      gauss: () => rng.gauss(),
      randint: (n) => rng.int(Math.max(1, Math.floor(n))),
      log(metrics) {
        const m: Record<string, number> = {};
        for (const [k, v] of Object.entries(metrics ?? {})) if (typeof v === 'number' && Number.isFinite(v)) m[k.slice(0, 32)] = v;
        self.lastMetrics = m;
        self.stats.addMetrics({ step: vec.totalSteps, update: self.updates, ...m });
      },
    };
  }

  work(): void {
    const vec = this.vec;
    const S = vec.obsSize;
    const D = vec.actDims;
    this.prevObs.set(vec.obs);
    for (let i = 0; i < vec.n; i++) {
      const obs = vec.obs.slice(i * S, i * S + S);
      const a = this.call('act', () => this.def.act!.call(this.def, obs, this.ctx));
      const arr = typeof a === 'number' ? [a] : a;
      const ch = this.actChoices[i];
      for (let k = 0; k < D; k++) {
        const v = Math.round(Number(arr?.[k] ?? 0));
        ch[k] = Math.max(0, Math.min(this.branches[k] - 1, Number.isFinite(v) ? v : 0));
        this.act[i * D + k] = ch[k];
      }
    }
    vec.step(this.act);
    if (this.def.observe) {
      for (let i = 0; i < vec.n; i++) {
        const done = !!vec.dones[i];
        const t: Transition = {
          obs: this.prevObs.slice(i * S, i * S + S),
          action: this.actChoices[i].slice(),
          reward: vec.rewards[i],
          done,
          timeout: !!vec.truncs[i],
          success: !!vec.successes[i],
          next: (done ? vec.termObs : vec.obs).slice(i * S, i * S + S),
          world: i,
        };
        this.call('observe', () => this.def.observe!.call(this.def, t, this.ctx));
      }
    }
    const changed = this.call('update', () => this.def.update!.call(this.def, this.ctx));
    if (changed === true) this.updates++;
    this.phase = `your algorithm · ${this.updates} updates`;
  }

  snapshot(): BrainSnapshot {
    const p = this.policy()!;
    return {
      format: 'neuroblocks-brain',
      version: 1,
      algo: 'custom',
      kind: 'policy',
      net: p.mlp.toJSON(),
      critic: null,
      logStd: null,
      obsNorm: null,
      space: {
        kind: 'discrete',
        branches: this.branches.slice(),
        labels: this.spec.axes.map((a) => a.axis),
        optionLabels: this.spec.axes.map((a) => a.labels.slice()),
      },
      obsLabels: this.vec.envs[0].layout.labels.slice(),
      steps: this.vec.totalSteps,
      updates: this.updates,
    };
  }

  explain(): Record<string, number | string> {
    return { updates: this.updates, params: this.policy()?.numParams ?? 0, ...this.lastMetrics };
  }
}
