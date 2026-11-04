/**
 * PPO: Proximal Policy Optimization (Schulman et al., 2017).
 *
 * Two networks:
 *   actor  π(a|s): "what should I do?"  (outputs action probabilities)
 *   critic V(s):   "how good is this situation?" (predicts future reward)
 *
 * Loop:
 *  1. COLLECT: run the current policy in N parallel worlds for T steps.
 *  2. ADVANTAGES (GAE): for each step, was the outcome better or worse than
 *     the critic expected?   δ_t = r_t + γ·V(s_{t+1}) − V(s_t)
 *                            A_t = δ_t + γλ·A_{t+1}
 *  3. LEARN: a few epochs of minibatch gradient descent on
 *        L = −min(ρ·A, clip(ρ, 1−ε, 1+ε)·A)  −  β·entropy  +  c·(V − R)²
 *     where ρ = π_new(a|s) / π_old(a|s). The clip stops any single update
 *     from changing the policy too much ("proximal").
 */
import type { AgentSpec } from '../../sim/types.ts';
import { MLP } from '../nn.ts';
import { RunningMeanStd } from '../normalizer.ts';
import { clipGradNorm, makeOptimizer } from '../optim.ts';
import type { Optimizer } from '../optim.ts';
import { PolicyHead } from '../policy.ts';
import type { ActionSpace } from '../policy.ts';
import { RNG } from '../rng.ts';
import { StatsLog, VecEnv, makeHiddenLayers, shuffleIndices } from './common.ts';
import type { Algorithm, BrainSnapshot } from './common.ts';

export class PPO implements Algorithm {
  readonly kind = 'ppo' as const;
  phase = 'starting';
  stats = new StatsLog();
  updates = 0;

  private vec: VecEnv;
  private spec: AgentSpec;
  private rng: RNG;
  actor: MLP;
  critic: MLP;
  logStd: Float32Array | null;
  private dLogStd: Float32Array | null;
  head: PolicyHead;
  obsRms: RunningMeanStd;
  private retRms = new RunningMeanStd(1, 1e9);
  private retAcc: Float64Array;
  private optA: Optimizer;
  private optC: Optimizer;
  private space: ActionSpace;

  private T: number;
  private N: number;
  private S: number;
  private D: number;
  private O: number;
  private bObs: Float32Array;
  private bAct: Float32Array;
  private bLogp: Float32Array;
  private bVal: Float32Array;
  private bRew: Float32Array;
  private bDone: Uint8Array;
  private bAdv: Float32Array;
  private bRet: Float32Array;
  private nobs: Float32Array;
  private actBuf: Float32Array;
  private t = 0;
  private epoch = 0;
  private mb = 0;
  private perm: Int32Array;
  private mbObs: Float32Array;
  private mbAct: Float32Array;
  private dOut: Float32Array;
  private dV: Float32Array;
  private acc = { pl: 0, vl: 0, ent: 0, kl: 0, clip: 0, n: 0, gnA: 0, gnC: 0, k: 0 };
  private earlyStop = false;
  private last: Record<string, number> = {};
  private sampleRow: Record<string, number> = {};

  constructor(vec: VecEnv, spec: AgentSpec, space: ActionSpace, init: BrainSnapshot | null) {
    this.vec = vec;
    this.spec = spec;
    this.space = space;
    const tr = spec.train;
    this.rng = new RNG(tr.seed * 7 + 11);
    this.head = new PolicyHead(space);
    this.S = vec.obsSize;
    this.D = vec.actDims;
    this.O = this.head.outSize;
    this.N = vec.n;
    this.T = Math.max(8, Math.round(tr.horizon));
    const hidden = makeHiddenLayers(spec);
    this.actor = new MLP(this.S, hidden, this.O, 'linear', this.rng, 0.01);
    this.critic = new MLP(this.S, hidden, 1, 'linear', this.rng, 1);
    this.logStd = space.kind === 'continuous' ? new Float32Array(this.D).fill(-0.5) : null;
    this.dLogStd = this.logStd ? new Float32Array(this.D) : null;
    this.obsRms = new RunningMeanStd(this.S, 5);
    if (init && init.kind === 'policy' && init.net.inSize === this.S && init.net.layers[init.net.layers.length - 1].outSize === this.O) {
      if (this.actor.sameShape(init.net)) this.actor = MLP.fromJSON(init.net);
      if (init.critic && this.critic.sameShape(init.critic)) this.critic = MLP.fromJSON(init.critic);
      if (init.logStd && this.logStd) this.logStd.set(init.logStd);
      if (init.obsNorm) this.obsRms = RunningMeanStd.fromJSON(init.obsNorm);
      this.updates = init.updates;
    }
    const aParams = this.actor.params('actor.');
    if (this.logStd && this.dLogStd) aParams.push({ w: this.logStd, g: this.dLogStd, name: 'actor.log_std' });
    this.optA = makeOptimizer(tr.optimizer, aParams, tr.lr);
    this.optC = makeOptimizer(tr.optimizer, this.critic.params('critic.'), tr.lr);
    const B = this.T * this.N;
    this.bObs = new Float32Array(B * this.S);
    this.bAct = new Float32Array(B * this.D);
    this.bLogp = new Float32Array(B);
    this.bVal = new Float32Array(B);
    this.bRew = new Float32Array(B);
    this.bDone = new Uint8Array(B);
    this.bAdv = new Float32Array(B);
    this.bRet = new Float32Array(B);
    this.nobs = new Float32Array(this.N * this.S);
    this.actBuf = new Float32Array(this.N * this.D);
    this.retAcc = new Float64Array(this.N);
    this.perm = new Int32Array(B);
    const M = Math.min(B, Math.max(16, tr.minibatch));
    this.mbObs = new Float32Array(M * this.S);
    this.mbAct = new Float32Array(M * this.D);
    this.dOut = new Float32Array(M * this.O);
    this.dV = new Float32Array(M);
    vec.onEpisode = (r) => this.stats.addEpisode(r);
    vec.reset();
  }

  setLearningRate(lr: number): void {
    this.optA.lr = lr;
    this.optC.lr = lr;
  }

  private normalizeInto(src: Float32Array, dst: Float32Array, n: number, update: boolean): void {
    if (this.spec.train.normObs) {
      if (update) this.obsRms.update(src, n);
      this.obsRms.normalize(src, dst, n);
    } else dst.set(src.subarray(0, n * this.S));
  }

  work(): void {
    if (this.t < this.T) {
      this.collectStep();
      this.phase = `collecting experience: step ${this.t}/${this.T} in ${this.N} worlds`;
      if (this.t === this.T) this.computeAdvantages();
      return;
    }
    this.learnMinibatch();
  }

  private collectStep(): void {
    const { N, S, D, O } = this;
    const tr = this.spec.train;
    const base = this.t * N;
    this.normalizeInto(this.vec.obs, this.nobs, N, true);
    this.bObs.set(this.nobs, base * S);
    const out = this.actor.forward(this.nobs, N);
    const vals = this.critic.forward(this.nobs, N);
    for (let i = 0; i < N; i++) {
      this.bVal[base + i] = vals[i];
      this.bLogp[base + i] = this.head.sample(out, i * O, this.logStd, this.rng, this.actBuf, i * D, false);
    }
    this.bAct.set(this.actBuf, base * D);
    this.vec.step(this.actBuf);
    // reward scaling: divide by the std of the running discounted return
    let truncNeeded = false;
    for (let i = 0; i < N; i++) {
      let r = this.vec.rewards[i];
      if (tr.normReward) {
        this.retAcc[i] = this.retAcc[i] * tr.gamma + r;
        this.retRms.update(Float32Array.of(this.retAcc[i]), 1);
        r = r / Math.sqrt(this.retRms.var[0] + 1e-8);
        r = Math.max(-10, Math.min(10, r));
        if (this.vec.dones[i]) this.retAcc[i] = 0;
      }
      this.bRew[base + i] = r;
      this.bDone[base + i] = this.vec.dones[i];
      if (this.vec.truncs[i]) truncNeeded = true;
    }
    // time-limit truncation is not a real ending: bootstrap with V(final state)
    if (truncNeeded) {
      const tn = new Float32Array(S);
      for (let i = 0; i < N; i++) {
        if (!this.vec.truncs[i]) continue;
        this.normalizeInto(this.vec.termObs.subarray(i * S, (i + 1) * S), tn, 1, false);
        const v = this.critic.forward(tn, 1)[0];
        this.bRew[base + i] += tr.gamma * v;
      }
    }
    this.t++;
  }

  private computeAdvantages(): void {
    const { N, T } = this;
    const tr = this.spec.train;
    this.normalizeInto(this.vec.obs, this.nobs, N, false);
    const lastV = Float32Array.from(this.critic.forward(this.nobs, N).subarray(0, N));
    const gae = new Float64Array(N);
    for (let t = T - 1; t >= 0; t--) {
      for (let i = 0; i < N; i++) {
        const k = t * N + i;
        const nonTerm = 1 - this.bDone[k];
        const nextV = t === T - 1 ? lastV[i] : this.bVal[k + N];
        const delta = this.bRew[k] + tr.gamma * nextV * nonTerm - this.bVal[k];
        gae[i] = delta + tr.gamma * tr.lambda * nonTerm * gae[i];
        this.bAdv[k] = gae[i];
        this.bRet[k] = gae[i] + this.bVal[k];
      }
    }
    // explained variance of the critic (1 = perfect, 0 = useless)
    const B = T * N;
    let mr = 0;
    for (let k = 0; k < B; k++) mr += this.bRet[k];
    mr /= B;
    let vr = 0;
    let ve = 0;
    let me = 0;
    for (let k = 0; k < B; k++) me += this.bRet[k] - this.bVal[k];
    me /= B;
    for (let k = 0; k < B; k++) {
      vr += (this.bRet[k] - mr) ** 2;
      ve += (this.bRet[k] - this.bVal[k] - me) ** 2;
    }
    const explVar = vr > 1e-12 ? 1 - ve / vr : 0;
    // remember one sample for the explainer
    const k = Math.floor(this.rng.next() * B);
    this.sampleRow = { reward: this.bRew[k], value: this.bVal[k], advantage: this.bAdv[k], ret: this.bRet[k] };
    // normalise advantages
    let ma = 0;
    for (let k2 = 0; k2 < B; k2++) ma += this.bAdv[k2];
    ma /= B;
    let va = 0;
    for (let k2 = 0; k2 < B; k2++) va += (this.bAdv[k2] - ma) ** 2;
    const sa = Math.sqrt(va / B) + 1e-8;
    for (let k2 = 0; k2 < B; k2++) this.bAdv[k2] = (this.bAdv[k2] - ma) / sa;
    this.last.explainedVariance = explVar;
    this.epoch = 0;
    this.mb = 0;
    this.earlyStop = false;
    this.acc = { pl: 0, vl: 0, ent: 0, kl: 0, clip: 0, n: 0, gnA: 0, gnC: 0, k: 0 };
    shuffleIndices(B, this.rng, this.perm);
  }

  private learnMinibatch(): void {
    const tr = this.spec.train;
    const { S, D, O } = this;
    const B = this.T * this.N;
    const M = Math.min(B, Math.max(16, tr.minibatch));
    const numMb = Math.max(1, Math.floor(B / M));
    this.phase = `learning: epoch ${this.epoch + 1}/${tr.epochs}, minibatch ${this.mb + 1}/${numMb}`;
    const idx = this.perm.subarray(this.mb * M, this.mb * M + M);
    for (let j = 0; j < M; j++) {
      const k = idx[j];
      this.mbObs.set(this.bObs.subarray(k * S, k * S + S), j * S);
      this.mbAct.set(this.bAct.subarray(k * D, k * D + D), j * D);
    }
    // ---- actor ----
    this.optA.zeroGrad();
    const out = this.actor.forward(this.mbObs, M);
    this.dOut.fill(0);
    let pl = 0;
    let ent = 0;
    let kl = 0;
    let clipN = 0;
    const eps = tr.clip;
    for (let j = 0; j < M; j++) {
      const k = idx[j];
      const lp = this.head.logProb(out, j * O, this.logStd, this.mbAct, j * D);
      const logRatio = lp - this.bLogp[k];
      const ratio = Math.exp(Math.max(-20, Math.min(20, logRatio)));
      const A = this.bAdv[k];
      const s1 = ratio * A;
      const s2 = Math.max(1 - eps, Math.min(1 + eps, ratio)) * A;
      pl += -Math.min(s1, s2);
      const clipped = (A >= 0 && ratio > 1 + eps) || (A < 0 && ratio < 1 - eps);
      if (clipped) clipN++;
      kl += ratio - 1 - logRatio;
      const h = this.head.entropy(out, j * O, this.logStd);
      ent += h;
      const gLogp = clipped ? 0 : -ratio * A;
      this.head.accumGrad(out, j * O, this.logStd, this.mbAct, j * D, gLogp / M, -tr.entropy / M, this.dOut, j * O, this.dLogStd);
    }
    this.actor.backward(this.dOut, M);
    const gnA = clipGradNorm(this.optA.params, tr.maxGradNorm);
    this.optA.step();
    // ---- critic ----
    this.optC.zeroGrad();
    const v = this.critic.forward(this.mbObs, M);
    let vl = 0;
    for (let j = 0; j < M; j++) {
      const e = v[j] - this.bRet[idx[j]];
      vl += 0.5 * e * e;
      this.dV[j] = (tr.valueCoef * e) / M;
    }
    this.critic.backward(this.dV, M);
    const gnC = clipGradNorm(this.optC.params, tr.maxGradNorm);
    this.optC.step();

    const a = this.acc;
    a.pl += pl / M;
    a.vl += vl / M;
    a.ent += ent / M;
    a.kl += kl / M;
    a.clip += clipN / M;
    a.gnA += gnA;
    a.gnC += gnC;
    a.k++;
    // stop this round early if the policy moved too far (approx. KL too big)
    if (kl / M > 0.05) this.earlyStop = true;

    this.mb++;
    if (this.mb >= numMb) {
      this.mb = 0;
      this.epoch++;
      shuffleIndices(B, this.rng, this.perm);
    }
    if (this.epoch >= tr.epochs || this.earlyStop) this.finishUpdate();
  }

  private finishUpdate(): void {
    const a = this.acc;
    const k = Math.max(1, a.k);
    this.updates++;
    const std = this.logStd ? Array.from(this.logStd, (x) => Math.exp(x)).reduce((p, q) => p + q, 0) / this.logStd.length : 0;
    this.last = {
      ...this.last,
      policyLoss: a.pl / k,
      valueLoss: a.vl / k,
      entropy: a.ent / k,
      approxKL: a.kl / k,
      clipFraction: a.clip / k,
      gradNormActor: a.gnA / k,
      gradNormCritic: a.gnC / k,
      updateRatio: this.optA.lastUpdateRatio,
      actionStd: std,
      earlyStop: this.earlyStop ? 1 : 0,
      lr: this.spec.train.lr,
      meanReward: this.stats.meanRecent(),
    };
    this.stats.addMetrics({ step: this.vec.totalSteps, update: this.updates, ...this.last });
    this.t = 0;
  }

  explain(): Record<string, number | string> {
    const tr = this.spec.train;
    return {
      algo: 'PPO',
      worlds: this.N,
      horizon: this.T,
      batch: this.T * this.N,
      minibatch: Math.min(this.T * this.N, tr.minibatch),
      epochs: tr.epochs,
      gamma: tr.gamma,
      lambda: tr.lambda,
      clip: tr.clip,
      entropyCoef: tr.entropy,
      lr: tr.lr,
      updates: this.updates,
      ...Object.fromEntries(Object.entries(this.last).map(([k, v]) => [k, v])),
      ...Object.fromEntries(Object.entries(this.sampleRow).map(([k, v]) => [`sample_${k}`, v])),
    };
  }

  snapshot(): BrainSnapshot {
    return {
      format: 'neuroblocks-brain',
      version: 1,
      algo: 'ppo',
      kind: 'policy',
      net: this.actor.toJSON(),
      critic: this.critic.toJSON(),
      logStd: this.logStd ? Array.from(this.logStd) : null,
      obsNorm: this.spec.train.normObs ? this.obsRms.toJSON() : null,
      space: this.space,
      obsLabels: this.vec.envs[0].layout.labels,
      steps: this.vec.totalSteps,
      updates: this.updates,
    };
  }
}
