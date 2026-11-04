/**
 * REINFORCE (Williams, 1992): the simplest policy-gradient algorithm.
 *
 *  1. Play whole episodes with the current policy.
 *  2. For every step compute the return  G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …
 *  3. Nudge the policy to make good actions more likely:
 *         ∇J ≈ Σ_t ∇log π(a_t|s_t) · (G_t − b(s_t))
 *     b(s) is an optional learned BASELINE (a critic) that reduces noise.
 *
 * Easy to understand, but noisy and sample-hungry: PPO builds on this idea.
 */
import type { AgentSpec } from '../../sim/types.ts';
import { MLP } from '../nn.ts';
import { RunningMeanStd } from '../normalizer.ts';
import { clipGradNorm, makeOptimizer } from '../optim.ts';
import type { Optimizer } from '../optim.ts';
import { PolicyHead } from '../policy.ts';
import type { ActionSpace } from '../policy.ts';
import { RNG } from '../rng.ts';
import { StatsLog, VecEnv, makeHiddenLayers } from './common.ts';
import type { Algorithm, BrainSnapshot } from './common.ts';

interface Traj {
  obs: Float32Array[];
  act: Float32Array[];
  rew: number[];
}

const CHUNK = 512;

export class Reinforce implements Algorithm {
  readonly kind = 'reinforce' as const;
  phase = 'starting';
  stats = new StatsLog();
  updates = 0;

  private vec: VecEnv;
  private spec: AgentSpec;
  private space: ActionSpace;
  private rng: RNG;
  actor: MLP;
  critic: MLP | null;
  logStd: Float32Array | null;
  private dLogStd: Float32Array | null;
  private head: PolicyHead;
  obsRms: RunningMeanStd;
  private optA: Optimizer;
  private optC: Optimizer | null;
  private S: number;
  private D: number;
  private O: number;
  private N: number;
  private traj: Traj[];
  private bObs: Float32Array[] = [];
  private bAct: Float32Array[] = [];
  private bRet: number[] = [];
  private episodesInBatch = 0;
  private nobs: Float32Array;
  private actBuf: Float32Array;
  private last: Record<string, number> = {};

  constructor(vec: VecEnv, spec: AgentSpec, space: ActionSpace, init: BrainSnapshot | null) {
    this.vec = vec;
    this.spec = spec;
    this.space = space;
    const tr = spec.train;
    this.rng = new RNG(tr.seed * 17 + 3);
    this.head = new PolicyHead(space);
    this.S = vec.obsSize;
    this.D = vec.actDims;
    this.O = this.head.outSize;
    this.N = vec.n;
    const hidden = makeHiddenLayers(spec);
    this.actor = new MLP(this.S, hidden, this.O, 'linear', this.rng, 0.01);
    this.critic = tr.baseline ? new MLP(this.S, hidden, 1, 'linear', this.rng, 1) : null;
    this.logStd = space.kind === 'continuous' ? new Float32Array(this.D).fill(-0.5) : null;
    this.dLogStd = this.logStd ? new Float32Array(this.D) : null;
    this.obsRms = new RunningMeanStd(this.S, 5);
    if (init && init.kind === 'policy' && this.actor.sameShape(init.net)) {
      this.actor = MLP.fromJSON(init.net);
      if (init.critic && this.critic && this.critic.sameShape(init.critic)) this.critic = MLP.fromJSON(init.critic);
      if (init.obsNorm) this.obsRms = RunningMeanStd.fromJSON(init.obsNorm);
      if (init.logStd && this.logStd) this.logStd.set(init.logStd);
    }
    const ap = this.actor.params('actor.');
    if (this.logStd && this.dLogStd) ap.push({ w: this.logStd, g: this.dLogStd, name: 'actor.log_std' });
    this.optA = makeOptimizer(tr.optimizer, ap, tr.lr);
    this.optC = this.critic ? makeOptimizer(tr.optimizer, this.critic.params('critic.'), tr.lr * 3) : null;
    this.traj = Array.from({ length: this.N }, () => ({ obs: [], act: [], rew: [] }));
    this.nobs = new Float32Array(this.N * this.S);
    this.actBuf = new Float32Array(this.N * this.D);
    vec.onEpisode = (r) => this.stats.addEpisode(r);
    vec.reset();
  }

  private norm(src: Float32Array, dst: Float32Array, n: number, update: boolean): void {
    if (this.spec.train.normObs) {
      if (update) this.obsRms.update(src, n);
      this.obsRms.normalize(src, dst, n);
    } else dst.set(src.subarray(0, n * this.S));
  }

  work(): void {
    const tr = this.spec.train;
    if (this.episodesInBatch >= Math.max(1, tr.episodesPerUpdate)) {
      this.learn();
      return;
    }
    const { N, S, D, O } = this;
    this.norm(this.vec.obs, this.nobs, N, true);
    const out = this.actor.forward(this.nobs, N);
    for (let i = 0; i < N; i++) {
      this.head.sample(out, i * O, this.logStd, this.rng, this.actBuf, i * D, false);
      this.traj[i].obs.push(this.nobs.slice(i * S, i * S + S));
      this.traj[i].act.push(this.actBuf.slice(i * D, i * D + D));
    }
    this.vec.step(this.actBuf);
    for (let i = 0; i < N; i++) {
      const tj = this.traj[i];
      tj.rew.push(this.vec.rewards[i]);
      if (!this.vec.dones[i]) continue;
      // episode finished: compute discounted returns backwards
      let G = 0;
      if (this.vec.truncs[i] && this.critic) {
        const t = new Float32Array(S);
        this.norm(this.vec.termObs.subarray(i * S, i * S + S), t, 1, false);
        G = this.critic.forward(t, 1)[0];
      }
      const rets = new Array<number>(tj.rew.length);
      for (let t = tj.rew.length - 1; t >= 0; t--) {
        G = tj.rew[t] + tr.gamma * G;
        rets[t] = G;
      }
      for (let t = 0; t < rets.length; t++) {
        this.bObs.push(tj.obs[t]);
        this.bAct.push(tj.act[t]);
        this.bRet.push(rets[t]);
      }
      tj.obs = [];
      tj.act = [];
      tj.rew = [];
      this.episodesInBatch++;
    }
    this.phase = `playing episodes: ${this.episodesInBatch}/${tr.episodesPerUpdate} finished (${this.bRet.length} steps saved)`;
  }

  private learn(): void {
    const tr = this.spec.train;
    const { S, D, O } = this;
    const B = this.bRet.length;
    this.phase = `learning from ${this.episodesInBatch} episodes (${B} steps)`;
    const adv = new Float32Array(B);
    let vl = 0;
    if (this.critic && this.optC) this.optC.zeroGrad();
    for (let c0 = 0; c0 < B; c0 += CHUNK) {
      const m = Math.min(CHUNK, B - c0);
      const x = new Float32Array(m * S);
      for (let j = 0; j < m; j++) x.set(this.bObs[c0 + j], j * S);
      if (this.critic) {
        const v = this.critic.forward(x, m);
        const dv = new Float32Array(m);
        for (let j = 0; j < m; j++) {
          const e = v[j] - this.bRet[c0 + j];
          adv[c0 + j] = this.bRet[c0 + j] - v[j];
          vl += 0.5 * e * e;
          dv[j] = e / B;
        }
        this.critic.backward(dv, m);
      } else for (let j = 0; j < m; j++) adv[c0 + j] = this.bRet[c0 + j];
    }
    if (this.critic && this.optC) {
      clipGradNorm(this.optC.params, 1);
      this.optC.step();
    }
    let ma = 0;
    for (let k = 0; k < B; k++) ma += adv[k];
    ma /= B;
    let va = 0;
    for (let k = 0; k < B; k++) va += (adv[k] - ma) ** 2;
    const sa = Math.sqrt(va / B) + 1e-8;
    for (let k = 0; k < B; k++) adv[k] = (adv[k] - ma) / sa;
    this.optA.zeroGrad();
    let pl = 0;
    let ent = 0;
    for (let c0 = 0; c0 < B; c0 += CHUNK) {
      const m = Math.min(CHUNK, B - c0);
      const x = new Float32Array(m * S);
      const a = new Float32Array(m * D);
      for (let j = 0; j < m; j++) {
        x.set(this.bObs[c0 + j], j * S);
        a.set(this.bAct[c0 + j], j * D);
      }
      const out = this.actor.forward(x, m);
      const dOut = new Float32Array(m * O);
      for (let j = 0; j < m; j++) {
        const A = adv[c0 + j];
        pl += -this.head.logProb(out, j * O, this.logStd, a, j * D) * A;
        ent += this.head.entropy(out, j * O, this.logStd);
        // L = −logπ·A − β·H  → ∂L/∂logπ = −A
        this.head.accumGrad(out, j * O, this.logStd, a, j * D, -A / B, -tr.entropy / B, dOut, j * O, this.dLogStd);
      }
      this.actor.backward(dOut, m);
    }
    const gn = clipGradNorm(this.optA.params, 1);
    this.optA.step();
    this.updates++;
    const meanRet = this.bRet.reduce((p, q) => p + q, 0) / Math.max(1, B);
    this.last = {
      policyLoss: pl / B,
      valueLoss: vl / B,
      entropy: ent / B,
      gradNorm: gn,
      batchSteps: B,
      meanReturn: meanRet,
      lr: tr.lr,
      meanReward: this.stats.meanRecent(),
    };
    this.stats.addMetrics({ step: this.vec.totalSteps, update: this.updates, ...this.last });
    this.bObs = [];
    this.bAct = [];
    this.bRet = [];
    this.episodesInBatch = 0;
  }

  explain(): Record<string, number | string> {
    const tr = this.spec.train;
    return {
      algo: 'REINFORCE',
      worlds: this.N,
      gamma: tr.gamma,
      episodesPerUpdate: tr.episodesPerUpdate,
      baseline: tr.baseline ? 'learned critic' : 'none',
      lr: tr.lr,
      updates: this.updates,
      ...this.last,
    };
  }

  snapshot(): BrainSnapshot {
    return {
      format: 'neuroblocks-brain',
      version: 1,
      algo: 'reinforce',
      kind: 'policy',
      net: this.actor.toJSON(),
      critic: this.critic ? this.critic.toJSON() : null,
      logStd: this.logStd ? Array.from(this.logStd) : null,
      obsNorm: this.spec.train.normObs ? this.obsRms.toJSON() : null,
      space: this.space,
      obsLabels: this.vec.envs[0].layout.labels,
      steps: this.vec.totalSteps,
      updates: this.updates,
    };
  }
}
