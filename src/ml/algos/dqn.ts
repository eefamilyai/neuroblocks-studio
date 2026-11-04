/**
 * DQN: Deep Q-Network (Mnih et al., 2015), the algorithm that learned Atari.
 *
 * Instead of a policy, the network predicts Q(s, a): "how much total future
 * reward will I get if I take action a now and play well afterwards?"
 * The agent then simply picks the action with the highest Q.
 *
 *  • Explore with ε-greedy: with probability ε act randomly (ε shrinks over time).
 *  • Store every transition (s, a, r, s′) in a REPLAY BUFFER and learn from
 *    random minibatches of old memories (breaks correlations).
 *  • Bellman target:  y = r + γ · max_a′ Q_target(s′, a′)
 *    using a slowly-updated TARGET NETWORK for stability.
 *  • Double DQN: pick a′ with the online net, score it with the target net
 *    (reduces over-optimism).
 *  • Loss: Huber(Q(s,a) − y), robust to outliers.
 */
import type { AgentSpec } from '../../sim/types.ts';
import { MLP } from '../nn.ts';
import { clipGradNorm, makeOptimizer } from '../optim.ts';
import type { Optimizer } from '../optim.ts';
import { decodeJoint, jointActionCount } from '../policy.ts';
import type { ActionSpace } from '../policy.ts';
import { RNG } from '../rng.ts';
import { StatsLog, VecEnv, makeHiddenLayers } from './common.ts';
import type { Algorithm, BrainSnapshot } from './common.ts';

export class DQN implements Algorithm {
  readonly kind = 'dqn' as const;
  phase = 'starting';
  stats = new StatsLog();
  updates = 0;

  private vec: VecEnv;
  private spec: AgentSpec;
  private space: ActionSpace;
  private rng: RNG;
  q: MLP;
  target: MLP;
  private opt: Optimizer;
  private J: number;
  private S: number;
  private N: number;
  private cap: number;
  private rS: Float32Array;
  private rS2: Float32Array;
  private rA: Int32Array;
  private rR: Float32Array;
  private rD: Uint8Array;
  private size = 0;
  private ptr = 0;
  private pending = 0;
  private sinceSync = 0;
  eps = 1;
  private prevObs: Float32Array;
  private actBuf: Float32Array;
  private jointBuf: Int32Array;
  private bS: Float32Array;
  private bS2: Float32Array;
  private dq: Float32Array;
  private acc = { loss: 0, q: 0, td: 0, gn: 0, k: 0 };
  private last: Record<string, number> = {};
  private syncs = 0;

  constructor(vec: VecEnv, spec: AgentSpec, space: ActionSpace, init: BrainSnapshot | null) {
    this.vec = vec;
    this.spec = spec;
    this.space = space;
    const tr = spec.train;
    this.rng = new RNG(tr.seed * 13 + 5);
    this.S = vec.obsSize;
    this.N = vec.n;
    this.J = jointActionCount(space);
    const hidden = makeHiddenLayers(spec);
    this.q = new MLP(this.S, hidden, this.J, 'linear', this.rng, 0.1);
    if (init && init.kind === 'q' && this.q.sameShape(init.net)) {
      this.q = MLP.fromJSON(init.net);
      this.updates = init.updates;
    }
    this.target = this.q.clone();
    this.opt = makeOptimizer(tr.optimizer, this.q.params('q.'), tr.lr);
    this.cap = Math.max(1000, Math.round(tr.bufferSize));
    this.rS = new Float32Array(this.cap * this.S);
    this.rS2 = new Float32Array(this.cap * this.S);
    this.rA = new Int32Array(this.cap);
    this.rR = new Float32Array(this.cap);
    this.rD = new Uint8Array(this.cap);
    this.prevObs = new Float32Array(this.N * this.S);
    this.actBuf = new Float32Array(this.N * vec.actDims);
    this.jointBuf = new Int32Array(this.N);
    const B = Math.max(8, tr.batchSize);
    this.bS = new Float32Array(B * this.S);
    this.bS2 = new Float32Array(B * this.S);
    this.dq = new Float32Array(B * this.J);
    vec.onEpisode = (r) => this.stats.addEpisode(r);
    vec.reset();
  }

  work(): void {
    const tr = this.spec.train;
    if (this.pending >= 1 && this.size >= Math.max(tr.learnStart, tr.batchSize)) {
      this.learn();
      this.pending -= 1;
      this.phase = `learning from replay memory (${this.size.toLocaleString()} memories)`;
      return;
    }
    this.collect();
    this.phase =
      this.size < tr.learnStart
        ? `filling replay memory ${this.size}/${tr.learnStart} (random-ish play)`
        : `playing with ε = ${this.eps.toFixed(2)} (explore ${(this.eps * 100).toFixed(0)}% of the time)`;
  }

  private collect(): void {
    const tr = this.spec.train;
    const { N, S, J } = this;
    const frac = Math.min(1, this.vec.totalSteps / Math.max(1, tr.epsDecay));
    this.eps = tr.epsStart + (tr.epsEnd - tr.epsStart) * frac;
    const qv = this.q.forward(this.vec.obs, N);
    for (let i = 0; i < N; i++) {
      let a: number;
      if (this.rng.next() < this.eps) a = this.rng.int(J);
      else {
        a = 0;
        for (let j = 1; j < J; j++) if (qv[i * J + j] > qv[i * J + a]) a = j;
      }
      this.jointBuf[i] = a;
      decodeJoint(this.space, a, this.actBuf, i * this.vec.actDims);
    }
    this.prevObs.set(this.vec.obs);
    this.vec.step(this.actBuf);
    for (let i = 0; i < N; i++) {
      const p = this.ptr;
      this.rS.set(this.prevObs.subarray(i * S, i * S + S), p * S);
      const next = this.vec.dones[i] ? this.vec.termObs : this.vec.obs;
      this.rS2.set(next.subarray(i * S, i * S + S), p * S);
      this.rA[p] = this.jointBuf[i];
      this.rR[p] = this.vec.rewards[i];
      // a time-limit cut is not a real "game over": keep bootstrapping through it
      this.rD[p] = this.vec.dones[i] && !this.vec.truncs[i] ? 1 : 0;
      this.ptr = (p + 1) % this.cap;
      this.size = Math.min(this.cap, this.size + 1);
    }
    this.pending += N / Math.max(1, tr.trainEvery);
    this.sinceSync += N;
    if (this.sinceSync >= tr.targetUpdate) {
      this.target.copyFrom(this.q);
      this.sinceSync = 0;
      this.syncs++;
    }
  }

  private learn(): void {
    const tr = this.spec.train;
    const { S, J } = this;
    const B = Math.max(8, tr.batchSize);
    const idx = new Int32Array(B);
    for (let j = 0; j < B; j++) {
      const k = this.rng.int(this.size);
      idx[j] = k;
      this.bS.set(this.rS.subarray(k * S, k * S + S), j * S);
      this.bS2.set(this.rS2.subarray(k * S, k * S + S), j * S);
    }
    // targets (no gradient)
    const qT = Float32Array.from(this.target.forward(this.bS2, B).subarray(0, B * J));
    const qO = tr.double ? Float32Array.from(this.q.forward(this.bS2, B).subarray(0, B * J)) : null;
    const y = new Float32Array(B);
    for (let j = 0; j < B; j++) {
      let best = 0;
      const src = qO ?? qT;
      for (let a = 1; a < J; a++) if (src[j * J + a] > src[j * J + best]) best = a;
      const k = idx[j];
      y[j] = this.rR[k] + (this.rD[k] ? 0 : tr.gamma * qT[j * J + best]);
    }
    // online network: Huber loss on the taken action only
    this.opt.zeroGrad();
    const qs = this.q.forward(this.bS, B);
    this.dq.fill(0);
    let loss = 0;
    let qm = 0;
    let tdm = 0;
    for (let j = 0; j < B; j++) {
      const a = this.rA[idx[j]];
      const e = qs[j * J + a] - y[j];
      const ae = Math.abs(e);
      loss += ae <= 1 ? 0.5 * e * e : ae - 0.5;
      this.dq[j * J + a] = (ae <= 1 ? e : Math.sign(e)) / B;
      qm += qs[j * J + a];
      tdm += ae;
    }
    this.q.backward(this.dq, B);
    const gn = clipGradNorm(this.opt.params, 10);
    this.opt.step();
    this.updates++;
    const acc = this.acc;
    acc.loss += loss / B;
    acc.q += qm / B;
    acc.td += tdm / B;
    acc.gn += gn;
    acc.k++;
    if (acc.k >= 100) {
      this.last = {
        loss: acc.loss / acc.k,
        meanQ: acc.q / acc.k,
        tdError: acc.td / acc.k,
        gradNorm: acc.gn / acc.k,
        epsilon: this.eps,
        replaySize: this.size,
        targetSyncs: this.syncs,
        lr: tr.lr,
        meanReward: this.stats.meanRecent(),
      };
      this.stats.addMetrics({ step: this.vec.totalSteps, update: this.updates, ...this.last });
      this.acc = { loss: 0, q: 0, td: 0, gn: 0, k: 0 };
    }
  }

  explain(): Record<string, number | string> {
    const tr = this.spec.train;
    return {
      algo: 'DQN',
      worlds: this.N,
      jointActions: this.J,
      gamma: tr.gamma,
      epsilon: this.eps,
      replaySize: this.size,
      replayCapacity: this.cap,
      batch: tr.batchSize,
      targetUpdate: tr.targetUpdate,
      double: tr.double ? 'yes' : 'no',
      updates: this.updates,
      lr: tr.lr,
      ...this.last,
    };
  }

  snapshot(): BrainSnapshot {
    return {
      format: 'neuroblocks-brain',
      version: 1,
      algo: 'dqn',
      kind: 'q',
      net: this.q.toJSON(),
      critic: null,
      logStd: null,
      obsNorm: null,
      space: this.space,
      obsLabels: this.vec.envs[0].layout.labels,
      steps: this.vec.totalSteps,
      updates: this.updates,
    };
  }
}
