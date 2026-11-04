/**
 * Behaviour Cloning: plain SUPERVISED learning from your demonstrations.
 *
 * You drive the robot; we record (what it saw, what you pressed) pairs.
 * The network is trained like an image classifier: input = observation,
 * label = your action. Loss = cross-entropy  −log π(a_you | s).
 *
 * Teaches the difference between supervised learning (copy the teacher) and
 * reinforcement learning (discover by trial and error). A cloned brain is a
 * great starting point for PPO fine-tuning.
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

export interface DemoSet {
  obs: Float32Array;
  acts: Float32Array;
  count: number;
  obsSize: number;
  actDims: number;
}

export class BehaviorCloning implements Algorithm {
  readonly kind = 'bc' as const;
  phase = 'starting';
  stats = new StatsLog();
  updates = 0;

  private vec: VecEnv;
  private spec: AgentSpec;
  private space: ActionSpace;
  private demos: DemoSet;
  private rng: RNG;
  actor: MLP;
  private head: PolicyHead;
  private norm: RunningMeanStd;
  private nDemo: Float32Array;
  private opt: Optimizer;
  private perm: Int32Array;
  private epoch = 0;
  private mb = 0;
  private sinceEval = 0;
  private acc = { loss: 0, correct: 0, n: 0 };
  private last: Record<string, number> = {};
  private nobs: Float32Array;
  private actBuf: Float32Array;

  constructor(vec: VecEnv, spec: AgentSpec, space: ActionSpace, demos: DemoSet, init: BrainSnapshot | null) {
    this.vec = vec;
    this.spec = spec;
    this.space = space;
    this.demos = demos;
    if (demos.obsSize !== vec.obsSize || demos.actDims !== vec.actDims)
      throw new Error('The recorded demonstrations were made with different senses/actions. Record new ones for this setup.');
    this.rng = new RNG(spec.train.seed * 41 + 1);
    this.head = new PolicyHead(space);
    const hidden = makeHiddenLayers(spec);
    this.actor = new MLP(vec.obsSize, hidden, this.head.outSize, 'linear', this.rng, 0.01);
    if (init && init.kind === 'policy' && this.actor.sameShape(init.net)) this.actor = MLP.fromJSON(init.net);
    // fit the observation normaliser to the demonstrations once
    this.norm = new RunningMeanStd(vec.obsSize, 5);
    this.norm.update(demos.obs, demos.count);
    this.nDemo = new Float32Array(demos.count * vec.obsSize);
    this.norm.normalize(demos.obs, this.nDemo, demos.count);
    this.opt = makeOptimizer(spec.train.optimizer, this.actor.params('actor.'), Math.max(1e-4, spec.train.lr * 3));
    this.perm = shuffleIndices(demos.count, this.rng);
    this.nobs = new Float32Array(vec.n * vec.obsSize);
    this.actBuf = new Float32Array(vec.n * vec.actDims);
    vec.onEpisode = (r) => this.stats.addEpisode(r);
    vec.reset();
  }

  work(): void {
    // interleave: 6 learning steps, then 1 step of testing the copy in the world
    if (this.sinceEval >= 6) {
      this.sinceEval = 0;
      this.evalStep();
      return;
    }
    this.sinceEval++;
    this.learn();
  }

  private evalStep(): void {
    const { vec } = this;
    this.norm.normalize(vec.obs, this.nobs, vec.n);
    const out = this.actor.forward(this.nobs, vec.n);
    for (let i = 0; i < vec.n; i++) this.head.sample(out, i * this.head.outSize, null, this.rng, this.actBuf, i * vec.actDims, true);
    vec.step(this.actBuf);
  }

  private learn(): void {
    const { demos } = this;
    const S = demos.obsSize;
    const D = demos.actDims;
    const O = this.head.outSize;
    const M = Math.min(demos.count, 128);
    const numMb = Math.max(1, Math.floor(demos.count / M));
    const x = new Float32Array(M * S);
    const a = new Float32Array(M * D);
    for (let j = 0; j < M; j++) {
      const k = this.perm[(this.mb * M + j) % demos.count];
      x.set(this.nDemo.subarray(k * S, k * S + S), j * S);
      a.set(demos.acts.subarray(k * D, k * D + D), j * D);
    }
    this.opt.zeroGrad();
    const out = this.actor.forward(x, M);
    const dOut = new Float32Array(M * O);
    let loss = 0;
    let correct = 0;
    const tmp = new Float32Array(D);
    for (let j = 0; j < M; j++) {
      if (this.space.kind === 'discrete') {
        loss += -this.head.logProb(out, j * O, null, a, j * D);
        // maximise log-likelihood of the demonstrated action
        this.head.accumGrad(out, j * O, null, a, j * D, -1 / M, 0, dOut, j * O, null);
        this.head.sample(out, j * O, null, this.rng, tmp, 0, true);
        let ok = true;
        for (let d = 0; d < D; d++) if (tmp[d] !== a[j * D + d]) ok = false;
        if (ok) correct++;
      } else {
        for (let d = 0; d < D; d++) {
          const e = out[j * O + d] - a[j * D + d];
          loss += e * e;
          dOut[j * O + d] = (2 * e) / M;
          if (Math.abs(e) < 0.25) correct += 1 / D;
        }
      }
    }
    this.actor.backward(dOut, M);
    clipGradNorm(this.opt.params, 5);
    this.opt.step();
    this.updates++;
    this.acc.loss += loss / M;
    this.acc.correct += correct / M;
    this.acc.n++;
    this.mb++;
    this.phase = `copying your driving: epoch ${this.epoch + 1}, batch ${this.mb}/${numMb}`;
    if (this.mb >= numMb) {
      this.mb = 0;
      this.epoch++;
      shuffleIndices(demos.count, this.rng, this.perm);
      this.last = {
        epoch: this.epoch,
        loss: this.acc.loss / this.acc.n,
        accuracy: this.acc.correct / this.acc.n,
        demos: demos.count,
        meanReward: this.stats.meanRecent(),
      };
      this.stats.addMetrics({ step: this.vec.totalSteps, update: this.updates, ...this.last });
      this.acc = { loss: 0, correct: 0, n: 0 };
    }
  }

  explain(): Record<string, number | string> {
    return { algo: 'Behaviour Cloning', demos: this.demos.count, updates: this.updates, ...this.last };
  }

  snapshot(): BrainSnapshot {
    return {
      format: 'neuroblocks-brain',
      version: 1,
      algo: 'bc',
      kind: 'policy',
      net: this.actor.toJSON(),
      critic: null,
      logStd: this.space.kind === 'continuous' ? new Array(this.vec.actDims).fill(-1) : null,
      obsNorm: this.norm.toJSON(),
      space: this.space,
      obsLabels: this.vec.envs[0].layout.labels,
      steps: this.vec.totalSteps,
      updates: this.updates,
    };
  }
}
