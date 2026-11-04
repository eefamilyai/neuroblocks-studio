/**
 * Neuroevolution: a genetic algorithm over network weights. No gradients!
 *
 *  1. Start with a POPULATION of random brains.
 *  2. Let each one play an episode; its total reward is its FITNESS.
 *  3. Keep the best few (the ELITES) unchanged.
 *  4. Fill the rest of the next generation with MUTATED copies of elites:
 *        child = parent + σ · noise
 *  5. Repeat. Like natural selection, but for neural networks.
 *
 * It is simple, parallel, and surprisingly strong on small control tasks.
 */
import type { AgentSpec } from '../../sim/types.ts';
import { MLP } from '../nn.ts';
import { PolicyHead } from '../policy.ts';
import type { ActionSpace } from '../policy.ts';
import { RNG } from '../rng.ts';
import { StatsLog, VecEnv, makeHiddenLayers } from './common.ts';
import type { Algorithm, BrainSnapshot } from './common.ts';

export class Evolution implements Algorithm {
  readonly kind = 'evolution' as const;
  phase = 'starting';
  stats = new StatsLog();
  updates = 0;

  private vec: VecEnv;
  private spec: AgentSpec;
  private space: ActionSpace;
  private rng: RNG;
  private head: PolicyHead;
  private P: number;
  private nets: MLP[];
  private flats: Float32Array[];
  private sums: Float64Array;
  private evals: Int32Array;
  private assign: Int32Array;
  private nextIdx = 0;
  private finished = 0;
  generation = 0;
  sigma: number;
  best: MLP;
  bestFitness = -Infinity;
  private actBuf: Float32Array;
  private last: Record<string, number> = {};
  private noop: Float32Array;
  /** fitness of every individual in the last finished generation (for the histogram) */
  lastFitness: number[] = [];

  constructor(vec: VecEnv, spec: AgentSpec, space: ActionSpace, init: BrainSnapshot | null) {
    this.vec = vec;
    this.spec = spec;
    this.space = space;
    const tr = spec.train;
    this.rng = new RNG(tr.seed * 31 + 7);
    this.head = new PolicyHead(space);
    this.P = Math.max(4, Math.round(tr.population));
    this.sigma = tr.mutation;
    const hidden = makeHiddenLayers(spec);
    const S = vec.obsSize;
    const O = this.head.outSize;
    const seed = init && init.kind === 'policy' ? MLP.fromJSON(init.net) : null;
    this.nets = [];
    this.flats = [];
    for (let k = 0; k < this.P; k++) {
      const m = new MLP(S, hidden, O, 'linear', this.rng, 1);
      if (seed && m.sameShape(seed.toJSON())) {
        m.copyFrom(seed);
        if (k > 0) {
          const f = m.getFlat();
          for (let i = 0; i < f.length; i++) f[i] += this.sigma * this.rng.gauss();
          m.setFlat(f);
        }
      }
      this.nets.push(m);
      this.flats.push(m.getFlat());
    }
    this.best = this.nets[0].clone();
    this.sums = new Float64Array(this.P);
    this.evals = new Int32Array(this.P);
    this.assign = new Int32Array(vec.n).fill(-1);
    this.actBuf = new Float32Array(vec.n * vec.actDims);
    this.noop = new Float32Array(vec.actDims);
    for (let d = 0; d < vec.actDims; d++) {
      const ax = spec.axes[d];
      if (space.kind === 'discrete') this.noop[d] = Math.max(0, ax.options.indexOf(0));
    }
    vec.onEpisode = (r, i) => this.onEpisode(r.reward, i, r);
    vec.reset();
    this.startGeneration();
  }

  private startGeneration(): void {
    this.sums.fill(0);
    this.evals.fill(0);
    this.nextIdx = 0;
    this.finished = 0;
    this.assign.fill(-1);
    for (let i = 0; i < this.vec.n; i++) {
      if (this.nextIdx < this.P) {
        this.assign[i] = this.nextIdx++;
        this.vec.obs.set(this.vec.envs[i].reset(), i * this.vec.obsSize);
      }
    }
  }

  private onEpisode(reward: number, envIndex: number, rec: { step: number; reward: number; length: number; success: boolean }): void {
    const k = this.assign[envIndex];
    if (k < 0) return;
    this.stats.addEpisode(rec);
    this.sums[k] += reward;
    this.evals[k]++;
    if (this.evals[k] >= Math.max(1, this.spec.train.episodesPerEval)) {
      this.finished++;
      this.assign[envIndex] = this.nextIdx < this.P ? this.nextIdx++ : -1;
    }
  }

  work(): void {
    const { vec } = this;
    const S = vec.obsSize;
    const D = vec.actDims;
    const O = this.head.outSize;
    for (let i = 0; i < vec.n; i++) {
      const k = this.assign[i];
      if (k < 0) {
        this.actBuf.set(this.noop, i * D);
        continue;
      }
      const out = this.nets[k].forward(vec.obs.subarray(i * S, i * S + S), 1);
      this.head.sample(out, 0, null, this.rng, this.actBuf, i * D, true);
    }
    vec.step(this.actBuf);
    this.phase = `generation ${this.generation + 1}: evaluated ${this.finished}/${this.P} brains`;
    if (this.finished >= this.P) this.nextGeneration();
  }

  private nextGeneration(): void {
    const tr = this.spec.train;
    const fit = Array.from({ length: this.P }, (_, k) => this.sums[k] / Math.max(1, this.evals[k]));
    this.lastFitness = fit.slice();
    const order = fit.map((_, k) => k).sort((a, b) => fit[b] - fit[a]);
    const elites = Math.max(1, Math.round(tr.eliteFrac * this.P));
    const mean = fit.reduce((a, b) => a + b, 0) / this.P;
    const bestF = fit[order[0]];
    this.best = this.nets[order[0]].clone();
    this.bestFitness = bestF;
    const newFlats: Float32Array[] = [];
    for (let r = 0; r < elites; r++) newFlats.push(this.flats[order[r]].slice());
    while (newFlats.length < this.P) {
      // tournament: pick two elites, take the fitter one as parent
      const a = order[this.rng.int(elites)];
      const b = order[this.rng.int(elites)];
      const parent = this.flats[fit[a] >= fit[b] ? a : b];
      const child = parent.slice();
      for (let i = 0; i < child.length; i++) child[i] += this.sigma * this.rng.gauss();
      newFlats.push(child);
    }
    this.flats = newFlats;
    this.nets.forEach((m, k) => m.setFlat(this.flats[k]));
    this.generation++;
    this.updates++;
    this.last = {
      generation: this.generation,
      bestFitness: bestF,
      meanFitness: mean,
      worstFitness: fit[order[order.length - 1]],
      medianFitness: fit[order[Math.floor(order.length / 2)]],
      mutationSigma: this.sigma,
      population: this.P,
      elites,
      meanReward: this.stats.meanRecent(),
    };
    this.stats.addMetrics({ step: this.vec.totalSteps, update: this.updates, ...this.last });
    this.sigma = Math.max(0.005, this.sigma * tr.mutationDecay);
    this.startGeneration();
  }

  explain(): Record<string, number | string> {
    const tr = this.spec.train;
    return {
      algo: 'Neuroevolution',
      population: this.P,
      elites: Math.max(1, Math.round(tr.eliteFrac * this.P)),
      sigma: this.sigma,
      generation: this.generation,
      worlds: this.vec.n,
      params: this.flats[0].length,
      ...this.last,
    };
  }

  snapshot(): BrainSnapshot {
    return {
      format: 'neuroblocks-brain',
      version: 1,
      algo: 'evolution',
      kind: 'policy',
      net: this.best.toJSON(),
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
