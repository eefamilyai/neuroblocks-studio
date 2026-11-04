/**
 * Evaluation: run a fixed brain (no learning) for a number of episodes and
 * measure how good it really is. Training curves mix in exploration noise
 * and an ever-changing policy; an evaluation answers "how well does THIS
 * brain do?", like a test set in supervised learning.
 */
import { compileScript } from '../../sim/env.ts';
import { WorldRuntime, mergeEdits } from '../../sim/runtime.ts';
import type { WorldEdits } from '../../sim/runtime.ts';
import type { AgentSpec, WorldDef } from '../../sim/types.ts';
import { RNG } from '../rng.ts';
import { Brain, VecEnv } from './common.ts';
import type { BrainSnapshot } from './common.ts';

export interface EvalResult {
  episodes: number;
  successRate: number;
  meanReward: number;
  stdReward: number;
  meanLength: number;
  rewards: number[];
  lengths: number[];
  successes: boolean[];
  /** mean reward per episode, per reward label */
  parts: Record<string, number>;
  deterministic: boolean;
  /** how much training the evaluated brain had */
  brainSteps: number;
  seconds: number;
}

export class Evaluator {
  private vec: VecEnv;
  private brains: Brain[];
  private act: Float32Array;
  private rng: RNG;
  private quota: number;
  private done: number[];
  private rewards: number[] = [];
  private lengths: number[] = [];
  private successes: boolean[] = [];
  private parts: Record<string, number> = {};
  private t0 = performance.now();
  private snap: BrainSnapshot;
  private deterministic: boolean;
  readonly target: number;

  constructor(spec: AgentSpec, world: WorldDef, edits: WorldEdits | undefined, snap: BrainSnapshot, episodes: number, deterministic: boolean, seed = 4242) {
    this.snap = snap;
    this.deterministic = deterministic;
    const rt = new WorldRuntime(world, mergeEdits(edits, spec));
    this.target = Math.max(1, Math.min(2000, Math.round(episodes)));
    const n = Math.min(16, this.target);
    // every world plays the same number of episodes, so short episodes are not over-counted
    this.quota = Math.ceil(this.target / n);
    this.vec = new VecEnv(rt, spec, compileScript(spec.script), n, seed);
    this.brains = Array.from({ length: n }, () => new Brain(snap));
    this.act = new Float32Array(n * Math.max(1, spec.axes.length));
    this.rng = new RNG(seed);
    this.done = new Array(n).fill(0);
    this.vec.onEpisode = (r, i) => {
      if (this.done[i] >= this.quota || this.rewards.length >= this.target) return;
      this.done[i]++;
      this.rewards.push(r.reward);
      this.lengths.push(r.length);
      this.successes.push(r.success);
      for (const [k, v] of Object.entries(r.parts ?? {})) this.parts[k] = (this.parts[k] ?? 0) + v;
    };
    this.vec.reset();
  }

  get finished(): boolean {
    return this.rewards.length >= this.target;
  }

  progress(): number {
    return this.rewards.length / this.target;
  }

  /** Step all worlds for about `ms` milliseconds. Returns true when finished. */
  work(ms: number): boolean {
    const t0 = performance.now();
    const { vec } = this;
    const S = vec.obsSize;
    const D = vec.actDims;
    const tmp = new Float32Array(Math.max(1, D));
    while (!this.finished && performance.now() - t0 < ms) {
      for (let i = 0; i < vec.n; i++) {
        this.brains[i].act(vec.obs.subarray(i * S, i * S + S), tmp, this.rng, this.deterministic);
        this.act.set(tmp.subarray(0, D), i * D);
      }
      vec.step(this.act);
    }
    return this.finished;
  }

  result(): EvalResult {
    const n = this.rewards.length || 1;
    const mean = this.rewards.reduce((a, b) => a + b, 0) / n;
    const sd = Math.sqrt(this.rewards.reduce((a, b) => a + (b - mean) ** 2, 0) / n);
    const parts: Record<string, number> = {};
    for (const [k, v] of Object.entries(this.parts)) parts[k] = v / n;
    return {
      episodes: this.rewards.length,
      successRate: this.successes.filter(Boolean).length / n,
      meanReward: mean,
      stdReward: sd,
      meanLength: this.lengths.reduce((a, b) => a + b, 0) / n,
      rewards: this.rewards.slice(),
      lengths: this.lengths.slice(),
      successes: this.successes.slice(),
      parts,
      deterministic: this.deterministic,
      brainSteps: this.snap.steps,
      seconds: (performance.now() - this.t0) / 1000,
    };
  }
}
