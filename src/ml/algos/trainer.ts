/**
 * Trainer: wires a spec + world into a vectorised env and a learning
 * algorithm, and runs it in small time slices (so the UI/worker stays
 * responsive and can pause at any moment).
 */
import { compileScript } from '../../sim/env.ts';
import type { ScriptFactory } from '../../sim/env.ts';
import { WorldRuntime, mergeEdits } from '../../sim/runtime.ts';
import type { WorldEdits } from '../../sim/runtime.ts';
import { actionSpaceOf } from '../../sim/spec.ts';
import type { AgentSpec, WorldDef } from '../../sim/types.ts';
import { VecEnv } from './common.ts';
import type { Algorithm, BrainSnapshot } from './common.ts';
import { PPO } from './ppo.ts';
import { DQN } from './dqn.ts';
import { Reinforce } from './reinforce.ts';
import { Evolution } from './evolution.ts';
import { BehaviorCloning } from './bc.ts';
import { CustomAlgo } from './custom.ts';
import type { DemoSet } from './bc.ts';

export interface TrainerOptions {
  spec: AgentSpec;
  world: WorldDef;
  edits?: WorldEdits;
  init?: BrainSnapshot | null;
  demos?: DemoSet | null;
  /** use behaviour cloning on the demos instead of the spec's RL algorithm */
  mode?: 'rl' | 'bc';
  onError?: (msg: string) => void;
  onLog?: (msg: string) => void;
}

export class Trainer {
  /** the trainer's own copy of the spec (schedules change lr/entropy in it as training runs) */
  readonly spec: AgentSpec;
  readonly rt: WorldRuntime;
  readonly vec: VecEnv;
  readonly algo: Algorithm;
  readonly factory: ScriptFactory;
  private ghostBuf: Float32Array;
  wallTime = 0;
  private baseLr: number;
  private baseEntropy: number;

  constructor(o: TrainerOptions) {
    this.spec = structuredClone(o.spec);
    this.baseLr = this.spec.train.lr;
    this.baseEntropy = this.spec.train.entropy;
    o = { ...o, spec: this.spec };
    this.rt = new WorldRuntime(o.world, mergeEdits(o.edits, this.spec));
    this.factory = compileScript(o.spec.script);
    const tr = o.spec.train;
    const n = Math.max(1, Math.min(64, Math.round(o.mode === 'bc' ? 8 : tr.algo === 'evolution' ? Math.min(tr.population, tr.numEnvs) : tr.numEnvs)));
    this.vec = new VecEnv(this.rt, o.spec, this.factory, n, tr.seed, o.onError, o.onLog);
    const space = actionSpaceOf(o.spec);
    if (o.mode === 'bc') {
      if (!o.demos || !o.demos.count) throw new Error('No demonstrations recorded yet. Drive the robot with the keyboard in 🎮 Drive mode and press ⏺ Record first.');
      this.algo = new BehaviorCloning(this.vec, o.spec, space, o.demos, o.init ?? null);
    } else {
      switch (tr.algo) {
        case 'dqn':
          if (space.kind !== 'discrete') throw new Error('DQN only works with discrete actions. Switch the actions block to "discrete" or pick PPO.');
          this.algo = new DQN(this.vec, o.spec, space, o.init ?? null);
          break;
        case 'reinforce':
          this.algo = new Reinforce(this.vec, o.spec, space, o.init ?? null);
          break;
        case 'evolution':
          this.algo = new Evolution(this.vec, o.spec, space, o.init ?? null);
          break;
        case 'custom':
          this.algo = new CustomAlgo(this.vec, o.spec, space, this.factory, o.init ?? null);
          break;
        default:
          this.algo = new PPO(this.vec, o.spec, space, o.init ?? null);
      }
    }
    this.ghostBuf = new Float32Array(n * 4);
  }

  /** Do work for roughly `ms` milliseconds (or until `untilUpdate` more updates happened). */
  runFor(ms: number, untilUpdate = 0): void {
    const t0 = performance.now();
    let guard = 0;
    this.applySchedules();
    const target = this.algo.updates + untilUpdate;
    while (performance.now() - t0 < ms && guard++ < 100000) {
      this.algo.work();
      if (untilUpdate && this.algo.updates >= target) break;
    }
    this.wallTime += performance.now() - t0;
  }

  /** How far through the schedule we are (0 → 1). */
  scheduleProgress(): number {
    const tr = this.spec.train;
    const horizon = tr.scheduleSteps > 0 ? tr.scheduleSteps : tr.stopSteps > 0 ? tr.stopSteps : 1e6;
    return Math.min(1, this.totalSteps / horizon);
  }

  /**
   * Learning-rate schedule: lr(p) = lr₀ · (end + (1 − end) · shape(p)),
   * shape = 1 − p (linear) or ½(1 + cos πp) (cosine). The entropy bonus can
   * move linearly from its start value to `entropyEnd`.
   */
  private applySchedules(): void {
    const tr = this.spec.train;
    const p = this.scheduleProgress();
    if (tr.lrSchedule === 'linear' || tr.lrSchedule === 'cosine') {
      const shape = tr.lrSchedule === 'cosine' ? 0.5 * (1 + Math.cos(Math.PI * p)) : 1 - p;
      const end = Math.max(0, Math.min(1, Number.isFinite(tr.lrEnd) ? tr.lrEnd : 0));
      tr.lr = this.baseLr * (end + (1 - end) * shape);
      this.algo.setLearningRate?.(tr.lr);
    }
    if (Number.isFinite(tr.entropyEnd)) tr.entropy = Math.max(0, this.baseEntropy + (tr.entropyEnd - this.baseEntropy) * p);
  }

  get totalSteps(): number {
    return this.vec.totalSteps;
  }

  ghosts(): Float32Array {
    return this.vec.ghosts(this.ghostBuf);
  }

  snapshot(): BrainSnapshot {
    return this.algo.snapshot();
  }
}
