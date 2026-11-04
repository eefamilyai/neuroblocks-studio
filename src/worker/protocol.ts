import type { BrainSnapshot, EpisodeRecord, MetricPoint } from '../ml/algos/common.ts';
import type { DemoSet } from '../ml/algos/bc.ts';
import type { EvalResult } from '../ml/algos/evaluate.ts';
import type { WorldEdits } from '../sim/runtime.ts';
import type { AgentSpec } from '../sim/types.ts';

export type ToWorker =
  | {
      type: 'start';
      spec: AgentSpec;
      edits: WorldEdits;
      init: BrainSnapshot | null;
      mode: 'rl' | 'bc';
      demos: DemoSet | null;
      /** start paused after exactly one update (the "step" button) */
      stepOnly?: boolean;
    }
  | { type: 'pause' }
  | { type: 'resume' }
  | { type: 'stop' }
  /** run exactly one more update, then pause */
  | { type: 'step' }
  | { type: 'ghosts'; on: boolean }
  /** limit training speed: max environment steps per second (0 = as fast as possible) */
  | { type: 'throttle'; sps: number }
  /** test a brain without learning (runs as its own job) */
  | { type: 'evaluate'; spec: AgentSpec; edits: WorldEdits; snap: BrainSnapshot; episodes: number; deterministic: boolean };

export interface Progress {
  type: 'progress';
  steps: number;
  episodes: EpisodeRecord[];
  metrics: MetricPoint[];
  phase: string;
  sps: number;
  updates: number;
  explain: Record<string, number | string>;
  episodeCount: number;
  wallTime: number;
  fitness?: number[];
  /** current learning rate / entropy bonus (they change with schedules) */
  lr: number;
  entropyCoef: number;
  /** 0 → 1 through the schedule horizon */
  schedule: number;
}

export type FromWorker =
  | { type: 'started'; obsSize: number; algo: string }
  | Progress
  | { type: 'brain'; snap: BrainSnapshot }
  | { type: 'ghosts'; data: Float32Array }
  | { type: 'log'; level: 'info' | 'warn' | 'err' | 'good'; msg: string }
  | { type: 'error'; msg: string }
  | { type: 'finished'; reason: string }
  /** paused by itself (after a single step) */
  | { type: 'paused'; reason: string }
  | { type: 'evalProgress'; done: number; total: number }
  | { type: 'evaluated'; result: EvalResult };
