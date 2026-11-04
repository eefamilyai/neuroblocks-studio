import type { BrainSnapshot, EpisodeRecord, MetricPoint } from '../ml/algos/common.ts';
import type { DemoSet } from '../ml/algos/bc.ts';
import type { WorldEdits } from '../sim/runtime.ts';
import type { AgentSpec } from '../sim/types.ts';

export type ToWorker =
  | { type: 'start'; spec: AgentSpec; edits: WorldEdits; init: BrainSnapshot | null; mode: 'rl' | 'bc'; demos: DemoSet | null }
  | { type: 'pause' }
  | { type: 'resume' }
  | { type: 'stop' }
  | { type: 'ghosts'; on: boolean };

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
}

export type FromWorker =
  | { type: 'started'; obsSize: number; algo: string }
  | Progress
  | { type: 'brain'; snap: BrainSnapshot }
  | { type: 'ghosts'; data: Float32Array }
  | { type: 'log'; level: 'info' | 'warn' | 'err' | 'good'; msg: string }
  | { type: 'error'; msg: string }
  | { type: 'finished'; reason: string };
