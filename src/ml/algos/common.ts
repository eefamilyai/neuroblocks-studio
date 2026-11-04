/**
 * Shared pieces for every learning algorithm: a vectorised environment
 * (many robots training in parallel), statistics logging, and the
 * "brain snapshot" format used for showing, saving and exporting.
 */
import { Env } from '../../sim/env.ts';
import type { ScriptFactory } from '../../sim/env.ts';
import type { WorldRuntime } from '../../sim/runtime.ts';
import type { AgentSpec, AlgoKind } from '../../sim/types.ts';
import { MLP } from '../nn.ts';
import type { MLPJSON } from '../nn.ts';
import { RunningMeanStd } from '../normalizer.ts';
import { PolicyHead, decodeJoint, softmax } from '../policy.ts';
import type { ActionSpace } from '../policy.ts';
import { RNG } from '../rng.ts';

export interface EpisodeRecord {
  step: number;
  reward: number;
  length: number;
  success: boolean;
}

export interface MetricPoint {
  step: number;
  [key: string]: number;
}

/** What the learner has learned so far: everything needed to run the brain. */
export interface BrainSnapshot {
  format: 'neuroblocks-brain';
  version: 1;
  algo: AlgoKind | 'bc';
  /** 'policy' = outputs logits/means; 'q' = outputs a Q-value per joint action */
  kind: 'policy' | 'q';
  net: MLPJSON;
  critic: MLPJSON | null;
  logStd: number[] | null;
  obsNorm: { mean: number[]; var: number[]; count: number; clip: number } | null;
  space: ActionSpace;
  obsLabels: string[];
  steps: number;
  updates: number;
}

export class VecEnv {
  readonly envs: Env[];
  readonly n: number;
  readonly obsSize: number;
  readonly actDims: number;
  readonly obs: Float32Array;
  readonly termObs: Float32Array;
  readonly rewards: Float32Array;
  readonly dones: Uint8Array;
  readonly truncs: Uint8Array;
  readonly successes: Uint8Array;
  totalSteps = 0;
  onEpisode: ((r: EpisodeRecord, envIndex: number) => void) | null = null;

  constructor(rt: WorldRuntime, spec: AgentSpec, factory: ScriptFactory | null, n: number, seed: number, onError?: (m: string) => void, onLog?: (m: string) => void) {
    this.n = n;
    this.envs = [];
    for (let i = 0; i < n; i++) {
      const e = new Env(rt, spec, factory, seed * 1000 + i * 7919 + 1);
      if (onError) e.onError = onError;
      if (onLog && i === 0) e.onLog = onLog;
      this.envs.push(e);
    }
    this.obsSize = this.envs[0].obs.length;
    this.actDims = spec.axes.length;
    this.obs = new Float32Array(n * this.obsSize);
    this.termObs = new Float32Array(n * this.obsSize);
    this.rewards = new Float32Array(n);
    this.dones = new Uint8Array(n);
    this.truncs = new Uint8Array(n);
    this.successes = new Uint8Array(n);
  }

  reset(): Float32Array {
    for (let i = 0; i < this.n; i++) this.obs.set(this.envs[i].reset(), i * this.obsSize);
    return this.obs;
  }

  /** Step every env; finished envs are reset automatically (their last obs goes to termObs). */
  step(actions: Float32Array, actOff = 0): void {
    const S = this.obsSize;
    for (let i = 0; i < this.n; i++) {
      const e = this.envs[i];
      e.step(actions, actOff + i * this.actDims);
      this.rewards[i] = e.stepReward;
      this.dones[i] = e.done ? 1 : 0;
      this.truncs[i] = e.truncated ? 1 : 0;
      this.successes[i] = e.success ? 1 : 0;
      if (e.done) {
        this.termObs.set(e.obs, i * S);
        this.totalSteps++;
        if (this.onEpisode) this.onEpisode({ step: this.totalSteps, reward: e.epReturn, length: e.steps, success: e.success }, i);
        this.obs.set(e.reset(), i * S);
      } else {
        this.totalSteps++;
        this.obs.set(e.obs, i * S);
      }
    }
  }

  /** Positions of all robots: [x, y, z, yaw] per env (for the "ghost" view). */
  ghosts(out?: Float32Array): Float32Array {
    const g = out ?? new Float32Array(this.n * 4);
    for (let i = 0; i < this.n; i++) {
      const a = this.envs[i].a;
      g[i * 4] = a.x;
      g[i * 4 + 1] = a.y;
      g[i * 4 + 2] = a.z;
      g[i * 4 + 3] = a.yaw;
    }
    return g;
  }
}

/** Accumulates episode results & training metrics; the UI drains them periodically. */
export class StatsLog {
  episodes: EpisodeRecord[] = [];
  metrics: MetricPoint[] = [];
  private epCursor = 0;
  private mCursor = 0;
  episodeCount = 0;
  successCount = 0;
  recentRewards: number[] = [];
  bestMean = -Infinity;

  addEpisode(r: EpisodeRecord): void {
    this.episodes.push(r);
    this.episodeCount++;
    if (r.success) this.successCount++;
    this.recentRewards.push(r.reward);
    if (this.recentRewards.length > 100) this.recentRewards.shift();
    // keep memory bounded
    if (this.episodes.length > 20000) {
      this.episodes.splice(0, 10000);
      this.epCursor = Math.max(0, this.epCursor - 10000);
    }
  }

  addMetrics(m: MetricPoint): void {
    this.metrics.push(m);
    if (this.metrics.length > 20000) {
      this.metrics.splice(0, 10000);
      this.mCursor = Math.max(0, this.mCursor - 10000);
    }
  }

  meanRecent(): number {
    const r = this.recentRewards;
    if (!r.length) return 0;
    return r.reduce((a, b) => a + b, 0) / r.length;
  }

  drain(): { episodes: EpisodeRecord[]; metrics: MetricPoint[] } {
    const out = { episodes: this.episodes.slice(this.epCursor), metrics: this.metrics.slice(this.mCursor) };
    this.epCursor = this.episodes.length;
    this.mCursor = this.metrics.length;
    return out;
  }
}

export interface Algorithm {
  readonly kind: AlgoKind | 'bc';
  /** human-readable current activity, e.g. "collecting experience 54/128" */
  phase: string;
  /** a small, bounded unit of work */
  work(): void;
  snapshot(): BrainSnapshot;
  /** extra live info for the "Math" panel */
  explain(): Record<string, number | string>;
  stats: StatsLog;
  updates: number;
  /** optional: algorithms with a natural end (behaviour cloning) */
  finished?(): boolean;
}

/**
 * Brain: a runnable copy of a snapshot. Used by the 3D showcase robot on the
 * main thread and by the exporters.
 */
export class Brain {
  readonly snap: BrainSnapshot;
  readonly net: MLP;
  readonly critic: MLP | null;
  readonly head: PolicyHead;
  readonly norm: RunningMeanStd | null;
  readonly logStd: Float32Array | null;
  private nbuf: Float32Array;
  lastOut: Float32Array;
  lastValue = 0;
  lastProbs: number[][] = [];
  lastQ: Float32Array | null = null;

  constructor(snap: BrainSnapshot) {
    this.snap = snap;
    this.net = MLP.fromJSON(snap.net);
    this.critic = snap.critic ? MLP.fromJSON(snap.critic) : null;
    this.head = new PolicyHead(snap.space);
    this.norm = snap.obsNorm ? RunningMeanStd.fromJSON(snap.obsNorm) : null;
    this.logStd = snap.logStd ? Float32Array.from(snap.logStd) : null;
    this.nbuf = new Float32Array(snap.net.inSize);
    this.lastOut = new Float32Array(this.net.outSize);
  }

  normalize(obs: Float32Array): Float32Array {
    if (this.norm) this.norm.normalize(obs, this.nbuf, 1);
    else this.nbuf.set(obs.subarray(0, this.nbuf.length));
    return this.nbuf;
  }

  /** Choose an action for one observation. Fills `act` (one value per action axis). */
  act(obs: Float32Array, act: Float32Array, rng: RNG, deterministic: boolean): void {
    // evolved brains were selected while acting deterministically
    if (this.snap.algo === 'evolution') deterministic = true;
    const x = this.normalize(obs);
    const out = this.net.forward(x, 1);
    this.lastOut.set(out.subarray(0, this.net.outSize));
    if (this.critic) this.lastValue = this.critic.forward(x, 1)[0];
    if (this.snap.kind === 'q') {
      const q = this.lastOut;
      this.lastQ = q;
      let best = 0;
      for (let i = 1; i < q.length; i++) if (q[i] > q[best]) best = i;
      this.lastValue = q[best];
      // show a softmax over Q as "preferences" per joint action
      const p = new Array<number>(q.length);
      const scaled = Array.from(q, (v) => v * 5);
      softmax(scaled, 0, q.length, p, 0);
      this.lastProbs = [p];
      if (!deterministic && rng.next() < 0.02) best = rng.int(q.length);
      decodeJoint(this.snap.space, best, act, 0);
      return;
    }
    this.lastProbs = this.head.probs(this.lastOut, 0);
    this.head.sample(this.lastOut, 0, this.logStd, rng, act, 0, deterministic);
  }
}

export function makeHiddenLayers(spec: AgentSpec) {
  return spec.brain.layers.length ? spec.brain.layers : [{ units: 32, activation: 'tanh' as const }];
}

export function shuffleIndices(n: number, rng: RNG, out?: Int32Array): Int32Array {
  const a = out && out.length === n ? out : new Int32Array(n);
  for (let i = 0; i < n; i++) a[i] = i;
  rng.shuffle(a);
  return a;
}
