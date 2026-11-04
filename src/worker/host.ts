/**
 * The training loop host. The same code runs inside a Web Worker (normal
 * case) or on the main thread in small slices (fallback).
 */
import { Trainer } from '../ml/algos/trainer.ts';
import { Evolution } from '../ml/algos/evolution.ts';
import { Evaluator } from '../ml/algos/evaluate.ts';
import { buildWorld } from '../sim/worldgen.ts';
import type { WorldDef } from '../sim/types.ts';
import type { FromWorker, ToWorker } from './protocol.ts';

let worldCache: WorldDef | null = null;
const world = (): WorldDef => (worldCache ??= buildWorld());

export class TrainingHost {
  private trainer: Trainer | null = null;
  private evaluator: Evaluator | null = null;
  private running = false;
  private ghosts = true;
  private lastProgress = 0;
  private lastBrain = 0;
  private lastGhost = 0;
  private lastUpdates = -1;
  private sps = 0;
  private spsWindow: [number, number][] = [];
  private errorsSent = new Set<string>();
  private post: (m: FromWorker, transfer?: Transferable[]) => void;
  /** pause again once `algo.updates` reaches this (single-step mode) */
  private stepTarget = -1;
  /** max environment steps per second (0 = unlimited) */
  private throttle = 0;
  private throttleT0 = 0;
  private throttleS0 = 0;
  private trainStart = 0;
  private trainedMs = 0;
  /** ms of work per slice */
  sliceMs: number;

  constructor(post: (m: FromWorker, transfer?: Transferable[]) => void, sliceMs = 30) {
    this.post = post;
    this.sliceMs = sliceMs;
  }

  handle(msg: ToWorker): void {
    switch (msg.type) {
      case 'start': {
        try {
          this.trainer = new Trainer({
            spec: msg.spec,
            world: world(),
            edits: msg.edits,
            init: msg.init,
            demos: msg.demos,
            mode: msg.mode,
            onError: (m) => {
              if (this.errorsSent.has(m)) return;
              this.errorsSent.add(m);
              this.post({ type: 'log', level: 'err', msg: m });
            },
            onLog: (m) => this.post({ type: 'log', level: 'info', msg: `📝 ${m}` }),
          });
        } catch (e) {
          this.post({ type: 'error', msg: (e as Error).message });
          return;
        }
        this.running = true;
        this.spsWindow = [];
        this.lastUpdates = -1;
        this.trainedMs = 0;
        this.trainStart = performance.now();
        this.stepTarget = msg.stepOnly ? this.trainer.algo.updates + 1 : -1;
        this.resetThrottle();
        this.post({ type: 'started', obsSize: this.trainer.vec.obsSize, algo: this.trainer.algo.kind });
        this.post({ type: 'brain', snap: this.trainer.snapshot() });
        break;
      }
      case 'pause':
        this.running = false;
        this.stepTarget = -1;
        this.sendProgress(true);
        break;
      case 'resume':
        if (this.trainer) this.running = true;
        this.stepTarget = -1;
        this.spsWindow = [];
        this.trainStart = performance.now();
        this.resetThrottle();
        break;
      case 'step':
        if (!this.trainer) break;
        this.stepTarget = this.trainer.algo.updates + 1;
        this.running = true;
        this.trainStart = performance.now();
        this.resetThrottle();
        break;
      case 'stop':
        this.running = false;
        this.stepTarget = -1;
        if (this.trainer) {
          this.sendProgress(true);
          this.post({ type: 'brain', snap: this.trainer.snapshot() });
        }
        this.trainer = null;
        this.evaluator = null;
        break;
      case 'ghosts':
        this.ghosts = msg.on;
        break;
      case 'throttle':
        this.throttle = Math.max(0, msg.sps || 0);
        this.resetThrottle();
        break;
      case 'evaluate':
        try {
          this.evaluator = new Evaluator(msg.spec, world(), msg.edits, msg.snap, msg.episodes, msg.deterministic);
        } catch (e) {
          this.post({ type: 'error', msg: `Evaluation failed: ${(e as Error).message}` });
        }
        break;
    }
  }

  /** Is there anything left to do? (used by the worker loop) */
  get busy(): boolean {
    return !!this.evaluator || (!!this.trainer && this.running);
  }

  private resetThrottle(): void {
    this.throttleT0 = performance.now();
    this.throttleS0 = this.trainer ? this.trainer.totalSteps : 0;
  }

  /**
   * Run one slice of work. Returns the delay in ms before the next slice,
   * or -1 when there is nothing to do.
   */
  tick(): number {
    if (this.evaluator) return this.tickEval();
    const tr = this.trainer;
    if (!tr || !this.running) return -1;
    let ms = this.sliceMs;
    if (this.throttle > 0) {
      // only do as many steps as the speed limit allows so far
      const allowed = this.throttleS0 + ((performance.now() - this.throttleT0) / 1000) * this.throttle;
      if (tr.totalSteps >= allowed) return 15;
      ms = Math.min(ms, 8);
    }
    try {
      const t0 = performance.now();
      tr.runFor(ms, this.stepTarget >= 0 ? 1 : 0);
      this.trainedMs += performance.now() - t0;
    } catch (e) {
      this.running = false;
      this.post({ type: 'error', msg: `Training crashed: ${(e as Error).message}` });
      return -1;
    }
    const now = performance.now();
    if (now - this.lastProgress > 250) this.sendProgress(false);
    if (tr.algo.updates !== this.lastUpdates && now - this.lastBrain > 400) {
      this.lastUpdates = tr.algo.updates;
      this.lastBrain = now;
      this.post({ type: 'brain', snap: tr.snapshot() });
    }
    if (this.ghosts && now - this.lastGhost > 90) {
      this.lastGhost = now;
      const g = tr.ghosts().slice();
      this.post({ type: 'ghosts', data: g }, [g.buffer]);
    }
    if (this.stepTarget >= 0 && tr.algo.updates >= this.stepTarget) {
      this.stepTarget = -1;
      this.running = false;
      this.finishSlice();
      this.post({ type: 'paused', reason: `one update done (update #${tr.algo.updates})` });
      return -1;
    }
    const reason = this.stopReason(tr);
    if (reason) {
      this.running = false;
      this.finishSlice();
      this.post({ type: 'finished', reason });
      return -1;
    }
    return 0;
  }

  /** Send the final numbers and brain of a run that just stopped. */
  private finishSlice(): void {
    const tr = this.trainer!;
    this.sendProgress(true);
    this.lastUpdates = tr.algo.updates;
    this.post({ type: 'brain', snap: tr.snapshot() });
  }

  private stopReason(tr: Trainer): string | null {
    if (tr.algo.finished?.()) return `the copy is done (${tr.algo.updates} updates). Press ▶ Train to polish it with reinforcement learning`;
    const t = tr.spec.train;
    if (t.stopSteps > 0 && tr.totalSteps >= t.stopSteps) return `reached the step limit (${t.stopSteps.toLocaleString()} steps)`;
    const st = tr.algo.stats;
    if (st.recentSuccess.length >= 100) {
      if (t.targetSuccess > 0 && st.successRecent() >= t.targetSuccess) return `success rate reached ${(st.successRecent() * 100).toFixed(0)}% (goal ${(t.targetSuccess * 100).toFixed(0)}%)`;
      if (Number.isFinite(t.targetReward) && st.meanRecent() >= t.targetReward) return `mean reward reached ${st.meanRecent().toFixed(2)} (goal ${t.targetReward})`;
    }
    if (t.stopMinutes > 0 && this.trainedMs / 60000 >= t.stopMinutes) return `time limit reached (${t.stopMinutes} min)`;
    return null;
  }

  private tickEval(): number {
    const ev = this.evaluator!;
    try {
      const done = ev.work(this.sliceMs);
      this.post({ type: 'evalProgress', done: Math.round(ev.progress() * ev.target), total: ev.target });
      if (done) {
        this.evaluator = null;
        this.post({ type: 'evaluated', result: ev.result() });
        return this.trainer && this.running ? 0 : -1;
      }
    } catch (e) {
      this.evaluator = null;
      this.post({ type: 'error', msg: `Evaluation crashed: ${(e as Error).message}` });
      return -1;
    }
    return 0;
  }

  private sendProgress(force: boolean): void {
    const tr = this.trainer;
    if (!tr) return;
    const now = performance.now();
    if (!force && now - this.lastProgress < 250) return;
    // steps per second over a 4 s window (smooths over PPO's collect/learn phases)
    this.spsWindow.push([now, tr.totalSteps]);
    while (this.spsWindow.length > 2 && now - this.spsWindow[0][0] > 4000) this.spsWindow.shift();
    const [t0, s0] = this.spsWindow[0];
    if (now - t0 > 400) this.sps = ((tr.totalSteps - s0) * 1000) / (now - t0);
    if (!this.running) this.sps = 0;
    this.lastProgress = now;
    const d = tr.algo.stats.drain();
    this.post({
      type: 'progress',
      steps: tr.totalSteps,
      episodes: d.episodes,
      metrics: d.metrics,
      phase: this.running ? tr.algo.phase : 'paused',
      sps: this.sps,
      updates: tr.algo.updates,
      explain: tr.algo.explain(),
      episodeCount: tr.algo.stats.episodeCount,
      wallTime: tr.wallTime,
      fitness: tr.algo instanceof Evolution ? tr.algo.lastFitness : undefined,
      lr: tr.spec.train.lr,
      entropyCoef: tr.spec.train.entropy,
      schedule: tr.scheduleProgress(),
    });
  }
}
