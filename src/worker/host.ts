/**
 * The training loop host. The same code runs inside a Web Worker (normal
 * case) or on the main thread in small slices (fallback).
 */
import { Trainer } from '../ml/algos/trainer.ts';
import { Evolution } from '../ml/algos/evolution.ts';
import { buildWorld } from '../sim/worldgen.ts';
import type { FromWorker, ToWorker } from './protocol.ts';

export class TrainingHost {
  private trainer: Trainer | null = null;
  private running = false;
  private ghosts = true;
  private lastProgress = 0;
  private lastBrain = 0;
  private lastGhost = 0;
  private lastUpdates = -1;
  private stepsAtLast = 0;
  private timeAtLast = 0;
  private sps = 0;
  private spsWindow: [number, number][] = [];
  private errorsSent = new Set<string>();
  private post: (m: FromWorker, transfer?: Transferable[]) => void;
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
            world: buildWorld(),
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
        this.stepsAtLast = 0;
        this.timeAtLast = performance.now();
        this.post({ type: 'started', obsSize: this.trainer.vec.obsSize, algo: this.trainer.algo.kind });
        this.post({ type: 'brain', snap: this.trainer.snapshot() });
        break;
      }
      case 'pause':
        this.running = false;
        this.sendProgress(true);
        break;
      case 'resume':
        if (this.trainer) this.running = true;
        this.spsWindow = [];
        this.timeAtLast = performance.now();
        this.stepsAtLast = this.trainer ? this.trainer.totalSteps : 0;
        break;
      case 'stop':
        this.running = false;
        if (this.trainer) {
          this.sendProgress(true);
          this.post({ type: 'brain', snap: this.trainer.snapshot() });
        }
        this.trainer = null;
        break;
      case 'ghosts':
        this.ghosts = msg.on;
        break;
    }
  }

  /** Run one slice of training work. Returns true if there is more to do. */
  tick(): boolean {
    const tr = this.trainer;
    if (!tr || !this.running) return false;
    try {
      tr.runFor(this.sliceMs);
    } catch (e) {
      this.running = false;
      this.post({ type: 'error', msg: `Training crashed: ${(e as Error).message}` });
      return false;
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
    if (tr.algo.finished?.()) {
      this.running = false;
      this.sendProgress(true);
      this.post({ type: 'brain', snap: tr.snapshot() });
      this.post({ type: 'finished', reason: `the copy is done (${tr.algo.updates} updates). Press ▶ Train to polish it with reinforcement learning` });
      return false;
    }
    const stop = tr.spec.train.stopSteps;
    if (stop > 0 && tr.totalSteps >= stop) {
      this.running = false;
      this.sendProgress(true);
      this.post({ type: 'brain', snap: tr.snapshot() });
      this.post({ type: 'finished', reason: `reached the step limit (${stop.toLocaleString()} steps)` });
      return false;
    }
    return true;
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
    this.stepsAtLast = tr.totalSteps;
    this.timeAtLast = now;
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
    });
  }
}
