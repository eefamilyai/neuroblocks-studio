/**
 * TrainerClient: talks to a training worker. Falls back to running the very
 * same TrainingHost on the main thread if workers are unavailable, either at
 * construction time or when the worker fails to boot (e.g. some browsers
 * block workers in pages opened from file://).
 */
import TrainerWorker from '@trainer-worker';
import { TrainingHost } from './host.ts';
import type { FromWorker, ToWorker } from './protocol.ts';

export class TrainerClient {
  private worker: Worker | null = null;
  private local: TrainingHost | null = null;
  private localTimer = 0;
  private localRaf = 0;
  private workerAlive = false;
  private pending: ToWorker[] = [];
  onMessage: (m: FromWorker) => void = () => {};
  onFallback: (() => void) | null = null;
  mode: 'worker' | 'main-thread' = 'worker';

  constructor(opts: { allowFallback?: boolean } = {}) {
    const allowFallback = opts.allowFallback ?? true;
    try {
      const w = new TrainerWorker();
      this.worker = w;
      w.onmessage = (e: MessageEvent<FromWorker>) => {
        this.workerAlive = true;
        this.pending = [];
        this.onMessage(e.data);
      };
      w.onerror = (e) => {
        if (!this.workerAlive && allowFallback) {
          e.preventDefault?.();
          this.fallBack();
        } else this.onMessage({ type: 'error', msg: `Worker error: ${e.message}` });
      };
    } catch (e) {
      if (!allowFallback) throw e;
      console.warn('Web Worker unavailable, training on the main thread', e);
      this.fallBack();
    }
  }

  private fallBack(): void {
    if (this.local) return;
    if (this.worker) this.worker.terminate();
    this.worker = null;
    this.mode = 'main-thread';
    this.local = new TrainingHost((m) => this.onMessage(m), 12);
    this.onFallback?.();
    const replay = this.pending;
    this.pending = [];
    for (const m of replay) this.send(m);
  }

  send(m: ToWorker): void {
    if (this.worker) {
      // remember messages until the worker proves it is alive, so we can replay them on fallback
      if (!this.workerAlive) this.pending.push(m);
      this.worker.postMessage(m);
      return;
    }
    const host = this.local;
    if (!host) return;
    host.handle(m);
    if (host.busy) this.scheduleLocal(0);
  }

  /** Main-thread fallback: run slices between frames. */
  private scheduleLocal(delay: number): void {
    const host = this.local!;
    cancelAnimationFrame(this.localRaf);
    clearTimeout(this.localTimer);
    const run = () => {
      const d = host.tick();
      if (d >= 0) this.scheduleLocal(d);
    };
    if (delay > 0) this.localTimer = window.setTimeout(run, delay);
    else this.localRaf = requestAnimationFrame(run);
  }

  terminate(): void {
    this.worker?.terminate();
    this.worker = null;
    cancelAnimationFrame(this.localRaf);
    clearTimeout(this.localTimer);
    this.local = null;
  }
}
