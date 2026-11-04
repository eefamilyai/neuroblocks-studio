/**
 * TrainerClient: talks to the training worker. Falls back to running the
 * very same TrainingHost on the main thread if workers are unavailable,
 * either at construction time or when the worker fails to boot (e.g. some
 * browsers block module workers in pages opened from file://).
 */
import TrainerWorker from './trainer.worker.ts?worker&inline';
import { TrainingHost } from './host.ts';
import type { FromWorker, ToWorker } from './protocol.ts';

export class TrainerClient {
  private worker: Worker | null = null;
  private local: TrainingHost | null = null;
  private localRaf = 0;
  private workerAlive = false;
  private pending: ToWorker[] = [];
  onMessage: (m: FromWorker) => void = () => {};
  onFallback: (() => void) | null = null;
  mode: 'worker' | 'main-thread' = 'worker';

  constructor() {
    try {
      const w = new TrainerWorker();
      this.worker = w;
      w.onmessage = (e: MessageEvent<FromWorker>) => {
        this.workerAlive = true;
        this.pending = [];
        this.onMessage(e.data);
      };
      w.onerror = (e) => {
        if (!this.workerAlive) {
          e.preventDefault?.();
          this.fallBack();
        } else this.onMessage({ type: 'error', msg: `Worker error: ${e.message}` });
      };
    } catch (e) {
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
    const host = this.local!;
    host.handle(m);
    if (m.type === 'start' || m.type === 'resume') {
      cancelAnimationFrame(this.localRaf);
      const loop = () => {
        if (host.tick()) this.localRaf = requestAnimationFrame(loop);
      };
      this.localRaf = requestAnimationFrame(loop);
    }
  }
}
