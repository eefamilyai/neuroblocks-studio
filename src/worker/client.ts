/**
 * TrainerClient: talks to the training worker. Falls back to running the
 * very same TrainingHost on the main thread if workers are unavailable.
 */
import TrainerWorker from './trainer.worker.ts?worker&inline';
import { TrainingHost } from './host.ts';
import type { FromWorker, ToWorker } from './protocol.ts';

export class TrainerClient {
  private worker: Worker | null = null;
  private local: TrainingHost | null = null;
  private localRaf = 0;
  onMessage: (m: FromWorker) => void = () => {};
  readonly mode: 'worker' | 'main-thread';

  constructor() {
    let w: Worker | null = null;
    try {
      w = new TrainerWorker();
    } catch (e) {
      console.warn('Web Worker unavailable, training on the main thread', e);
    }
    if (w) {
      this.worker = w;
      this.mode = 'worker';
      w.onmessage = (e: MessageEvent<FromWorker>) => this.onMessage(e.data);
      w.onerror = (e) => this.onMessage({ type: 'error', msg: `Worker error: ${e.message}` });
    } else {
      this.mode = 'main-thread';
      this.local = new TrainingHost((m) => this.onMessage(m), 10);
    }
  }

  send(m: ToWorker): void {
    if (this.worker) {
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
