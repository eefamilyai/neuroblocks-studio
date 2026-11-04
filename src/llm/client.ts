/**
 * LLMClient: talks to the mini-LLM training worker (or runs the same host on
 * the main thread when workers are unavailable).
 */
import LLMWorker from '@llm-worker';
import { LLMHost } from './host.ts';
import type { FromLLM, ToLLM } from './protocol.ts';

export class LLMClient {
  private worker: Worker | null = null;
  private local: LLMHost | null = null;
  private alive = false;
  private pending: ToLLM[] = [];
  private raf = 0;
  private timer = 0;
  onMessage: (m: FromLLM) => void = () => {};
  mode: 'worker' | 'main-thread' = 'worker';

  constructor() {
    try {
      const w = new LLMWorker();
      this.worker = w;
      w.onmessage = (e: MessageEvent<FromLLM>) => {
        this.alive = true;
        this.pending = [];
        this.onMessage(e.data);
      };
      w.onerror = (e) => {
        if (!this.alive) {
          e.preventDefault?.();
          this.fallBack();
        } else this.onMessage({ type: 'error', msg: `Worker error: ${e.message}` });
      };
    } catch {
      this.fallBack();
    }
  }

  private fallBack(): void {
    if (this.local) return;
    this.worker?.terminate();
    this.worker = null;
    this.mode = 'main-thread';
    this.local = new LLMHost((m) => this.onMessage(m), 16);
    const replay = this.pending;
    this.pending = [];
    for (const m of replay) this.send(m);
  }

  send(m: ToLLM, transfer: Transferable[] = []): void {
    if (this.worker) {
      if (!this.alive) this.pending.push(m);
      this.worker.postMessage(m, this.alive ? transfer : []);
      return;
    }
    const host = this.local;
    if (!host) return;
    host.handle(m);
    if (host.busy) this.schedule(0);
  }

  private schedule(delay: number): void {
    const host = this.local!;
    cancelAnimationFrame(this.raf);
    clearTimeout(this.timer);
    const run = () => {
      const d = host.tick();
      if (d >= 0) this.schedule(d);
    };
    if (delay > 0) this.timer = window.setTimeout(run, delay);
    else this.raf = requestAnimationFrame(run);
  }

  terminate(): void {
    this.worker?.terminate();
    this.worker = null;
    this.local = null;
  }
}
