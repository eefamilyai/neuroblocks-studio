/**
 * LLMClient: talks to the mini-LLM training worker (or runs the same host on
 * the main thread when workers are unavailable).
 */
import LLMWorker from '@llm-worker';
import { LLMHost } from './host.ts';
import type { FromLLM, ToLLM } from './protocol.ts';

/** A sensible default: leave one core for the page, use at most 4. */
export function defaultThreads(): number {
  const cores = typeof navigator !== 'undefined' ? navigator.hardwareConcurrency || 2 : 2;
  return Math.max(1, Math.min(4, cores - 1));
}

export class LLMClient {
  private worker: Worker | null = null;
  private helpers: Worker[] = [];
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
    for (const w of this.helpers) w.terminate();
    this.helpers = [];
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

  /**
   * Train on n threads: the training worker plus n − 1 helper workers, each
   * connected to it by a MessageChannel. (No helpers without workers.)
   */
  setThreads(n: number): number {
    for (const w of this.helpers) w.terminate();
    this.helpers = [];
    if (!this.worker) return 1;
    const ports: MessagePort[] = [];
    for (let i = 1; i < n; i++) {
      try {
        const w = new LLMWorker();
        const ch = new MessageChannel();
        w.postMessage({ type: 'helper', port: ch.port1 }, [ch.port1]);
        ports.push(ch.port2);
        this.helpers.push(w);
      } catch {
        break;
      }
    }
    // sent directly (ports must be transferred, and can't be replayed on the main-thread fallback)
    this.worker.postMessage({ type: 'helpers', ports } satisfies ToLLM, ports);
    return 1 + ports.length;
  }

  terminate(): void {
    for (const w of this.helpers) w.terminate();
    this.helpers = [];
    this.worker?.terminate();
    this.worker = null;
    this.local = null;
  }
}
