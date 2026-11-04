/// <reference lib="webworker" />
/**
 * The mini-LLM training worker. The first message decides its job: the
 * coordinator (runs training, LLMHost) or a helper that computes gradients
 * for part of every batch (multi-threaded training).
 */
import { GPT } from './gpt.ts';
import { LLMHost } from './host.ts';
import { serveHelper } from './parallel.ts';
import type { FromLLM, ToLLM } from './protocol.ts';

const ctx = self as unknown as DedicatedWorkerGlobalScope;
let host: LLMHost | null = null;
let timer: ReturnType<typeof setTimeout> | null = null;

function loop(): void {
  timer = null;
  const delay = host!.tick();
  if (delay >= 0) schedule(delay);
}
function schedule(delay = 0): void {
  if (timer !== null) return;
  timer = setTimeout(loop, delay);
}

ctx.onmessage = (e: MessageEvent<ToLLM | { type: 'helper'; port: MessagePort }>) => {
  const m = e.data;
  if (m.type === 'helper') {
    ctx.onmessage = null;
    serveHelper(m.port, (cfg) => new GPT(cfg), () => ctx.close());
    return;
  }
  if (!host) {
    host = new LLMHost((msg: FromLLM, transfer?: Transferable[]) => ctx.postMessage(msg, transfer ?? []), 40);
    host.onWake = () => schedule(0);
  }
  host.handle(m);
  if (host.busy) schedule();
};
