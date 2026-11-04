/// <reference lib="webworker" />
import { LLMHost } from './host.ts';
import type { FromLLM, ToLLM } from './protocol.ts';

const ctx = self as unknown as DedicatedWorkerGlobalScope;
const host = new LLMHost((m: FromLLM, transfer?: Transferable[]) => ctx.postMessage(m, transfer ?? []), 40);
let timer: ReturnType<typeof setTimeout> | null = null;

function loop(): void {
  timer = null;
  const delay = host.tick();
  if (delay >= 0) schedule(delay);
}
function schedule(delay = 0): void {
  if (timer !== null) return;
  timer = setTimeout(loop, delay);
}

ctx.onmessage = (e: MessageEvent<ToLLM>) => {
  host.handle(e.data);
  if (host.busy) schedule();
};
