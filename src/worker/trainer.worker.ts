/// <reference lib="webworker" />
import { TrainingHost } from './host.ts';
import type { ToWorker, FromWorker } from './protocol.ts';

const ctx = self as unknown as DedicatedWorkerGlobalScope;
const host = new TrainingHost((m: FromWorker, transfer?: Transferable[]) => ctx.postMessage(m, transfer ?? []), 40);
let scheduled = false;

function loop(): void {
  scheduled = false;
  if (host.tick()) schedule();
}
function schedule(): void {
  if (scheduled) return;
  scheduled = true;
  setTimeout(loop, 0);
}

ctx.onmessage = (e: MessageEvent<ToWorker>) => {
  host.handle(e.data);
  if (e.data.type === 'start' || e.data.type === 'resume') schedule();
};
