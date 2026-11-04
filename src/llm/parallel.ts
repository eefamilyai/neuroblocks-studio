/**
 * Data-parallel training on several CPU cores.
 *
 * Every step, the batch of text windows is split between the coordinator
 * (the training worker) and a few helper workers. Each computes the
 * gradient for its share with the same weights; the coordinator adds the
 * gradients up (weighted by share size, which gives exactly the gradient of
 * the whole batch) and updates the weights. Like multi-GPU training, on
 * CPU cores.
 */
import type { GPT, GPTConfig } from './gpt.ts';

/** coordinator → helper */
export type ToHelper =
  | { type: 'init'; cfg: GPTConfig }
  | { type: 'work'; id: number; weights: Float32Array; x: Int32Array; y: Int32Array; B: number; T: number }
  | { type: 'close' };

/** helper → coordinator */
export type FromHelper = { type: 'done'; id: number; loss: number; grads: Float32Array } | { type: 'error'; id: number; msg: string };

export function numWeights(model: GPT): number {
  let n = 0;
  for (const p of model.params) n += p.w.length;
  return n;
}

/** All weights in one flat array (params order). */
export function packWeights(model: GPT, out: Float32Array = new Float32Array(numWeights(model))): Float32Array {
  let o = 0;
  for (const p of model.params) {
    out.set(p.w, o);
    o += p.w.length;
  }
  return out;
}

export function unpackWeights(model: GPT, flat: Float32Array): void {
  let o = 0;
  for (const p of model.params) {
    p.w.set(flat.subarray(o, o + p.w.length));
    o += p.w.length;
  }
  if (o !== flat.length) throw new Error('weights do not fit the model');
}

export function packGrads(model: GPT): Float32Array {
  const out = new Float32Array(numWeights(model));
  let o = 0;
  for (const p of model.params) {
    out.set(p.g, o);
    o += p.g.length;
  }
  return out;
}

/** g += s · flat */
export function addGrads(model: GPT, flat: Float32Array, s: number): void {
  let o = 0;
  for (const p of model.params) {
    const g = p.g;
    for (let i = 0; i < g.length; i++) g[i] += s * flat[o + i];
    o += g.length;
  }
}

export function scaleGrads(model: GPT, s: number): void {
  for (const p of model.params) {
    const g = p.g;
    for (let i = 0; i < g.length; i++) g[i] *= s;
  }
}

/** Split B sequences into k nearly equal shares (largest first). */
export function splitBatch(B: number, k: number): number[] {
  const n = Math.max(1, Math.min(B, k));
  return Array.from({ length: n }, (_, i) => Math.floor(B / n) + (i < B % n ? 1 : 0));
}

/** The helper side: compute gradients for whatever the coordinator sends. */
export function serveHelper(port: MessagePort, makeModel: (cfg: GPTConfig) => GPT, onClose: () => void): void {
  let model: GPT | null = null;
  port.onmessage = (e: MessageEvent<ToHelper>) => {
    const m = e.data;
    if (m.type === 'init') {
      model = makeModel(m.cfg);
      return;
    }
    if (m.type === 'close') {
      port.close();
      onClose();
      return;
    }
    try {
      if (!model) throw new Error('the helper has no model');
      unpackWeights(model, m.weights);
      model.forward(m.x, m.B, m.T);
      const loss = model.lossAndGrad(m.y);
      model.zeroGrad();
      model.backward();
      const grads = packGrads(model);
      port.postMessage({ type: 'done', id: m.id, loss, grads } satisfies FromHelper, [grads.buffer]);
    } catch (err) {
      port.postMessage({ type: 'error', id: m.id, msg: (err as Error).message } satisfies FromHelper);
    }
  };
}

/** The coordinator side: a set of helpers to send work to. */
export class HelperPool {
  private ports: MessagePort[];
  private pending = new Map<number, { resolve: (r: { loss: number; grads: Float32Array }) => void; reject: (e: Error) => void; timer: ReturnType<typeof setTimeout> }>();
  private nextId = 1;
  closed = false;

  constructor(ports: MessagePort[]) {
    this.ports = ports;
    for (const p of ports) p.onmessage = (e: MessageEvent<FromHelper>) => this.onMsg(e.data);
  }

  get size(): number {
    return this.ports.length;
  }

  init(cfg: GPTConfig): void {
    for (const p of this.ports) p.postMessage({ type: 'init', cfg: { ...cfg } } satisfies ToHelper);
  }

  work(k: number, weights: Float32Array, x: Int32Array, y: Int32Array, B: number, T: number): Promise<{ loss: number; grads: Float32Array }> {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      if (this.closed) return reject(new Error('closed'));
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error('a helper thread did not answer'));
      }, 20000);
      this.pending.set(id, { resolve, reject, timer });
      this.ports[k].postMessage({ type: 'work', id, weights, x, y, B, T } satisfies ToHelper, [x.buffer, y.buffer]);
    });
  }

  private onMsg(m: FromHelper): void {
    const p = this.pending.get(m.id);
    if (!p) return;
    this.pending.delete(m.id);
    clearTimeout(p.timer);
    if (m.type === 'error') p.reject(new Error(m.msg));
    else p.resolve({ loss: m.loss, grads: m.grads });
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    for (const p of this.ports) {
      try {
        p.postMessage({ type: 'close' } satisfies ToHelper);
        p.close();
      } catch {
        /* already gone */
      }
    }
    for (const [, q] of this.pending) {
      clearTimeout(q.timer);
      q.reject(new Error('closed'));
    }
    this.pending.clear();
  }
}
