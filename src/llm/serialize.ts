/**
 * Saving and loading mini language models: weights as base64 float32 per
 * named tensor, plus the tokenizer and the project settings.
 */
import { GPT } from './gpt.ts';
import type { LLMProject } from './protocol.ts';
import { Tokenizer } from './tokenizer.ts';
import type { TokenizerJSON } from './tokenizer.ts';

export function f32ToB64(a: Float32Array): string {
  const bytes = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
  let s = '';
  const CH = 0x8000;
  for (let i = 0; i < bytes.length; i += CH) s += String.fromCharCode(...bytes.subarray(i, i + CH));
  return btoa(s);
}

export function b64ToF32(s: string): Float32Array {
  const bin = atob(s);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer, 0, bytes.length >> 2);
}

export interface SavedModel {
  tokenizer: TokenizerJSON;
  shapes: Record<string, number[]>;
  weights: Record<string, string>;
  step: number;
  tokens: number;
}

/** A training curve, thinned out for saving. */
export interface SavedHistory {
  loss: [number, number][];
  val: [number, number][];
  /** held-out task accuracy in % (datasets with right answers) */
  acc?: [number, number][];
}

export interface LLMFile extends LLMProject {
  savedAt?: string;
  /** the trained model (weights + tokenizer), if there is one */
  trained?: SavedModel;
  history?: SavedHistory;
  samples?: { step: number; text: string }[];
}

export function saveModel(model: GPT, tok: Tokenizer, step: number, tokens: number): SavedModel {
  const shapes: Record<string, number[]> = {};
  const weights: Record<string, string> = {};
  for (const p of model.params) {
    shapes[p.name] = p.shape;
    weights[p.name] = f32ToB64(p.w);
  }
  return { tokenizer: tok.toJSON(), shapes, weights, step, tokens };
}

/** Rebuild a saved model. Throws a readable error if it does not fit the settings. */
export function loadModel(s: SavedModel, cfg: LLMProject['model']): { model: GPT; tok: Tokenizer } {
  if (!s.tokenizer || !Array.isArray(s.tokenizer.vocab)) throw new Error('the file has no tokenizer');
  const tok = new Tokenizer(s.tokenizer);
  const model = new GPT({ ...cfg, vocab: tok.size });
  for (const p of model.params) {
    const b = s.weights?.[p.name];
    if (typeof b !== 'string') throw new Error(`the weights for "${p.name}" are missing`);
    const w = b64ToF32(b);
    if (w.length !== p.w.length) throw new Error(`"${p.name}" has ${w.length} weights, the model needs ${p.w.length}`);
    p.w.set(w);
  }
  return { model, tok };
}

/** Check that something read from a file is a mini-LLM project (fills in missing parts). */
export function validateLLMFile(j: unknown, defaults: LLMProject): LLMFile {
  const o = j as Partial<LLMFile> | null;
  if (!o || typeof o !== 'object' || o.format !== 'neuroblocks-llm') throw new Error('this is not a NeuroBlocks language-model file');
  const d = defaults;
  const f: LLMFile = {
    ...d,
    ...o,
    format: 'neuroblocks-llm',
    version: 1,
    name: typeof o.name === 'string' && o.name ? o.name.slice(0, 80) : d.name,
    data: { ...d.data, ...(o.data ?? {}) },
    tokenizer: { ...d.tokenizer, ...(o.tokenizer ?? {}) },
    model: { ...d.model, ...(o.model ?? {}) },
    train: { ...d.train, ...(o.train ?? {}) },
    sample: { ...d.sample, ...(o.sample ?? {}) },
  };
  if (!['char', 'word', 'bpe'].includes(f.tokenizer.kind)) throw new Error(`unknown tokenizer "${f.tokenizer.kind}"`);
  if (f.data.dataset === 'custom' && typeof f.data.text !== 'string') f.data.text = '';
  return f;
}

/** Keep at most n points of a curve (bucket means), for saving. */
export function thin(xs: ArrayLike<number>, ys: ArrayLike<number>, n: number): [number, number][] {
  const len = Math.min(xs.length, ys.length);
  const out: [number, number][] = [];
  if (len <= n) {
    for (let i = 0; i < len; i++) out.push([xs[i], ys[i]]);
    return out;
  }
  const per = len / n;
  for (let k = 0; k < n; k++) {
    const a = Math.floor(k * per);
    const b = Math.max(a + 1, Math.floor((k + 1) * per));
    let s = 0;
    let c = 0;
    for (let i = a; i < b; i++)
      if (Number.isFinite(ys[i])) {
        s += ys[i];
        c++;
      }
    if (c) out.push([xs[b - 1], s / c]);
  }
  return out;
}
