/** The default mini-LLM project and ready-made presets. */
import type { LLMProject } from '../protocol.ts';
import { DEFAULT_TRAIN } from '../train.ts';

export function defaultProject(): LLMProject {
  return {
    format: 'neuroblocks-llm',
    version: 1,
    name: 'Story writer',
    data: { dataset: 'stories', seed: 1 },
    tokenizer: { kind: 'char', vocabSize: 256 },
    model: { ctx: 64, d: 64, layers: 2, heads: 4, mlp: 4, act: 'gelu', pos: 'learned', tie: true },
    train: { ...DEFAULT_TRAIN, batch: 8, steps: 1500, lr: 3e-3, warmup: 100, evalEvery: 100 },
    sample: { prompt: 'Once upon a time', temperature: 0.8, every: 100 },
  };
}

export interface Preset {
  id: string;
  emoji: string;
  title: string;
  blurb: string;
  time: string;
  make(): LLMProject;
}

const base = defaultProject;

export const PRESETS: Preset[] = [
  {
    id: 'stories',
    emoji: '📖',
    title: 'Story writer',
    blurb: 'Characters → words → sentences. A 2-layer transformer learns to write tiny stories.',
    time: '≈ 2–5 min',
    make: () => base(),
  },
  {
    id: 'keyboard',
    emoji: '⌨️',
    title: 'Phone keyboard',
    blurb: 'Word prediction and autocorrect, like on a phone. It learns from everyday text messages; then type in the Keyboard tab: it suggests your next word and fixes typos.',
    time: '≈ 1–3 min',
    make: () => ({
      ...base(),
      name: 'Phone keyboard',
      data: { dataset: 'messages', seed: 1 },
      tokenizer: { kind: 'word', vocabSize: 1024 },
      model: { ctx: 24, d: 64, layers: 2, heads: 4, mlp: 4, act: 'gelu', pos: 'learned', tie: true },
      train: { ...base().train, batch: 16, steps: 1500, lr: 3e-3, warmup: 100, evalEvery: 100 },
      sample: { prompt: 'See you', temperature: 0.8, every: 100 },
    }),
  },
  {
    id: 'names',
    emoji: '🏷️',
    title: 'Name generator',
    blurb: 'Invent new names that sound real: small, fast, a classic first language model. Then try 3,000 steps: the training loss keeps falling while the validation loss (names it never saw) goes UP. It is memorising the list.',
    time: '≈ 30 s',
    make: () => ({
      ...base(),
      name: 'Name generator',
      data: { dataset: 'names', seed: 1 },
      model: { ctx: 16, d: 32, layers: 2, heads: 4, mlp: 4, act: 'gelu', pos: 'learned', tie: true },
      train: { ...base().train, batch: 16, steps: 500, lr: 5e-3, warmup: 50, evalEvery: 50 },
      sample: { prompt: '', temperature: 0.8, every: 50 },
    }),
  },
  {
    id: 'adder',
    emoji: '➕',
    title: 'Calculator',
    blurb: 'Can a transformer learn to ADD? It trains on sums like 07+58=065 and is tested on 50 sums it has never seen: watch the held-out accuracy climb.',
    time: '≈ 2–6 min',
    make: () => ({
      ...base(),
      name: 'Calculator',
      data: { dataset: 'arithmetic', seed: 1 },
      model: { ctx: 30, d: 64, layers: 2, heads: 4, mlp: 4, act: 'gelu', pos: 'learned', tie: true },
      train: { ...base().train, batch: 16, steps: 2000, lr: 1e-3, warmup: 100, evalEvery: 100 },
      sample: { prompt: '25+17=', temperature: 0, every: 100 },
    }),
  },
  {
    id: 'copy',
    emoji: '🔁',
    title: 'Copy machine',
    blurb: 'Copy and reverse 5-letter strings it has never seen (hello<olleh). Then open the Attention tab: a head learns to point at exactly the letter to write next.',
    time: '≈ 2–4 min',
    make: () => ({
      ...base(),
      name: 'Copy machine',
      data: { dataset: 'copy', seed: 1 },
      model: { ctx: 24, d: 64, layers: 2, heads: 4, mlp: 4, act: 'gelu', pos: 'learned', tie: true },
      train: { ...base().train, batch: 16, steps: 1500, lr: 1e-3, warmup: 100, evalEvery: 100 },
      sample: { prompt: 'hello<', temperature: 0, every: 100 },
    }),
  },
  {
    id: 'python',
    emoji: '🐍',
    title: 'Code helper',
    blurb: 'A miniature code model trained on tiny Python programs: it learns colons, indentation and return.',
    time: '≈ 2–5 min',
    make: () => ({
      ...base(),
      name: 'Code helper',
      data: { dataset: 'python', seed: 1 },
      sample: { prompt: 'def ', temperature: 0.6, every: 100 },
    }),
  },
  {
    id: 'bpe',
    emoji: '🧩',
    title: 'Story writer with BPE',
    blurb: 'The same stories with GPT-style byte-pair tokens: fewer, longer tokens per sentence, a bigger vocabulary.',
    time: '≈ 2–5 min',
    make: () => ({
      ...base(),
      name: 'Story writer (BPE)',
      tokenizer: { kind: 'bpe', vocabSize: 384 },
      model: { ctx: 48, d: 64, layers: 2, heads: 4, mlp: 4, act: 'gelu', pos: 'learned', tie: true },
    }),
  },
];
