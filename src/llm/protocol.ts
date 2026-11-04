import type { GPTConfig } from './gpt.ts';
import type { TokenizerJSON, TokenizerKind } from './tokenizer.ts';
import type { LLMTrainConfig } from './train.ts';

/** Everything that describes a mini-LLM project (saved to files / autosave). */
export interface LLMProject {
  format: 'neuroblocks-llm';
  version: 1;
  name: string;
  data: {
    /** a built-in dataset id, or 'custom' for your own text */
    dataset: string;
    seed: number;
    /** your own text (dataset = 'custom') */
    text?: string;
  };
  tokenizer: { kind: TokenizerKind; vocabSize: number };
  model: Omit<GPTConfig, 'vocab'>;
  train: LLMTrainConfig;
  /** live samples while training */
  sample: { prompt: string; temperature: number; every: number };
}

export interface LLMStats {
  chars: number;
  tokens: number;
  trainTokens: number;
  valTokens: number;
  vocab: number;
  params: number;
}

export type ToLLM =
  | {
      type: 'start';
      project: LLMProject;
      text: string;
      valStart?: number;
      /** continue from a trained model (its tokenizer is reused) */
      init?: { tokenizer: TokenizerJSON; weights: Float32Array[]; step: number; tokens: number } | null;
      stepOnly?: boolean;
    }
  | { type: 'pause' }
  | { type: 'resume' }
  | { type: 'step' }
  | { type: 'stop' }
  | { type: 'threads'; n: number };

export type FromLLM =
  | { type: 'ready'; tokenizer: TokenizerJSON; stats: LLMStats; step: number }
  | {
      type: 'progress';
      step: number;
      tokens: number;
      /** (step, loss) pairs since the last message */
      losses: [number, number][];
      vals: [number, number][];
      lr: number;
      gradNorm: number;
      tokPerSec: number;
      running: boolean;
      threads: number;
    }
  | { type: 'sample'; step: number; text: string }
  | { type: 'weights'; step: number; tokens: number; weights: Float32Array[] }
  | { type: 'log'; level: 'info' | 'warn' | 'err' | 'good'; msg: string }
  | { type: 'error'; msg: string }
  | { type: 'paused'; reason: string }
  | { type: 'finished'; reason: string };
