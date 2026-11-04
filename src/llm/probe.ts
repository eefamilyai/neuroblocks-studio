/**
 * Accuracy on a task with right answers: give the model the start of a line
 * it never trained on ("37+58=") and check whether its most likely
 * continuation is exactly right ("95", then a new line).
 */
import { Decoder } from './decode.ts';
import type { LineMode } from './decode.ts';
import type { GPT } from './gpt.ts';
import type { Tokenizer } from './tokenizer.ts';
import { suggestWords } from './keyboard.ts';

export interface ProbeCase {
  prompt: string;
  answer: string;
}

/** Pick n cases spread evenly over the list (always the same ones). */
export function pickCases(all: ProbeCase[], n = 50): ProbeCase[] {
  if (all.length <= n) return all;
  return Array.from({ length: n }, (_, i) => all[Math.floor(((i + 0.5) * all.length) / n)]);
}

/** Greedy completion of one prompt, up to the end of the line (or maxTokens). */
export function complete(dec: Decoder, tok: Tokenizer, prompt: string, maxTokens: number): string {
  dec.reset();
  const ids = tok.encode(prompt);
  let logits = dec.feed(ids.length ? ids : [0]);
  let out = '';
  for (let i = 0; i < maxTokens; i++) {
    let best = 0;
    for (let k = 1; k < logits.length; k++) if (logits[k] > logits[best]) best = k;
    out += tok.vocab[best] ?? '';
    if (out.includes('\n')) break;
    logits = dec.step(best);
  }
  return out;
}

/**
 * Fraction of cases answered right (NaN without cases). 'exact': the greedy
 * completion is exactly the answer; 'next-word': the answer is one of the 3
 * suggested next words.
 */
export function probeAccuracy(model: GPT, tok: Tokenizer, cases: ProbeCase[], kind: 'exact' | 'next-word' = 'exact', align?: LineMode): number {
  if (!cases.length) return NaN;
  if (kind === 'next-word') {
    let hit = 0;
    for (const c of cases) if (suggestWords({ model, tok, align }, c.prompt, 3).some((s) => s.word === c.answer)) hit++;
    return hit / cases.length;
  }
  const dec = new Decoder(model);
  let ok = 0;
  for (const c of cases) {
    const out = complete(dec, tok, c.prompt, c.answer.length + 2);
    const nl = out.indexOf('\n');
    if (nl >= 0 && out.slice(0, nl) === c.answer) ok++;
  }
  return ok / cases.length;
}
