/**
 * Benchmark: train a mini language model headlessly and print loss + samples.
 *   node --experimental-strip-types tests/bench-llm.ts [dataset] [steps] [d] [layers] [ctx] [tokenizer] [batch]
 */
import { GPT, countParams } from '../src/llm/gpt.ts';
import { Decoder, sampleToken } from '../src/llm/decode.ts';
import { Tokenizer } from '../src/llm/tokenizer.ts';
import { getDataset } from '../src/llm/datasets.ts';
import { DEFAULT_TRAIN, LLMTrainer, lineStarts } from '../src/llm/train.ts';
import { RNG } from '../src/ml/rng.ts';
import { pickCases, probeAccuracy } from '../src/llm/probe.ts';

const [ds = 'stories', stepsS = '1000', dS = '64', lS = '2', ctxS = '64', kind = 'char', bS = '12'] = process.argv.slice(2);
const corpus = getDataset(ds)!.make(1);
const tok = Tokenizer.train(kind as 'char', corpus.text, 256);
const ids = tok.encode(corpus.text);
const split = corpus.valStart !== undefined ? tok.encode(corpus.text.slice(0, corpus.valStart)).length : Math.floor(ids.length * 0.9);
const model = new GPT({ vocab: tok.size, ctx: Number(ctxS), d: Number(dS), layers: Number(lS), heads: 4, mlp: 4, act: 'gelu', pos: 'learned', tie: true }, 1);
const steps = Number(stepsS);
const info = getDataset(ds)!;
const valText = corpus.text.slice(corpus.valStart ?? Math.floor(corpus.text.length * 0.9));
const cases = info.probe ? pickCases(info.probe.cases(valText), 50) : [];
const trainIds = ids.subarray(0, split);
const valIds = ids.subarray(split);
const starts = corpus.align ? { train: lineStarts(trainIds, tok.vocab, model.cfg.ctx), val: lineStarts(valIds, tok.vocab, model.cfg.ctx) } : null;
const tr = new LLMTrainer(model, trainIds, valIds, { ...DEFAULT_TRAIN, steps, batch: Number(bS), lr: Number(process.env.LR ?? DEFAULT_TRAIN.lr) }, starts);
console.log(`${ds} · ${kind} vocab ${tok.size} · ${ids.length} tokens · ${countParams(model.cfg)} weights`);
const t0 = performance.now();
const rng = new RNG(3);
const sample = (n: number) => {
  const dec = new Decoder(model);
  let t = tok.encode(corpus.text[0])[0] ?? 0;
  let out = tok.vocab[t];
  for (let i = 0; i < n; i++) {
    t = sampleToken(dec.step(t), rng, 0.8);
    out += tok.vocab[t];
  }
  return out;
};
while (tr.step < steps) {
  tr.trainStep();
  if (tr.step % Math.max(1, Math.floor(steps / 5)) === 0) {
    const secs = (performance.now() - t0) / 1000;
    const acc = cases.length ? ` · ${info.probe!.name} ${(probeAccuracy(model, tok, cases) * 100).toFixed(0)}%` : '';
    console.log(`step ${tr.step} loss ${tr.lastLoss.toFixed(3)} val ${tr.evaluate().toFixed(3)}${acc} · ${(tr.tokensSeen / secs).toFixed(0)} tok/s · ${secs.toFixed(0)} s\n  ${JSON.stringify(sample(120))}`);
  }
}
