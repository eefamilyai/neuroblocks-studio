/**
 * Mini language models: gradient checks for the whole transformer, KV-cache
 * decoding, tokenizers, datasets, and "does it actually learn?".
 */
import { GPT, countParams } from '../src/llm/gpt.ts';
import type { GPTConfig } from '../src/llm/gpt.ts';
import { Decoder, sampleToken } from '../src/llm/decode.ts';
import { Tokenizer } from '../src/llm/tokenizer.ts';
import { DATASETS, getDataset } from '../src/llm/datasets.ts';
import { DEFAULT_TRAIN, LLMTrainer } from '../src/llm/train.ts';
import { RNG } from '../src/ml/rng.ts';
import { test, assert } from './harness.ts';

const tiny = (over: Partial<GPTConfig> = {}): GPTConfig => ({ vocab: 7, ctx: 5, d: 8, layers: 2, heads: 2, mlp: 2, act: 'gelu', pos: 'learned', tie: false, ...over });

for (const over of [{}, { act: 'relu' as const }, { act: 'swish' as const, tie: true }, { pos: 'sinusoidal' as const }, { pos: 'none' as const, heads: 4, layers: 1 }]) {
  test(`transformer gradients match finite differences ${JSON.stringify(over)}`, () => {
    const cfg = tiny(over);
    const model = new GPT(cfg, 3);
    // bigger random weights so every part of the network matters (and the finite-difference
    // step of 0.01 is small compared with every weight's scale, embeddings included)
    const rng = new RNG(9);
    for (const p of model.params) if (!p.name.includes('ln') && (p.name.endsWith('weight') || p.name === 'wte' || p.name === 'wpe')) for (let i = 0; i < p.w.length; i++) p.w[i] = rng.gauss() * 0.4;
    const B = 2;
    const T = cfg.ctx;
    const x = Array.from({ length: B * T }, () => rng.int(cfg.vocab));
    const y = Array.from({ length: B * T }, () => rng.int(cfg.vocab));
    const lossAt = () => {
      model.forward(x, B, T);
      return model.loss(y);
    };
    model.forward(x, B, T);
    model.lossAndGrad(y);
    model.zeroGrad();
    model.backward();
    const h = 1e-2;
    let worst = 0;
    let where = '';
    let checked = 0;
    let kinks = 0;
    const numeric = (p: { w: Float32Array }, i: number, hh: number) => {
      const old = p.w[i];
      p.w[i] = old + hh;
      const lp = lossAt();
      p.w[i] = old - hh;
      const lm = lossAt();
      p.w[i] = old;
      return (lp - lm) / (2 * hh);
    };
    for (const p of model.params) {
      const step = Math.max(1, Math.floor(p.w.length / 6));
      for (let i = 0; i < p.w.length; i += step) {
        const num = numeric(p, i, h);
        // ReLU has a kink at 0: if a smaller step disagrees, the difference quotient is not a slope here
        const num2 = numeric(p, i, h / 4);
        if (Math.abs(num - num2) > 0.05 * Math.max(Math.abs(num), Math.abs(num2)) + 2e-4) {
          kinks++;
          continue;
        }
        const ana = p.g[i];
        const err = Math.abs(num - ana) / Math.max(2e-3, Math.abs(num) + Math.abs(ana));
        checked++;
        if (err > worst) {
          worst = err;
          where = `${p.name}[${i}] numeric ${num.toExponential(3)} vs backprop ${ana.toExponential(3)}`;
        }
      }
    }
    console.log(`      ${checked} weights checked${kinks ? ` (${kinks} skipped at ReLU kinks)` : ''}, worst relative error ${worst.toExponential(2)} (${where})`);
    assert(kinks <= checked * 0.1, `too many non-smooth points: ${kinks}`);
    // ReLU's kinks make finite differences only approximate; the smooth activations (GELU, swish)
    // run through exactly the same code otherwise and are held to a tight tolerance
    assert(worst < (cfg.act === 'relu' ? 0.12 : 0.02), `gradient mismatch: ${where}`);
  });
}

test('KV-cache decoding gives the same next-token scores as the full forward pass', () => {
  const cfg: GPTConfig = { vocab: 11, ctx: 12, d: 16, layers: 2, heads: 4, mlp: 4, act: 'gelu', pos: 'learned', tie: true };
  const model = new GPT(cfg, 5);
  const rng = new RNG(4);
  for (const p of model.params) if (p.name.endsWith('weight') && !p.name.includes('ln')) for (let i = 0; i < p.w.length; i++) p.w[i] = rng.gauss() * 0.3;
  const seq = Array.from({ length: cfg.ctx }, () => rng.int(cfg.vocab));
  const full = Float32Array.from(model.forward(seq, 1, cfg.ctx));
  const dec = new Decoder(model);
  let worst = 0;
  for (let t = 0; t < cfg.ctx; t++) {
    const l = dec.step(seq[t]);
    for (let i = 0; i < cfg.vocab; i++) worst = Math.max(worst, Math.abs(l[i] - full[t * cfg.vocab + i]));
  }
  assert(worst < 1e-4, `decoder differs by ${worst}`);
  // past the window: it keeps working (drops the oldest half)
  for (let k = 0; k < 20; k++) dec.step(sampleToken(dec.logits, rng, 1, 0, 1));
  assert(dec.pos <= cfg.ctx && dec.tokens.length === dec.pos, 'window handling');
});

test('tokenizers turn every dataset into tokens and back without losing text', () => {
  for (const ds of DATASETS) {
    const text = ds.make(1).text.slice(0, 20000);
    const sizes: string[] = [];
    for (const kind of ['char', 'word', 'bpe'] as const) {
      const tok = Tokenizer.train(kind, text, 200);
      const ids = tok.encode(text);
      assert(tok.decode(ids) === text, `${ds.id}/${kind}: round trip failed`);
      const again = new Tokenizer(tok.toJSON());
      assert(again.decode(again.encode(text)) === text && again.encode(text).length === ids.length, `${ds.id}/${kind}: JSON round trip`);
      sizes.push(`${kind} ${tok.size} tokens → ${(text.length / ids.length).toFixed(2)} chars/token`);
    }
    console.log(`      ${ds.id}: ${sizes.join(' · ')}`);
  }
  const bpe = Tokenizer.train('bpe', getDataset('stories')!.make(1).text, 300);
  assert(bpe.size === 300 && bpe.merges.length === 300 - new Set(getDataset('stories')!.make(1).text).size, `bpe vocab ${bpe.size}`);
  assert(bpe.pieces(' the cat').length <= 3, `BPE should merge common words: ${JSON.stringify(bpe.pieces(' the cat'))}`);
});

test('a tiny transformer learns to invent names (loss falls well below chance)', () => {
  const corpus = getDataset('names')!.make(1);
  const tok = Tokenizer.train('char', corpus.text);
  const ids = tok.encode(corpus.text);
  const split = Math.floor(ids.length * 0.9);
  const model = new GPT({ vocab: tok.size, ctx: 16, d: 32, layers: 2, heads: 4, mlp: 4, act: 'gelu', pos: 'learned', tie: true }, 1);
  const tr = new LLMTrainer(model, ids.subarray(0, split), ids.subarray(split), { ...DEFAULT_TRAIN, batch: 16, steps: 400, warmup: 30, lr: 5e-3 });
  const chance = Math.log(tok.size);
  const before = tr.evaluate();
  const t0 = performance.now();
  while (tr.step < 400) tr.trainStep();
  const after = tr.evaluate();
  const secs = (performance.now() - t0) / 1000;
  const dec = new Decoder(model);
  const rng = new RNG(1);
  let out = '';
  let t = tok.encode('\n')[0];
  for (let i = 0; i < 60; i++) {
    t = sampleToken(dec.step(t), rng, 0.8);
    out += tok.vocab[t];
  }
  console.log(`      ${countParams(model.cfg)} weights · val loss ${before.toFixed(3)} → ${after.toFixed(3)} (chance ${chance.toFixed(3)}) · ${((400 * 16 * 16) / secs).toFixed(0)} tokens/s · sample: ${JSON.stringify(out)}`);
  assert(Math.abs(before - chance) < 0.3, 'an untrained model should be near chance');
  assert(after < 2.3, `val loss only reached ${after}`);
  assert(countParams(model.cfg) === model.numParams, 'countParams');
});
