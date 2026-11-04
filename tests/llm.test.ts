/**
 * Mini language models: gradient checks for the whole transformer, KV-cache
 * decoding, tokenizers, datasets, and "does it actually learn?".
 */
import { GPT, countParams } from '../src/llm/gpt.ts';
import type { GPTConfig } from '../src/llm/gpt.ts';
import { Decoder, lineBreakTable, sampleToken, topK, windowStart, withStart } from '../src/llm/decode.ts';
import { Tokenizer } from '../src/llm/tokenizer.ts';
import { DATASETS, getDataset } from '../src/llm/datasets.ts';
import { DEFAULT_TRAIN, LLMTrainer, lineStarts } from '../src/llm/train.ts';
import { RNG } from '../src/ml/rng.ts';
import { addGrads, packGrads, packWeights, scaleGrads, splitBatch, unpackWeights } from '../src/llm/parallel.ts';
import { pickCases, probeAccuracy } from '../src/llm/probe.ts';
import { b64ToF32, f32ToB64, loadModel, saveModel, thin, validateLLMFile } from '../src/llm/serialize.ts';
import { correctWord, editDistance, knownWords, splitForSuggest, suggestWords } from '../src/llm/keyboard.ts';
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

test('splitting a batch over threads gives exactly the whole-batch gradient', () => {
  const cfg = tiny({ ctx: 6, tie: true });
  const a = new GPT(cfg, 5);
  const helper = new GPT(cfg, 99); // different init: it gets the weights from the coordinator
  unpackWeights(helper, packWeights(a));
  const rng = new RNG(4);
  const B = 5;
  const T = cfg.ctx;
  const x = Int32Array.from({ length: B * T }, () => rng.int(cfg.vocab));
  const y = Int32Array.from({ length: B * T }, () => rng.int(cfg.vocab));
  a.forward(x, B, T);
  const fullLoss = a.lossAndGrad(y);
  a.zeroGrad();
  a.backward();
  const full = packGrads(a);
  // coordinator share + two helper shares, combined the way the training worker does it
  const shares = splitBatch(B, 3);
  assert(shares.join(',') === '2,2,1', `shares ${shares}`);
  a.forward(x.subarray(0, shares[0] * T), shares[0], T);
  let loss = (a.lossAndGrad(y.subarray(0, shares[0] * T)) * shares[0]) / B;
  a.zeroGrad();
  a.backward();
  scaleGrads(a, shares[0] / B);
  let off = shares[0];
  for (const s of shares.slice(1)) {
    helper.forward(x.slice(off * T, (off + s) * T), s, T);
    loss += (helper.lossAndGrad(y.slice(off * T, (off + s) * T)) * s) / B;
    helper.zeroGrad();
    helper.backward();
    addGrads(a, packGrads(helper), s / B);
    off += s;
  }
  const split = packGrads(a);
  let maxDiff = 0;
  let maxG = 0;
  for (let i = 0; i < full.length; i++) {
    maxDiff = Math.max(maxDiff, Math.abs(full[i] - split[i]));
    maxG = Math.max(maxG, Math.abs(full[i]));
  }
  assert(Math.abs(loss - fullLoss) < 1e-5, `loss ${loss} vs ${fullLoss}`);
  assert(maxDiff < 1e-5 * Math.max(1, maxG), `gradients differ by ${maxDiff} (largest ${maxG})`);
});

test('saved models load back with the same weights and tokenizer', () => {
  const text = 'the cat sat on the mat. the dog sat on the log.\n'.repeat(20);
  const tok = Tokenizer.train('bpe', text, 40);
  const cfg = { ctx: 8, d: 12, layers: 1, heads: 3, mlp: 2, act: 'gelu' as const, pos: 'sinusoidal' as const, tie: false };
  const model = new GPT({ ...cfg, vocab: tok.size }, 7);
  const saved = JSON.parse(JSON.stringify(saveModel(model, tok, 123, 4567)));
  const { model: m2, tok: t2 } = loadModel(saved, cfg);
  assert(t2.decode(t2.encode(text)) === text, 'tokenizer round trip');
  model.params.forEach((p, i) => assert(p.w.every((v, k) => v === m2.params[i].w[k]), `weights of ${p.name}`));
  const a = Float32Array.from([1.5, -2.25, 3e-8, 1e9]);
  assert(b64ToF32(f32ToB64(a)).every((v, i) => v === a[i]), 'base64 round trip');
  let threw = false;
  try {
    loadModel(saved, { ...cfg, d: 16, heads: 4 });
  } catch {
    threw = true;
  }
  assert(threw, 'a model that does not fit the settings must not load');
  const f = validateLLMFile({ format: 'neuroblocks-llm', name: 'x', model: { d: 32 } }, { format: 'neuroblocks-llm', version: 1, name: 'd', data: { dataset: 'stories', seed: 1 }, tokenizer: { kind: 'char', vocabSize: 256 }, model: { ...cfg, ctx: 64, d: 64, heads: 4 }, train: { ...DEFAULT_TRAIN }, sample: { prompt: '', temperature: 1, every: 100 } });
  assert(f.model.d === 32 && f.model.heads === 4 && f.train.lr === DEFAULT_TRAIN.lr, 'missing settings are filled in');
  const th = thin([1, 2, 3, 4, 5, 6], [1, 1, 3, 3, 5, 5], 3);
  assert(JSON.stringify(th) === '[[2,1],[4,3],[6,5]]', `thin ${JSON.stringify(th)}`);
});

test('the arithmetic probe checks held-out sums exactly', () => {
  const ds = getDataset('arithmetic')!;
  const c = ds.make(1);
  const val = c.text.slice(c.valStart);
  const cases = pickCases(ds.probe!.cases(val), 50);
  assert(cases.length === 50, `cases ${cases.length}`);
  assert(c.align === 'line', 'one problem per line, each starting at position 0');
  for (const k of cases) {
    const [a, b] = k.prompt.slice(0, -1).split('+').map(Number);
    assert(a + b === Number(k.answer) && k.answer.length === 3, `${k.prompt} ${k.answer}`);
    assert(!('\n' + c.text.slice(0, c.valStart)).includes('\n' + k.prompt), `${k.prompt} appears in the training text`);
  }
  // an untrained model gets (almost) nothing right
  const tok = Tokenizer.train('char', c.text);
  const model = new GPT({ vocab: tok.size, ctx: 16, d: 16, layers: 1, heads: 2, mlp: 2, act: 'gelu', pos: 'learned', tie: true }, 1);
  assert(probeAccuracy(model, tok, cases) < 0.1, 'random model should fail');
});

test('line-aligned windows start right after line breaks', () => {
  const text = 'ab\ncd\nefg\nh\n';
  const tok = Tokenizer.train('char', text);
  const ids = tok.encode(text);
  const st = Array.from(lineStarts(ids, tok.vocab, 2, 'start'));
  // 'start': windows start at 0 and AT the line breaks 2, 5 and 9 (the break is the start token); 11 has no room
  assert(JSON.stringify(st) === '[0,2,5,9]', `starts ${JSON.stringify(st)}`);
  for (const i of st) assert(i === 0 || text[i] === '\n', `start ${i}`);
  // 'line': right AFTER the line breaks, so every line starts at position 0 (10 has no room)
  const sl = Array.from(lineStarts(ids, tok.vocab, 2, 'line'));
  assert(JSON.stringify(sl) === '[0,3,6]', `line starts ${JSON.stringify(sl)}`);
  // the trainer only samples aligned windows
  const model = new GPT({ vocab: tok.size, ctx: 2, d: 8, layers: 1, heads: 2, mlp: 2, act: 'gelu', pos: 'learned', tie: true }, 1);
  const tr = new LLMTrainer(model, ids, ids, { ...DEFAULT_TRAIN, batch: 8 }, { train: lineStarts(ids, tok.vocab, 2, 'start'), val: lineStarts(ids, tok.vocab, 2, 'start') });
  const { x } = tr.sampleBatch();
  for (let b = 0; b < 8; b++) assert(['a', '\n'].includes(tok.vocab[x[b * 2]]), `window ${b} starts with ${JSON.stringify(tok.vocab[x[b * 2]])}`);
});

test('generation windows slide to a line start for line-by-line data', () => {
  const vocab = ['\n', 'a', 'b'];
  const br = lineBreakTable(vocab);
  const ids = [1, 2, 0, 1, 1, 2, 0, 2, 1];
  // the newest 5 tokens would start mid-line (index 4); the earliest line break that fits is at 6…
  assert(windowStart(ids, 5, br, 'start') === 6, `got ${windowStart(ids, 5, br, 'start')}`);
  assert(windowStart(ids, 7, br, 'start') === 2, `got ${windowStart(ids, 7, br, 'start')}`);
  // …and the earliest line START (right after a break) is at 7
  assert(windowStart(ids, 5, br, 'line') === 7 && windowStart(ids, 7, br, 'line') === 3, `line mode ${windowStart(ids, 5, br, 'line')} ${windowStart(ids, 7, br, 'line')}`);
  assert(windowStart(ids, 5, null) === 4, 'without line breaks: the newest tokens that fit');
  assert(windowStart(ids, 20, br) === 0, 'everything fits');
  // the decoder keeps going past its context window either way
  const model = new GPT({ vocab: 3, ctx: 4, d: 8, layers: 1, heads: 2, mlp: 2, act: 'gelu', pos: 'learned', tie: true }, 2);
  const dec = new Decoder(model);
  dec.lineBreaks = br;
  for (const t of ids) dec.step(t);
  assert(dec.pos <= 4 && dec.tokens.length === dec.pos, 'window stays within the context');
  assert(dec.tokens[0] === ids[ids.length - dec.tokens.length], 'keeps the newest tokens');
  // a full window that ends with a line break: the new line starts again near position 0, after its line break
  const d2 = new Decoder(model);
  d2.lineBreaks = br;
  d2.feed([1, 2, 1, 0]);
  d2.step(2);
  assert(d2.pos === 2 && d2.tokens[0] === 0 && d2.tokens[1] === 2, `pos ${d2.pos} tokens ${d2.tokens}`);
  // 'line' mode: the new line starts fresh at position 0
  const d3 = new Decoder(model);
  d3.lineBreaks = br;
  d3.lineMode = 'line';
  d3.feed([1, 2, 1, 0]);
  d3.step(2);
  assert(d3.pos === 1 && d3.tokens[0] === 2, `line mode: pos ${d3.pos} tokens ${d3.tokens}`);
  assert(JSON.stringify(withStart([1, 2], br, vocab)) === '[0,1,2]' && JSON.stringify(withStart([0, 1], br, vocab)) === '[0,1]', 'start token');
  assert(JSON.stringify(withStart([1, 2], br, vocab, 'line')) === '[1,2]', 'no start token in line mode');
  // partial top-k selection agrees with a full sort
  const p = [0.1, 0.5, 0.05, 0.2, 0.15, 0.0];
  assert(JSON.stringify(topK(p, 3).map((x) => x[0])) === '[1,3,4]', `topK ${JSON.stringify(topK(p, 3))}`);
});

test('keyboard: edit distance, word splitting, suggestions and autocorrect', () => {
  // Damerau–Levenshtein: a swap of neighbours is ONE typo
  assert(editDistance('teh', 'the') === 1 && editDistance('tomorow', 'tomorrow') === 1 && editDistance('kitten', 'sitting') === 3, 'edit distances');
  assert(editDistance('abc', 'abcdefg', 2) === 3, 'gives up above the limit');
  // the leading space belongs to the word being typed (that is how the tokenizers cut text)
  assert(JSON.stringify(splitForSuggest('see you to')) === JSON.stringify({ context: 'see you', lead: ' to', partial: 'to' }), 'split partial word');
  assert(JSON.stringify(splitForSuggest('see you ')) === JSON.stringify({ context: 'see you', lead: ' ', partial: '' }), 'split after a space');
  assert(splitForSuggest('').lead === '' && splitForSuggest('hi').context === '', 'split at the start');
  // a tiny word-level model that knows four messages
  const lines = ['see you tomorrow', 'see you later', 'thank you so much', 'i am so tired'];
  const rng = new RNG(5);
  const text = Array.from({ length: 600 }, () => lines[rng.int(lines.length)]).join('\n') + '\n';
  const tok = Tokenizer.train('word', text, 64);
  const ids = tok.encode(text);
  const ctx = 8;
  const model = new GPT({ vocab: tok.size, ctx, d: 24, layers: 1, heads: 2, mlp: 2, act: 'gelu', pos: 'learned', tie: true }, 2);
  const tr = new LLMTrainer(model, ids, ids, { ...DEFAULT_TRAIN, steps: 300, batch: 16, lr: 1e-2, warmup: 20 }, { train: lineStarts(ids, tok.vocab, ctx, 'start'), val: lineStarts(ids, tok.vocab, ctx, 'start') });
  while (tr.step < 300) tr.trainStep();
  const km = { model, tok, align: 'start' as const };
  const next = suggestWords(km, 'see you ', 3).map((x) => x.word);
  assert(next.includes('tomorrow') && next.includes('later'), `after "see you": ${next}`);
  const done = suggestWords(km, 'thank you so mu', 3).map((x) => x.word);
  assert(done[0] === 'much', `completes "mu": ${done}`);
  const first = suggestWords(km, '', 3).map((x) => x.word);
  assert(first.includes('see') && first.includes('thank'), `first words: ${first}`);
  const known = knownWords(text);
  const fix = correctWord(km, 'see you ', 'tomorow', known);
  assert(fix.changed && fix.word === 'tomorrow', `tomorow → ${fix.word}`);
  const fix2 = correctWord(km, 'i am so ', 'tierd', known);
  assert(fix2.changed && fix2.word === 'tired', `tierd → ${fix2.word}`);
  assert(!correctWord(km, 'see you ', 'later', known).changed, 'known words are kept');
  const cap = correctWord(km, '', 'Thnak', known);
  assert(cap.changed && cap.word === 'Thank', `keeps the capital: ${cap.word}`);
});
