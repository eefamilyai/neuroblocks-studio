/**
 * Runs mini-LLM training in small time slices (inside a Web Worker, or on
 * the main thread as a fallback) and reports progress, samples and weights.
 */
import { RNG } from '../ml/rng.ts';
import { GPT } from './gpt.ts';
import { Decoder, sampleToken } from './decode.ts';
import { Tokenizer } from './tokenizer.ts';
import { LLMTrainer } from './train.ts';
import type { FromLLM, ToLLM } from './protocol.ts';

export class LLMHost {
  private post: (m: FromLLM, transfer?: Transferable[]) => void;
  private trainer: LLMTrainer | null = null;
  private tok: Tokenizer | null = null;
  private running = false;
  private stepTarget = -1;
  private losses: [number, number][] = [];
  private vals: [number, number][] = [];
  private lastProgress = 0;
  private lastWeights = 0;
  private lastSample = 0;
  private lastSampleStep = -1;
  private rate: [number, number][] = [];
  private sampleCfg = { prompt: '', temperature: 0.8, every: 200 };
  private rng = new RNG(77);
  threads = 1;
  sliceMs: number;

  constructor(post: (m: FromLLM, transfer?: Transferable[]) => void, sliceMs = 40) {
    this.post = post;
    this.sliceMs = sliceMs;
  }

  get busy(): boolean {
    return !!this.trainer && this.running;
  }

  handle(msg: ToLLM): void {
    switch (msg.type) {
      case 'start': {
        try {
          this.start(msg);
        } catch (e) {
          this.trainer = null;
          this.running = false;
          this.post({ type: 'error', msg: (e as Error).message });
        }
        break;
      }
      case 'pause':
        this.running = false;
        this.stepTarget = -1;
        this.sendProgress(true);
        this.sendWeights();
        break;
      case 'resume':
        if (this.trainer) this.running = true;
        this.stepTarget = -1;
        this.rate = [];
        break;
      case 'step':
        if (!this.trainer) break;
        this.stepTarget = this.trainer.step + 1;
        this.running = true;
        break;
      case 'stop':
        if (this.trainer) {
          this.sendProgress(true);
          this.sendWeights();
        }
        this.running = false;
        this.trainer = null;
        break;
      case 'threads':
        this.threads = Math.max(1, msg.n | 0);
        break;
    }
  }

  private start(msg: Extract<ToLLM, { type: 'start' }>): void {
    const p = msg.project;
    const t0 = performance.now();
    const tok = msg.init?.tokenizer ? new Tokenizer(msg.init.tokenizer) : Tokenizer.train(p.tokenizer.kind, msg.text, p.tokenizer.vocabSize);
    const ids = tok.encode(msg.text);
    const split = msg.valStart !== undefined ? tok.encode(msg.text.slice(0, msg.valStart)).length : Math.floor(ids.length * 0.9);
    const model = new GPT({ ...p.model, vocab: tok.size }, p.train.seed);
    if (msg.init?.weights) model.setWeights(msg.init.weights);
    const trainer = new LLMTrainer(model, ids.subarray(0, split), ids.subarray(split), p.train);
    if (msg.init) {
      trainer.step = msg.init.step;
      trainer.tokensSeen = msg.init.tokens;
    }
    this.trainer = trainer;
    this.tok = tok;
    this.sampleCfg = { ...p.sample };
    this.losses = [];
    this.vals = [];
    this.rate = [];
    this.lastSampleStep = -1;
    this.running = true;
    this.stepTarget = msg.stepOnly ? trainer.step + 1 : -1;
    const ms = performance.now() - t0;
    this.post({ type: 'ready', tokenizer: tok.toJSON(), stats: { chars: msg.text.length, tokens: ids.length, trainTokens: split, valTokens: ids.length - split, vocab: tok.size, params: model.numParams }, step: trainer.step });
    if (!msg.init && p.tokenizer.kind !== 'char') this.post({ type: 'log', level: 'info', msg: `Tokenizer (${p.tokenizer.kind}) learned ${tok.size} tokens in ${ms.toFixed(0)} ms.` });
    if (!msg.init) {
      // a first value: an untrained model should be close to −log(1/vocab)
      this.vals.push([trainer.step, trainer.evaluate(2)]);
    }
    this.sendWeights();
  }

  /** Do a slice of work. Returns the delay before the next slice (−1 = idle). */
  tick(): number {
    const tr = this.trainer;
    if (!tr || !this.running) return -1;
    const t0 = performance.now();
    try {
      do {
        const r = tr.trainStep();
        this.losses.push([tr.step, r.loss]);
        if (!Number.isFinite(r.loss)) throw new Error('the loss became NaN/∞: the learning rate is probably too high');
        if (tr.cfg.evalEvery > 0 && tr.step % tr.cfg.evalEvery === 0) this.vals.push([tr.step, tr.evaluate()]);
        if (this.stepTarget >= 0 && tr.step >= this.stepTarget) break;
      } while (performance.now() - t0 < this.sliceMs);
    } catch (e) {
      this.running = false;
      this.post({ type: 'error', msg: `Training stopped: ${(e as Error).message}` });
      return -1;
    }
    const now = performance.now();
    this.rate.push([now, tr.tokensSeen]);
    while (this.rate.length > 2 && now - this.rate[0][0] > 4000) this.rate.shift();
    if (now - this.lastProgress > 250) this.sendProgress(false);
    const sampleDue = this.sampleCfg.every > 0 && tr.step - this.lastSampleStep >= this.sampleCfg.every;
    if (sampleDue || (this.lastSampleStep < 0 && tr.step >= 20) || now - this.lastSample > 15000) this.sendSample();
    if (now - this.lastWeights > 1500) this.sendWeights();
    if (this.stepTarget >= 0 && tr.step >= this.stepTarget) {
      this.stepTarget = -1;
      this.running = false;
      this.vals.push([tr.step, tr.evaluate(2)]);
      this.sendProgress(true);
      this.sendSample();
      this.sendWeights();
      this.post({ type: 'paused', reason: `one training step (step ${tr.step})` });
      return -1;
    }
    if (tr.cfg.steps > 0 && tr.step >= tr.cfg.steps) {
      this.running = false;
      if (!this.vals.length || this.vals[this.vals.length - 1][0] !== tr.step) this.vals.push([tr.step, tr.evaluate()]);
      this.sendProgress(true);
      this.sendSample();
      this.sendWeights();
      this.post({ type: 'finished', reason: `reached ${tr.step.toLocaleString()} steps` });
      return -1;
    }
    return 0;
  }

  private sendProgress(force: boolean): void {
    const tr = this.trainer;
    if (!tr) return;
    const now = performance.now();
    if (!force && now - this.lastProgress < 250) return;
    this.lastProgress = now;
    let tps = 0;
    if (this.rate.length >= 2) {
      const [t0, s0] = this.rate[0];
      const [t1, s1] = this.rate[this.rate.length - 1];
      if (t1 > t0) tps = ((s1 - s0) * 1000) / (t1 - t0);
    }
    this.post({ type: 'progress', step: tr.step, tokens: tr.tokensSeen, losses: this.losses, vals: this.vals, lr: tr.lr, gradNorm: tr.lastGradNorm, tokPerSec: this.running ? tps : 0, running: this.running, threads: this.threads });
    this.losses = [];
    this.vals = [];
  }

  private sendSample(): void {
    const tr = this.trainer;
    const tok = this.tok;
    if (!tr || !tok) return;
    this.lastSample = performance.now();
    this.lastSampleStep = tr.step;
    const dec = new Decoder(tr.model);
    let ids = Array.from(tok.encode(this.sampleCfg.prompt));
    if (!ids.length) ids = [0];
    const out: number[] = [];
    let logits = dec.feed(ids);
    for (let i = 0; i < 160; i++) {
      const t = sampleToken(logits, this.rng, this.sampleCfg.temperature);
      out.push(t);
      logits = dec.step(t);
    }
    this.post({ type: 'sample', step: tr.step, text: this.sampleCfg.prompt + tok.decode(out) });
  }

  private sendWeights(): void {
    const tr = this.trainer;
    if (!tr) return;
    this.lastWeights = performance.now();
    const ws = tr.model.getWeights();
    this.post({ type: 'weights', step: tr.step, tokens: tr.tokensSeen, weights: ws }, ws.map((w) => w.buffer));
  }
}
