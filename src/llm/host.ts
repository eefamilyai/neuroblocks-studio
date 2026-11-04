/**
 * Runs mini-LLM training in small time slices (inside a Web Worker, or on
 * the main thread as a fallback) and reports progress, samples and weights.
 */
import { RNG } from '../ml/rng.ts';
import { GPT } from './gpt.ts';
import { Decoder, lineBreakTable, sampleToken, windowStart, withStart } from './decode.ts';
import type { LineMode } from './decode.ts';
import { Tokenizer } from './tokenizer.ts';
import { LLMTrainer, lineStarts } from './train.ts';
import { getDataset } from './datasets.ts';
import { pickCases, probeAccuracy } from './probe.ts';
import type { ProbeCase } from './probe.ts';
import type { FromLLM, ToLLM } from './protocol.ts';
import { HelperPool, addGrads, packWeights, scaleGrads, splitBatch } from './parallel.ts';

export class LLMHost {
  private out: (m: FromLLM, transfer?: Transferable[]) => void;
  private run = 0;
  private trainer: LLMTrainer | null = null;
  private tok: Tokenizer | null = null;
  private running = false;
  private stepTarget = -1;
  private losses: [number, number][] = [];
  private vals: [number, number][] = [];
  private accs: [number, number][] = [];
  private probeCases: ProbeCase[] = [];
  private probeKind: 'exact' | 'next-word' = 'exact';
  private align: LineMode | undefined = undefined;
  private lastProgress = 0;
  private lastWeights = 0;
  private lastSample = 0;
  private lastSampleStep = -1;
  private rate: [number, number][] = [];
  private sampleCfg = { prompt: '', temperature: 0.8, every: 200 };
  /** one example per line: generation windows start at line starts too */
  private breaks: Uint8Array | null = null;
  private rng = new RNG(77);
  /** helper workers for data-parallel training (null = one thread) */
  private pool: HelperPool | null = null;
  private inflight = false;
  private flat: Float32Array | null = null;
  /** called when an asynchronous (multi-threaded) step has finished and the loop should continue */
  onWake: (() => void) | null = null;
  sliceMs: number;

  constructor(post: (m: FromLLM, transfer?: Transferable[]) => void, sliceMs = 40) {
    this.out = post;
    this.sliceMs = sliceMs;
  }

  private post(m: FromLLM, transfer?: Transferable[]): void {
    m.run = this.run;
    this.out(m, transfer);
  }

  get busy(): boolean {
    return !!this.trainer && this.running;
  }

  handle(msg: ToLLM): void {
    switch (msg.type) {
      case 'start': {
        this.run = msg.run ?? 0;
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
      case 'config':
        this.sampleCfg = { ...msg.sample };
        this.trainer?.setConfig(msg.train);
        break;
      case 'helpers':
        this.setHelpers(msg.ports);
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
    const trainIds = ids.subarray(0, split);
    const valIds = ids.subarray(split);
    const starts = msg.align ? { train: lineStarts(trainIds, tok.vocab, p.model.ctx, msg.align), val: lineStarts(valIds, tok.vocab, p.model.ctx, msg.align) } : null;
    const trainer = new LLMTrainer(model, trainIds, valIds, p.train, starts);
    if (msg.init) {
      trainer.step = msg.init.step;
      trainer.tokensSeen = msg.init.tokens;
    }
    this.trainer = trainer;
    this.tok = tok;
    this.breaks = msg.align ? lineBreakTable(tok.vocab) : null;
    this.inflight = false;
    this.pool?.init(model.cfg);
    this.sampleCfg = { ...p.sample };
    this.losses = [];
    this.vals = [];
    this.accs = [];
    this.rate = [];
    this.lastSampleStep = -1;
    // a task with right answers (sums, copying): test cases from the validation text
    const probe = p.data.dataset !== 'custom' ? getDataset(p.data.dataset)?.probe : undefined;
    let valText = msg.text.slice(msg.valStart ?? Math.floor(msg.text.length * 0.9));
    valText = valText.slice(valText.indexOf('\n') + 1); // skip a possibly cut first line
    this.probeCases = probe ? pickCases(probe.cases(valText), 50) : [];
    this.probeKind = probe?.kind ?? 'exact';
    this.align = msg.align;
    this.running = true;
    this.stepTarget = msg.stepOnly ? trainer.step + 1 : -1;
    const ms = performance.now() - t0;
    this.post({ type: 'ready', tokenizer: tok.toJSON(), stats: { chars: msg.text.length, tokens: ids.length, trainTokens: split, valTokens: ids.length - split, vocab: tok.size, params: model.numParams }, step: trainer.step, probe: probe && this.probeCases.length ? { name: probe.name, help: probe.help } : null });
    if (!msg.init && p.tokenizer.kind !== 'char') this.post({ type: 'log', level: 'info', msg: `Tokenizer (${p.tokenizer.kind}) learned ${tok.size} tokens in ${ms.toFixed(0)} ms.` });
    if (!msg.init) {
      // a first value: an untrained model should be close to −log(1/vocab)
      this.vals.push([trainer.step, trainer.evaluate(2)]);
    }
    this.sendWeights();
  }

  get threads(): number {
    return 1 + (this.pool?.size ?? 0);
  }

  /** Use these helper workers (one MessagePort each) from now on. */
  private setHelpers(ports: MessagePort[]): void {
    this.pool?.close();
    this.pool = ports.length ? new HelperPool(ports) : null;
    if (this.pool && this.trainer) this.pool.init(this.trainer.model.cfg);
  }

  /** Do a slice of work. Returns the delay before the next slice (−1 = idle, −2 = waiting for helper threads). */
  tick(): number {
    const tr = this.trainer;
    if (!tr || !this.running) return -1;
    if (this.inflight) return -2;
    if (this.pool && tr.cfg.batch >= 2 && this.stepTarget < 0) {
      this.parallelStep(tr, this.pool);
      return -2;
    }
    const t0 = performance.now();
    try {
      do {
        const r = tr.trainStep();
        if (this.afterStep(tr, r.loss)) break;
      } while (performance.now() - t0 < this.sliceMs);
    } catch (e) {
      this.fail(e as Error);
      return -1;
    }
    return this.afterSlice(tr);
  }

  /** One step with the batch split between this thread and the helpers. */
  private parallelStep(tr: LLMTrainer, pool: HelperPool): void {
    const model = tr.model;
    const { x, y, B, T } = tr.sampleBatch();
    // largest shares to the helpers; this thread takes the smallest (it also does the update and the bookkeeping)
    const shares = splitBatch(B, pool.size + 1).reverse();
    this.inflight = true;
    this.flat = packWeights(model, this.flat && this.flat.length === this.flatLen(tr) ? this.flat : undefined);
    const jobs: Promise<{ loss: number; grads: Float32Array }>[] = [];
    let off = shares[0];
    for (let k = 1; k < shares.length; k++) {
      jobs.push(pool.work(k - 1, this.flat, x.slice(off * T, (off + shares[k]) * T), y.slice(off * T, (off + shares[k]) * T), shares[k], T));
      off += shares[k];
    }
    // this thread's share, while the helpers work on theirs
    const b0 = shares[0];
    model.forward(x.subarray(0, b0 * T), b0, T);
    let loss = (model.lossAndGrad(y.subarray(0, b0 * T)) * b0) / B;
    model.zeroGrad();
    model.backward();
    scaleGrads(model, b0 / B);
    Promise.all(jobs).then(
      (res) => {
        this.inflight = false;
        if (this.trainer !== tr) return; // a new run started meanwhile
        res.forEach((r, i) => {
          addGrads(model, r.grads, shares[i + 1] / B);
          loss += (r.loss * shares[i + 1]) / B;
        });
        const out = tr.applyUpdate(loss, B * T);
        try {
          this.afterStep(tr, out.loss);
        } catch (e) {
          this.fail(e as Error);
          return;
        }
        if (this.afterSlice(tr) >= 0) this.onWake?.();
      },
      (err: Error) => {
        this.inflight = false;
        if (this.pool === pool) {
          // a helper failed: carry on with one thread
          pool.close();
          this.pool = null;
          this.post({ type: 'log', level: 'warn', msg: `Multi-threaded training stopped (${err.message}); continuing on one thread.` });
        }
        if (this.trainer === tr && this.running) this.onWake?.();
      },
    );
  }

  private flatLen(tr: LLMTrainer): number {
    let n = 0;
    for (const p of tr.model.params) n += p.w.length;
    return n;
  }

  private fail(e: Error): void {
    this.running = false;
    this.post({ type: 'error', msg: `Training stopped: ${e.message}` });
  }

  /** Bookkeeping after every step. Returns true when a single step was asked for and is done. */
  private afterStep(tr: LLMTrainer, loss: number): boolean {
    this.losses.push([tr.step, loss]);
    if (!Number.isFinite(loss)) throw new Error('the loss became NaN/∞: the learning rate is probably too high');
    if (tr.cfg.evalEvery > 0 && tr.step % tr.cfg.evalEvery === 0) this.evaluate();
    return this.stepTarget >= 0 && tr.step >= this.stepTarget;
  }

  /** Reports and stop conditions after a slice of steps. Returns the delay before the next slice (−1 = stop). */
  private afterSlice(tr: LLMTrainer): number {
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
      if (!this.vals.length || this.vals[this.vals.length - 1][0] !== tr.step) this.evaluate();
      this.sendProgress(true);
      this.sendSample();
      this.sendWeights();
      this.post({ type: 'finished', reason: `reached ${tr.step.toLocaleString()} steps` });
      return -1;
    }
    return this.running ? 0 : -1;
  }

  /** Validation loss (and task accuracy, if the dataset has test cases). */
  private evaluate(): void {
    const tr = this.trainer!;
    this.vals.push([tr.step, tr.evaluate()]);
    if (this.probeCases.length && this.tok) this.accs.push([tr.step, probeAccuracy(tr.model, this.tok, this.probeCases, this.probeKind, this.align)]);
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
    this.post({ type: 'progress', step: tr.step, tokens: tr.tokensSeen, losses: this.losses, vals: this.vals, accs: this.accs, lr: tr.lr, gradNorm: tr.lastGradNorm, tokPerSec: this.running ? tps : 0, running: this.running, threads: this.threads });
    this.losses = [];
    this.vals = [];
    this.accs = [];
  }

  private sendSample(): void {
    const tr = this.trainer;
    const tok = this.tok;
    if (!tr || !tok) return;
    this.lastSample = performance.now();
    this.lastSampleStep = tr.step;
    const dec = new Decoder(tr.model);
    dec.lineBreaks = this.breaks;
    dec.lineMode = this.align ?? 'start';
    let ids = withStart(Array.from(tok.encode(this.sampleCfg.prompt)), this.breaks, tok.vocab, dec.lineMode);
    if (!ids.length) ids = [0];
    ids = ids.slice(windowStart(ids, tr.model.cfg.ctx - 1, this.breaks, dec.lineMode));
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
