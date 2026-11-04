/**
 * The showcase: a single world running on the main thread at real-time
 * speed, driven either by the latest brain ("watch AI") or by you
 * (keyboard). This is what you see in the 3D view.
 */
import { Env, PHYS, compileScript } from './sim/env.ts';
import type { WorldRuntime } from './sim/runtime.ts';
import type { AgentSpec } from './sim/types.ts';
import { Brain } from './ml/algos/common.ts';
import type { DemoSet } from './ml/algos/bc.ts';
import { RNG } from './ml/rng.ts';

export type DriveMode = 'ai' | 'drive' | 'paused';

export interface StepInfo {
  reward: number;
  done: boolean;
  success: boolean;
  events: string[];
}

export class Showcase {
  env: Env | null = null;
  spec: AgentSpec | null = null;
  brain: Brain | null = null;
  mode: DriveMode = 'ai';
  speed = 1;
  deterministic = false;
  private rng = new RNG(2024);
  private acc = 0;
  /** seconds per brain decision (physics tick × ticks per decision) */
  get stepDt(): number {
    return PHYS.dt * (this.env ? this.env.P.substeps : PHYS.substeps);
  }
  /** shown in the 3D view, e.g. "checkpoint 3/12" when watching an older brain */
  label = '';
  alpha = 1;
  prev = { x: 0, y: 0, z: 0, yaw: 0 };
  action = new Float32Array(8);
  keys = new Set<string>();
  recording = false;
  private demoObs: number[] = [];
  private demoActs: number[] = [];
  demoCount = 0;
  demoObsSize = 0;
  demoActDims = 0;
  /** per-episode history for sparklines */
  rewards: number[] = [];
  values: number[] = [];
  episodes: { reward: number; length: number; success: boolean }[] = [];
  lastTrace: { pre: Float32Array[]; post: Float32Array[]; raw: Float32Array } | null = null;
  saliency: Float32Array | null = null;
  wantTrace = false;
  wantSaliency = false;
  onStep: ((info: StepInfo) => void) | null = null;
  onBeforeStep: (() => void) | null = null;
  onError: ((m: string) => void) | null = null;
  onLog: ((m: string) => void) | null = null;
  pauseOnEnd = false;
  private errorsSeen = new Set<string>();

  setSpec(spec: AgentSpec, rt: WorldRuntime): void {
    this.spec = spec;
    let factory = null;
    try {
      factory = compileScript(spec.script);
    } catch (e) {
      this.onError?.(`Script compile error: ${(e as Error).message}`);
    }
    this.env = new Env(rt, spec, factory, 777);
    this.env.onError = (m) => {
      if (this.errorsSeen.has(m)) return;
      this.errorsSeen.add(m);
      this.onError?.(m);
    };
    this.env.onLog = (m) => this.onLog?.(m);
    this.action = new Float32Array(Math.max(1, spec.axes.length));
    this.reset();
    if (this.demoObsSize !== this.env.obs.length || this.demoActDims !== spec.axes.length) this.clearDemos();
  }

  reset(): void {
    if (!this.env) return;
    this.env.reset();
    this.rewards = [];
    this.values = [];
    this.copyPrev();
    this.computeTrace();
  }

  setBrain(b: Brain | null): void {
    this.brain = b;
    this.computeTrace();
  }

  private copyPrev(): void {
    if (!this.env) return;
    const a = this.env.a;
    this.prev.x = a.x;
    this.prev.y = a.y;
    this.prev.z = a.z;
    this.prev.yaw = a.yaw;
  }

  /** Advance by real time dt. Returns the number of decision steps taken. */
  update(dt: number): number {
    if (!this.env || this.mode === 'paused') {
      this.alpha = 1;
      return 0;
    }
    const sdt = this.stepDt;
    this.acc += dt * this.speed;
    let n = 0;
    const maxSteps = Math.max(5, Math.ceil(this.speed * 4));
    while (this.acc >= sdt && n < maxSteps) {
      this.acc -= sdt;
      this.stepOnce();
      n++;
    }
    if (n === maxSteps) this.acc = Math.min(this.acc, sdt);
    this.alpha = Math.min(1, this.acc / sdt);
    return n;
  }

  stepOnce(): void {
    const env = this.env;
    if (!env || !this.spec) return;
    this.onBeforeStep?.();
    this.copyPrev();
    const obsBefore = this.recording && this.mode === 'drive' ? Array.from(env.obs) : null;
    if (this.mode === 'drive') this.keyboardAction();
    else if (this.brain && this.brainFits()) this.brain.act(env.obs, this.action, this.rng, this.deterministic);
    else this.randomAction();
    if (obsBefore) {
      this.demoObs.push(...obsBefore);
      for (let i = 0; i < this.spec.axes.length; i++) this.demoActs.push(this.action[i]);
      this.demoCount++;
    }
    env.step(this.action);
    const info: StepInfo = { reward: env.stepReward, done: env.done, success: env.success, events: env.events.slice() };
    this.rewards.push(env.stepReward);
    this.values.push(this.brain && this.brainFits() ? this.brain.lastValue : 0);
    if (this.rewards.length > 2000) {
      this.rewards.shift();
      this.values.shift();
    }
    if (env.done) {
      this.episodes.push({ reward: env.epReturn, length: env.steps, success: env.success });
      if (this.episodes.length > 200) this.episodes.shift();
    }
    this.onStep?.(info);
    if (env.done) {
      env.reset();
      this.copyPrev();
      this.rewards = [];
      this.values = [];
      if (this.pauseOnEnd) this.mode = 'paused';
    }
    this.computeTrace();
  }

  brainFits(): boolean {
    const b = this.brain;
    const env = this.env;
    if (!b || !env || !this.spec) return false;
    if (b.net.inSize !== env.obs.length) return false;
    const sp = b.snap.space;
    if (sp.branches.length !== this.spec.axes.length) return false;
    if ((sp.kind === 'continuous') !== (this.spec.actionMode === 'continuous')) return false;
    if (sp.kind === 'discrete' && b.snap.kind === 'policy') return sp.branches.every((n, i) => n === this.spec!.axes[i].options.length);
    return true;
  }

  /** Activations of every layer for the current observation (for the brain view). */
  computeTrace(): void {
    const b = this.brain;
    const env = this.env;
    if (!b || !env || !this.brainFits() || (!this.wantTrace && !this.wantSaliency)) {
      this.lastTrace = null;
      return;
    }
    const x = Float32Array.from(b.normalize(env.obs));
    const t = b.net.trace(x);
    this.lastTrace = { ...t, raw: Float32Array.from(env.obs) };
    if (this.wantSaliency) {
      // saliency = |∂(chosen outputs)/∂(inputs)|: which senses would change the decision most?
      const out = b.net.forward(x, 1);
      const d = new Float32Array(b.net.outSize);
      const sp = b.snap.space;
      if (b.snap.kind === 'q') {
        let best = 0;
        for (let i = 1; i < out.length; i++) if (out[i] > out[best]) best = i;
        d[best] = 1;
      } else if (sp.kind === 'discrete') {
        let o = 0;
        for (let k = 0; k < sp.branches.length; k++) {
          let best = 0;
          for (let i = 1; i < sp.branches[k]; i++) if (out[o + i] > out[o + best]) best = i;
          d[o + best] = 1;
          o += sp.branches[k];
        }
      } else d.fill(1);
      const g = b.net.backward(d, 1, true);
      b.net.zeroGrad();
      this.saliency = Float32Array.from(g.subarray(0, x.length), Math.abs);
    }
  }

  private randomAction(): void {
    if (!this.spec) return;
    this.spec.axes.forEach((ax, i) => {
      this.action[i] = this.spec!.actionMode === 'discrete' ? this.rng.int(ax.options.length) : this.rng.uniform(-1, 1);
    });
  }

  private keyboardAction(): void {
    const spec = this.spec!;
    const k = this.keys;
    const fwd = (k.has('KeyW') || k.has('ArrowUp') ? 1 : 0) - (k.has('KeyS') || k.has('ArrowDown') ? 1 : 0);
    const turn = (k.has('KeyD') || k.has('ArrowRight') ? 1 : 0) - (k.has('KeyA') || k.has('ArrowLeft') ? 1 : 0);
    const strafe = (k.has('KeyE') ? 1 : 0) - (k.has('KeyQ') ? 1 : 0);
    const jump = k.has('Space') ? 1 : 0;
    const kick = k.has('KeyF') || k.has('ShiftLeft') ? 1 : 0;
    spec.axes.forEach((ax, i) => {
      let v = ax.axis === 'move' ? fwd : ax.axis === 'turn' ? turn : ax.axis === 'strafe' ? strafe : ax.axis === 'jump' ? jump : kick;
      if (spec.actionMode === 'discrete') {
        let best = 0;
        for (let j = 1; j < ax.options.length; j++) if (Math.abs(ax.options[j] - v) < Math.abs(ax.options[best] - v)) best = j;
        this.action[i] = best;
      } else {
        if (ax.axis === 'move' && Math.min(...ax.options) >= 0) v = v > 0 ? 1 : -1;
        if (ax.axis === 'jump' || ax.axis === 'kick') v = v > 0 ? 1 : -1;
        this.action[i] = v;
      }
    });
  }

  clearDemos(): void {
    this.demoObs = [];
    this.demoActs = [];
    this.demoCount = 0;
    this.demoObsSize = this.env ? this.env.obs.length : 0;
    this.demoActDims = this.spec ? this.spec.axes.length : 0;
  }

  demos(): DemoSet | null {
    if (!this.demoCount) return null;
    return { obs: Float32Array.from(this.demoObs), acts: Float32Array.from(this.demoActs), count: this.demoCount, obsSize: this.demoObsSize, actDims: this.demoActDims };
  }
}
