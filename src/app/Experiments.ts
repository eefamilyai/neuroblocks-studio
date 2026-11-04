/**
 * 🧪 Experiments: train several brains on the current program, changing ONE
 * setting (a hyperparameter sweep), in parallel background workers, and
 * compare their learning curves. This is how ML researchers find good
 * settings, and how you see that RL results vary from seed to seed.
 */
import type { BrainSnapshot } from '../ml/algos/common.ts';
import type { WorldEdits } from '../sim/runtime.ts';
import type { AgentSpec, AlgoKind } from '../sim/types.ts';
import { defaultTrain } from '../sim/types.ts';
import { TrainerClient } from '../worker/client.ts';
import type { FromWorker } from '../worker/protocol.ts';
import { LineChart } from '../ui/charts.ts';
import { btn, clear, fmt, fmtSteps, fmtTime, h, section, setText, toast } from '../ui/dom.ts';

interface Param {
  id: string;
  label: string;
  defaults: string;
  help: string;
  apply: (spec: AgentSpec, v: string) => string | null;
}

const num = (v: string) => Number(v.trim());

const PARAMS: Param[] = [
  {
    id: 'lr',
    label: 'Learning rate',
    defaults: '0.0001, 0.0003, 0.001, 0.003',
    help: 'Too small: slow. Too big: unstable. The classic first thing to tune.',
    apply: (s, v) => (Number.isFinite(num(v)) && num(v) > 0 ? ((s.train.lr = num(v)), null) : 'not a positive number'),
  },
  {
    id: 'gamma',
    label: 'Discount γ',
    defaults: '0.9, 0.97, 0.99, 0.999',
    help: 'How far ahead the robot plans. Low γ = short-sighted.',
    apply: (s, v) => (num(v) >= 0 && num(v) <= 1 ? ((s.train.gamma = num(v)), null) : 'must be between 0 and 1'),
  },
  {
    id: 'entropy',
    label: 'Entropy bonus',
    defaults: '0, 0.003, 0.01, 0.05',
    help: 'Exploration pressure (PPO / REINFORCE).',
    apply: (s, v) => (num(v) >= 0 ? ((s.train.entropy = num(v)), null) : 'must be ≥ 0'),
  },
  {
    id: 'width',
    label: 'Neurons per layer',
    defaults: '8, 32, 128',
    help: 'Network capacity: the same number of layers, each this wide.',
    apply: (s, v) => {
      const n = Math.round(num(v));
      if (!(n >= 1 && n <= 512)) return 'between 1 and 512';
      s.brain.layers = s.brain.layers.map((l) => ({ ...l, units: n }));
      return null;
    },
  },
  {
    id: 'depth',
    label: 'Number of layers',
    defaults: '1, 2, 3',
    help: 'Deeper networks can represent more, but are harder to train.',
    apply: (s, v) => {
      const n = Math.round(num(v));
      if (!(n >= 1 && n <= 6)) return 'between 1 and 6';
      const first = s.brain.layers[0] ?? { units: 32, activation: 'tanh' as const };
      s.brain.layers = Array.from({ length: n }, () => ({ ...first }));
      return null;
    },
  },
  {
    id: 'numEnvs',
    label: 'Parallel worlds',
    defaults: '4, 16, 32',
    help: 'More worlds = more varied experience per update.',
    apply: (s, v) => {
      const n = Math.round(num(v));
      return n >= 1 && n <= 64 ? ((s.train.numEnvs = n), null) : 'between 1 and 64';
    },
  },
  {
    id: 'algo',
    label: 'Algorithm',
    defaults: 'ppo, dqn, reinforce, evolution',
    help: 'Same task, different learning algorithms (default settings for each).',
    apply: (s, v) => {
      const a = v.trim().toLowerCase() as AlgoKind;
      if (!['ppo', 'dqn', 'reinforce', 'evolution'].includes(a)) return 'ppo, dqn, reinforce or evolution';
      if (a === 'dqn' && s.actionMode === 'continuous') return 'DQN needs discrete actions';
      const keep = { numEnvs: s.train.numEnvs, seed: s.train.seed, stopSteps: s.train.stopSteps };
      s.train = { ...defaultTrain(a), ...keep };
      return null;
    },
  },
  {
    id: 'seed',
    label: 'Random seed (repeat)',
    defaults: '1, 2, 3, 4',
    help: 'The same settings with different luck: shows how noisy RL results are.',
    apply: (s, v) => (Number.isFinite(num(v)) ? ((s.train.seed = Math.round(num(v))), null) : 'not a number'),
  },
  {
    id: 'clip',
    label: 'PPO clip ε',
    defaults: '0.05, 0.2, 0.5',
    help: 'How much the policy may change per update.',
    apply: (s, v) => (num(v) > 0 && num(v) <= 1 ? ((s.train.clip = num(v)), null) : 'between 0 and 1'),
  },
  {
    id: 'memory',
    label: 'Memory (frames)',
    defaults: '1, 2, 4',
    help: 'Frame stacking: how many recent observations the brain gets.',
    apply: (s, v) => {
      const n = Math.round(num(v));
      return n >= 1 && n <= 8 ? ((s.frameStack = n), null) : 'between 1 and 8';
    },
  },
];

const COLORS = ['#3ccf8e', '#5ab4ff', '#f472b6', '#f2b84b', '#9b7bff', '#fb923c', '#2dd4bf', '#e879f9'];

interface Run {
  value: string;
  spec: AgentSpec;
  color: string;
  xs: number[];
  rewards: number[];
  success: boolean[];
  steps: number;
  status: 'queued' | 'running' | 'done' | 'error' | 'stopped';
  snap: BrainSnapshot | null;
  t0: number;
  seconds: number;
  client: TrainerClient | null;
  error?: string;
}

export interface ExperimentHooks {
  spec(): AgentSpec | null;
  edits(): WorldEdits;
  watch(snap: BrainSnapshot, label: string): void;
  log(msg: string, level?: 'info' | 'warn' | 'err' | 'good' | 'learn'): void;
}

export class ExperimentsPanel {
  readonly el: HTMLElement;
  private hooks: ExperimentHooks;
  private param = PARAMS[0];
  private values: HTMLInputElement;
  private budget = 50000;
  private parallel = Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 4) - 2));
  private runs: Run[] = [];
  private chart: LineChart;
  private table: HTMLElement;
  private status: HTMLElement;
  private runBtn: HTMLButtonElement;
  private stopBtn: HTMLButtonElement;
  private help: HTMLElement;
  private dirty = false;
  private lastTable = 0;

  constructor(hooks: ExperimentHooks) {
    this.hooks = hooks;
    const paramSel = h('select', { class: 'sel sm' }) as HTMLSelectElement;
    for (const p of PARAMS) paramSel.append(h('option', { value: p.id }, p.label));
    this.values = h('input', { class: 'input sm', value: this.param.defaults, style: { width: '100%' }, title: 'Comma-separated values, one training run each' }) as HTMLInputElement;
    this.help = h('div', { class: 'note' }, this.param.help);
    paramSel.addEventListener('change', () => {
      this.param = PARAMS.find((p) => p.id === paramSel.value)!;
      this.values.value = this.param.defaults;
      setText(this.help, this.param.help);
    });
    const budgetSel = h('select', { class: 'sel sm', title: 'Environment steps per run' }) as HTMLSelectElement;
    for (const n of [20000, 50000, 100000, 200000, 500000]) budgetSel.append(h('option', { value: String(n), selected: n === this.budget }, `${fmtSteps(n)} steps each`));
    budgetSel.addEventListener('change', () => (this.budget = Number(budgetSel.value)));
    const parSel = h('select', { class: 'sel sm', title: 'How many runs train at the same time (each uses one CPU core)' }) as HTMLSelectElement;
    const maxPar = Math.max(1, Math.min(8, navigator.hardwareConcurrency || 4));
    for (let i = 1; i <= maxPar; i++) parSel.append(h('option', { value: String(i), selected: i === this.parallel }, `${i} at a time`));
    parSel.addEventListener('change', () => (this.parallel = Number(parSel.value)));
    this.runBtn = btn({ icon: 'play', label: 'Run experiment', cls: 'sm primary', onclick: () => this.start() });
    this.stopBtn = btn({ label: 'Stop', cls: 'sm', onclick: () => this.stopAll() });
    this.stopBtn.classList.add('disabled');
    this.chart = new LineChart('Reward per episode, every run', 'Each colour is one training run of the experiment. Compare how fast and how high they climb.', { tall: true });
    this.table = h('div');
    this.status = h('div', { class: 'note' }, 'Uses your current program and world. Your main training is not affected.');
    this.el = h(
      'div',
      null,
      section('Experiment', 'tubes', [
        h(
          'div',
          { class: 'card col' },
          h('div', { class: 'note', style: { margin: 0 } }, 'Train several brains on your current program, changing ONE setting, and compare. Researchers call this a hyperparameter sweep.'),
          h('label', { class: 'field' }, h('span', { class: 'lbl' }, 'Change this setting'), paramSel),
          h('label', { class: 'field' }, h('span', { class: 'lbl' }, 'Try these values'), this.values),
          this.help,
          h('div', { class: 'row', style: { flexWrap: 'wrap' } }, budgetSel, parSel, h('div', { class: 'spacer' }), this.stopBtn, this.runBtn),
          this.status,
        ),
      ]),
      section('Results', 'chart', [this.chart.el, h('div', { style: { marginTop: '8px' } }, this.table)]),
    );
    this.renderTable();
  }

  private start(): void {
    const base = this.hooks.spec();
    if (!base) return toast('Fix the errors in your program first');
    const vals = this.values.value
      .split(',')
      .map((v) => v.trim())
      .filter(Boolean)
      .slice(0, 8);
    if (!vals.length) return toast('Enter some values, separated by commas');
    this.stopAll(true);
    this.runs = [];
    for (const [i, v] of vals.entries()) {
      const spec = structuredClone(base);
      const err = this.param.apply(spec, v);
      if (err) {
        toast(`"${v}": ${err}`, 3500);
        return;
      }
      spec.train.stopSteps = this.budget;
      spec.train.targetSuccess = 0;
      spec.train.targetReward = NaN;
      spec.train.stopMinutes = 0;
      this.runs.push({ value: v, spec, color: COLORS[i % COLORS.length], xs: [], rewards: [], success: [], steps: 0, status: 'queued', snap: null, t0: 0, seconds: 0, client: null });
    }
    this.hooks.log(`Experiment: ${this.param.label} = ${vals.join(' / ')}, ${fmtSteps(this.budget)} steps each, ${this.parallel} at a time.`, 'learn');
    this.runBtn.classList.add('disabled');
    this.stopBtn.classList.remove('disabled');
    this.pump();
    this.dirty = true;
  }

  private pump(): void {
    const running = this.runs.filter((r) => r.status === 'running').length;
    let free = this.parallel - running;
    for (const r of this.runs) {
      if (free <= 0) break;
      if (r.status !== 'queued') continue;
      this.launch(r);
      free--;
    }
    if (!this.runs.some((r) => r.status === 'queued' || r.status === 'running')) {
      this.runBtn.classList.remove('disabled');
      this.stopBtn.classList.add('disabled');
      const done = this.runs.filter((r) => r.status === 'done');
      if (done.length) {
        const best = done.reduce((a, b) => (this.score(b) > this.score(a) ? b : a));
        this.hooks.log(`Experiment finished: best ${this.param.label} = ${best.value} (mean reward ${fmt(this.score(best), 2)}).`, 'good');
      }
    }
    this.dirty = true;
  }

  private launch(r: Run): void {
    let client: TrainerClient;
    try {
      client = new TrainerClient({ allowFallback: false });
    } catch {
      r.status = 'error';
      r.error = 'background workers are not available in this browser';
      return;
    }
    r.client = client;
    r.status = 'running';
    r.t0 = performance.now();
    client.onMessage = (m: FromWorker) => this.onMessage(r, m);
    client.send({ type: 'ghosts', on: false });
    client.send({ type: 'start', spec: r.spec, edits: this.hooks.edits(), init: null, mode: 'rl', demos: null });
  }

  private onMessage(r: Run, m: FromWorker): void {
    switch (m.type) {
      case 'progress':
        r.steps = m.steps;
        for (const e of m.episodes) {
          r.xs.push(e.step);
          r.rewards.push(e.reward);
          r.success.push(e.success);
        }
        r.seconds = (performance.now() - r.t0) / 1000;
        this.dirty = true;
        break;
      case 'brain':
        r.snap = m.snap;
        break;
      case 'finished':
        this.finish(r, 'done');
        break;
      case 'error':
        r.error = m.msg;
        this.finish(r, 'error');
        break;
    }
  }

  private finish(r: Run, status: Run['status']): void {
    r.status = status;
    r.seconds = (performance.now() - r.t0) / 1000;
    // let the final brain message arrive, then free the worker
    const c = r.client;
    r.client = null;
    setTimeout(() => c?.terminate(), 300);
    this.pump();
  }

  stopAll(silent = false): void {
    let any = false;
    for (const r of this.runs) {
      if (r.status === 'running' || r.status === 'queued') {
        any = true;
        r.client?.terminate();
        r.client = null;
        r.status = 'stopped';
      }
    }
    this.runBtn.classList.remove('disabled');
    this.stopBtn.classList.add('disabled');
    if (any && !silent) this.hooks.log('Experiment stopped.', 'info');
    this.dirty = true;
  }

  private score(r: Run): number {
    const n = r.rewards.length;
    const k = Math.min(100, n);
    if (!k) return -Infinity;
    let s = 0;
    for (let i = n - k; i < n; i++) s += r.rewards[i];
    return s / k;
  }

  private successRate(r: Run): number {
    const n = r.success.length;
    const k = Math.min(100, n);
    if (!k) return NaN;
    let s = 0;
    for (let i = n - k; i < n; i++) s += r.success[i] ? 1 : 0;
    return s / k;
  }

  render(visible: boolean): void {
    if (!visible) return;
    if (this.dirty) {
      const now = performance.now();
      // the table is rebuilt at most ~3× per second; stay dirty until it has been
      if (now - this.lastTable > 350) {
        this.dirty = false;
        this.lastTable = now;
        this.chart.smoothing = 0.95;
        this.chart.setSeries(this.runs.map((r) => ({ name: `${r.value}`, color: r.color, xs: r.xs, ys: r.rewards, width: 1.75 })));
        this.renderTable();
      }
    }
    this.chart.draw();
  }

  private renderTable(): void {
    clear(this.table);
    if (!this.runs.length) {
      this.table.append(h('div', { class: 'note' }, 'No experiment yet. Pick a setting and press Run experiment.'));
      return;
    }
    const done = this.runs.filter((r) => r.rewards.length);
    const best = done.length ? done.reduce((a, b) => (this.score(b) > this.score(a) ? b : a)) : null;
    const tbl = h('table', { class: 'exp-table' }, h('tr', null, h('th', null, this.param.label), h('th', null, 'progress'), h('th', null, 'reward'), h('th', null, 'success'), h('th', null, 'time'), h('th', null, '')));
    for (const r of this.runs) {
      const sr = this.successRate(r);
      const prog = Math.min(1, r.steps / this.budget);
      tbl.append(
        h(
          'tr',
          { class: r === best && r.status === 'done' ? 'best' : '' },
          h('td', null, h('span', { class: 'sw', style: { background: r.color } }), r.value),
          h('td', { style: { width: '22%' } }, r.status === 'error' ? h('span', { title: r.error }, 'error') : r.status === 'queued' ? 'queued' : r.status === 'stopped' ? 'stopped' : h('div', { class: 'progress', title: `${fmtSteps(r.steps)} steps` }, h('i', { style: { width: `${prog * 100}%`, background: r.status === 'done' ? 'var(--good)' : '' } }))),
          h('td', null, Number.isFinite(this.score(r)) ? fmt(this.score(r), 2) : '–'),
          h('td', null, Number.isFinite(sr) ? `${(sr * 100).toFixed(0)}%` : '–'),
          h('td', null, r.seconds ? fmtTime(r.seconds) : '–'),
          h('td', null, r.snap ? btn({ icon: 'eye', cls: 'xs ghost icon', title: 'Watch this brain in the 3D view', onclick: () => this.hooks.watch(r.snap!, `${this.param.label} = ${r.value}`) }) : null),
        ),
      );
    }
    this.table.append(tbl, h('div', { class: 'note' }, 'Reward and success are averaged over each run\'s last 100 episodes. Click the eye to watch a run\'s brain.'));
  }
}
