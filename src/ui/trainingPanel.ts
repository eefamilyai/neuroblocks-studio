/** 📈 Training tab: is it learning? Curves, reward breakdown, checkpoints, evaluation, algorithm health. */
import type { EpisodeRecord, MetricPoint } from '../ml/algos/common.ts';
import type { EvalResult } from '../ml/algos/evaluate.ts';
import { LineChart } from './charts.ts';
import type { Series } from './charts.ts';
import { btn, clear, fmt, fmtSteps, h, section, setText } from './dom.ts';

const METRICS: Record<string, [string, string, string, string][]> = {
  // key, title, colour, help
  ppo: [
    ['policyLoss', 'Policy loss', '#9b7bff', 'The clipped PPO objective (negated). It hovers around 0 and is noisy. That is normal: a policy loss going "down" does NOT mean the robot is improving. Watch the reward instead.'],
    ['valueLoss', 'Value loss', '#f472b6', "How wrong the critic's predictions of future reward are (mean squared error). Usually spikes when the robot discovers something new, then falls."],
    ['entropy', 'Entropy', '#5ab4ff', 'How random the policy is. High = exploring, low = confident. It should slowly decrease. If it collapses to ~0 early, the robot stopped exploring (raise the entropy bonus).'],
    ['approxKL', 'Approx. KL', '#fb923c', 'How much the policy changed in one update. PPO keeps this small (≈0.01). Big values mean the learning rate may be too high (we stop the update early above 0.05).'],
    ['clipFraction', 'Clip fraction', '#f2b84b', 'Fraction of samples where the ratio π_new/π_old hit the clip range ±ε. Around 0.1–0.2 is healthy.'],
    ['explainedVariance', 'Explained variance', '#3ccf8e', '1 = the critic predicts returns perfectly, 0 = no better than guessing the average, negative = worse. It should rise toward 1 as learning progresses.'],
    ['gradNormActor', 'Gradient norm', '#94a3b8', 'Size of the gradient vector before clipping. Sudden huge spikes = instability.'],
    ['lr', 'Learning rate', '#a3e635', 'The step size of every update. Constant unless you added a learning-rate schedule.'],
    ['actionStd', 'Action std', '#c4b5fd', 'For smooth controls: the standard deviation of the exploration noise. It shrinks as the robot becomes more precise.'],
  ],
  dqn: [
    ['loss', 'TD loss (Huber)', '#f472b6', 'How far Q(s,a) is from the Bellman target r + γ·max Q(s′,a′). It does not have to go to zero: targets keep moving as the agent improves.'],
    ['meanQ', 'Mean Q value', '#9b7bff', 'Average predicted value of the chosen actions. Should rise as the robot gets better, and stabilise. Exploding Q values = divergence.'],
    ['epsilon', 'Exploration ε', '#f2b84b', 'Probability of taking a random action. Decays from ε-start to ε-end.'],
    ['tdError', 'Mean |TD error|', '#fb923c', 'Average absolute surprise per transition.'],
    ['gradNorm', 'Gradient norm', '#94a3b8', 'Size of the gradient before clipping.'],
    ['lr', 'Learning rate', '#a3e635', 'The step size of every update.'],
  ],
  reinforce: [
    ['policyLoss', 'Policy loss', '#9b7bff', '−mean(log π · advantage). Noisy by nature: REINFORCE estimates gradients from whole episodes.'],
    ['valueLoss', 'Baseline loss', '#f472b6', 'How well the baseline (critic) predicts returns.'],
    ['entropy', 'Entropy', '#5ab4ff', 'Randomness of the policy.'],
    ['meanReturn', 'Mean return', '#3ccf8e', 'Average G_t in the batch: the learning signal.'],
    ['gradNorm', 'Gradient norm', '#94a3b8', 'Size of the gradient before clipping.'],
    ['lr', 'Learning rate', '#a3e635', 'The step size of every update.'],
  ],
  evolution: [['mutationSigma', 'Mutation σ', '#f2b84b', 'Size of the random changes applied to children. Decays slowly so search becomes finer.']],
  bc: [
    ['loss', 'Imitation loss', '#f472b6', 'How surprised the network is by YOUR actions (cross-entropy). Lower = better copy.'],
    ['accuracy', 'Accuracy', '#3ccf8e', 'Fraction of your recorded actions the network now picks itself (1 = perfect copy). 100% often means overfitting.'],
  ],
};

const PART_COLORS = ['#3ccf8e', '#f472b6', '#5ab4ff', '#f2b84b', '#9b7bff', '#fb923c', '#2dd4bf', '#e879f9'];
const PREV_COLORS = ['#94a3b8', '#f9a8d4', '#93c5fd', '#fcd34d'];

export interface CheckpointInfo {
  steps: number;
  updates: number;
  meanReward: number;
  success: number;
  best?: boolean;
}

export interface TrainingPanelHooks {
  onWatchCheckpoint(index: number | null): void;
  onResumeCheckpoint(index: number): void;
  onEvaluate(episodes: number, deterministic: boolean): void;
}

export class TrainingPanel {
  readonly el: HTMLElement;
  private rewardChart: LineChart;
  private successChart: LineChart;
  private lengthChart: LineChart;
  private partsChart: LineChart;
  private fitnessChart: LineChart;
  private metricCharts = new Map<string, LineChart>();
  private metricHost: HTMLElement;
  private partsSection: HTMLElement;
  private evoSection: HTMLElement;
  private summary: HTMLElement;
  private kpi: Record<string, HTMLElement> = {};
  private fitnessHist: HTMLElement;
  /** every finished episode (for CSV export and milestones) */
  episodes: EpisodeRecord[] = [];
  metrics: MetricPoint[] = [];
  algo = 'ppo';
  // column arrays for the charts (appended, so smoothing is incremental)
  private xs: number[] = [];
  private rewards: number[] = [];
  private lengths: number[] = [];
  private succ: number[] = [];
  private succWin = 0;
  private parts = new Map<string, number[]>();
  private mx: number[] = [];
  private mcols = new Map<string, number[]>();
  private prevRuns: { label: string; xs: number[]; ys: number[]; color: string }[] = [];
  private dirty = true;
  private smoothing = 0.9;
  private hooks: TrainingPanelHooks;
  // checkpoints
  private cps: CheckpointInfo[] = [];
  private cpSlider: HTMLInputElement;
  private cpMarks: HTMLElement;
  private cpLabel: HTMLElement;
  private cpWatching: number | null = null;
  private cpSection: HTMLElement;
  // evaluation
  private evalBody: HTMLElement;
  private evalBtn: HTMLButtonElement;
  private evalEpisodes = 100;
  private evalDet = true;
  lr = NaN;

  constructor(hooks: TrainingPanelHooks) {
    this.hooks = hooks;
    this.rewardChart = new LineChart('Episode reward', 'Total reward per episode (dots) and its moving average (line). THE number to watch: if it goes up, your robot is learning. Dashed lines are earlier runs, for comparison.', { tall: true });
    this.successChart = new LineChart('Success rate', 'Fraction of the last 50 episodes that ended with success.', { yRange: [0, 1] });
    this.lengthChart = new LineChart('Episode length', 'Steps per episode. For "reach the goal" tasks this should drop (it gets there faster); for "survive" tasks it should rise.');
    this.partsChart = new LineChart('Reward per episode, by source', 'Where the reward comes from: each line is one kind of reward (from the reward blocks, or api.reward(x, "label") in code), averaged per episode. Great for spotting reward hacking: a shaping reward growing while the real goal does not.', { tall: true });
    this.fitnessChart = new LineChart('Fitness per generation', 'Best / mean / worst total reward in each generation of the population.', { tall: true });
    this.fitnessHist = h('div', { class: 'hist' });
    this.metricHost = h('div', { class: 'grid2' });
    this.summary = h('div', { class: 'kv' });
    const kpis = h('div', { class: 'kpis' });
    for (const [k, l] of [
      ['reward', 'Reward'],
      ['success', 'Success'],
      ['episodes', 'Episodes'],
      ['updates', 'Updates'],
    ]) {
      this.kpi[k] = h('b', null, '–');
      kpis.append(h('div', { class: 'kpi', title: k === 'reward' || k === 'success' ? 'over the last 100 episodes' : '' }, h('small', null, l), this.kpi[k]));
    }
    const smooth = h('select', { class: 'sel sm', title: 'Smoothing of the curves (moving average)' }) as HTMLSelectElement;
    for (const [v, l] of [
      ['0', 'raw'],
      ['0.6', 'smooth: low'],
      ['0.9', 'smooth: medium'],
      ['0.97', 'smooth: high'],
    ])
      smooth.append(h('option', { value: v, selected: Number(v) === this.smoothing }, l));
    smooth.addEventListener('change', () => {
      this.smoothing = Number(smooth.value);
      this.dirty = true;
    });

    // checkpoints
    this.cpSlider = h('input', { type: 'range', min: '0', max: '0', step: '1', value: '0' }) as HTMLInputElement;
    this.cpMarks = h('div', { class: 'tl-marks' });
    this.cpLabel = h('div', { class: 'small dim' }, 'Brains are saved as training goes. Drag to watch an earlier one.');
    this.cpSlider.addEventListener('input', () => {
      const i = Number(this.cpSlider.value);
      const latest = i >= this.cps.length - 1;
      this.cpWatching = latest ? null : i;
      this.hooks.onWatchCheckpoint(this.cpWatching);
      this.renderCpLabel();
    });
    this.cpSection = section('Brain timeline', 'clock', [
      h('div', { class: 'card' }, this.cpLabel, h('div', { class: 'timeline' }, this.cpSlider, this.cpMarks), h(
        'div',
        { class: 'row', style: { marginTop: '6px' } },
        btn({ icon: 'star', label: 'Best', cls: 'xs', title: 'Watch the best brain so far', onclick: () => this.watchBest() }),
        btn({ icon: 'circle-play', label: 'Latest', cls: 'xs', title: 'Watch the newest brain', onclick: () => this.watchLatest() }),
        h('div', { class: 'spacer' }),
        btn({
          icon: 'reset',
          label: 'Train from here',
          cls: 'xs',
          title: 'Throw away what came after this checkpoint and continue training from it',
          onclick: () => {
            if (this.cpWatching !== null) this.hooks.onResumeCheckpoint(this.cpWatching);
          },
        }),
      )),
    ]);

    // evaluation
    this.evalBody = h('div', { class: 'note' }, 'Test the current brain without learning or exploring: like a test set in supervised learning.');
    this.evalBtn = btn({ icon: 'checks', label: 'Evaluate', cls: 'xs primary', title: 'Run the brain for N episodes and measure it', onclick: () => this.hooks.onEvaluate(this.evalEpisodes, this.evalDet) });
    const evSel = h('select', { class: 'sel sm' }) as HTMLSelectElement;
    for (const n of [20, 50, 100, 200, 500]) evSel.append(h('option', { value: String(n), selected: n === this.evalEpisodes }, `${n} episodes`));
    evSel.addEventListener('change', () => (this.evalEpisodes = Number(evSel.value)));
    const det = h('input', { type: 'checkbox', checked: this.evalDet }) as HTMLInputElement;
    det.addEventListener('change', () => (this.evalDet = det.checked));
    const evalSection = section('Evaluate', 'checks', [h('div', { class: 'card' }, h('div', { class: 'row', style: { flexWrap: 'wrap' } }, evSel, h('label', { class: 'row small dim', title: 'Always take the most likely action (no exploration noise)' }, det, 'best action only'), h('div', { class: 'spacer' }), this.evalBtn), this.evalBody)]);

    this.partsSection = section('Where does the reward come from?', 'zap', [this.partsChart.el]);
    this.evoSection = section('Evolution', 'shuffle', [this.fitnessChart.el, h('div', { class: 'note' }, 'Fitness of every brain in the last generation (sorted):'), this.fitnessHist]);
    this.el = h(
      'div',
      null,
      kpis,
      section('Is it learning?', 'chart', [this.rewardChart.el, h('div', { class: 'grid2', style: { marginTop: '8px' } }, this.successChart.el, this.lengthChart.el)], [smooth]),
      this.partsSection,
      this.cpSection,
      evalSection,
      this.evoSection,
      section('Algorithm health', 'activity', [this.metricHost]),
      section('Summary', 'info', [h('div', { class: 'card' }, this.summary)]),
    );
    this.partsSection.style.display = 'none';
    this.cpSection.style.display = 'none';
    this.setAlgo('ppo');
  }

  setAlgo(algo: string): void {
    if (algo === this.algo && this.metricCharts.size) return;
    this.algo = algo;
    clear(this.metricHost);
    this.metricCharts.clear();
    for (const [key, title, , help] of METRICS[algo] ?? []) {
      const c = new LineChart(title, help);
      this.metricCharts.set(key, c);
      this.metricHost.append(c.el);
    }
    this.evoSection.style.display = algo === 'evolution' ? '' : 'none';
    this.dirty = true;
  }

  /** Start a new run; the old reward curve is kept (dashed) for comparison. */
  newRun(label: string): void {
    if (this.xs.length > 5) {
      this.prevRuns.unshift({ label, xs: this.xs, ys: this.rewards, color: PREV_COLORS[this.prevRuns.length % PREV_COLORS.length] });
      this.prevRuns = this.prevRuns.slice(0, 4);
    }
    this.episodes = [];
    this.metrics = [];
    this.xs = [];
    this.rewards = [];
    this.lengths = [];
    this.succ = [];
    this.succWin = 0;
    this.parts.clear();
    this.mx = [];
    this.mcols.clear();
    this.partsSection.style.display = 'none';
    this.setCheckpoints([]);
    this.dirty = true;
  }

  clearComparisons(): void {
    this.prevRuns = [];
    this.dirty = true;
  }

  add(episodes: EpisodeRecord[], metrics: MetricPoint[]): void {
    if (!episodes.length && !metrics.length) return;
    for (const e of episodes) {
      const i = this.xs.length;
      this.episodes.push(e);
      this.xs.push(e.step);
      this.rewards.push(e.reward);
      this.lengths.push(e.length);
      this.succWin += e.success ? 1 : 0;
      if (i >= 50) this.succWin -= this.episodes[this.episodes.length - 51].success ? 1 : 0;
      this.succ.push(this.succWin / Math.min(i + 1, 50));
      if (e.parts)
        for (const [k, v] of Object.entries(e.parts)) {
          let col = this.parts.get(k);
          if (!col) {
            if (this.parts.size >= 12) continue;
            col = new Array(i).fill(0);
            this.parts.set(k, col);
          }
          col.push(v);
        }
      for (const col of this.parts.values()) if (col.length < i + 1) col.push(0);
    }
    for (const m of metrics) {
      const i = this.mx.length;
      this.metrics.push(m);
      this.mx.push(m.step);
      for (const [k, v] of Object.entries(m)) {
        if (k === 'step') continue;
        let col = this.mcols.get(k);
        if (!col) this.mcols.set(k, (col = new Array(i).fill(NaN)));
        col.push(v);
      }
      for (const col of this.mcols.values()) if (col.length < i + 1) col.push(NaN);
    }
    if (this.xs.length > 30000) this.trim();
    if (this.parts.size && this.partsSection.style.display === 'none' && (this.parts.size > 1 || this.episodes.length > 3)) this.partsSection.style.display = '';
    this.dirty = true;
  }

  private trim(): void {
    const k = 10000;
    for (const a of [this.xs, this.rewards, this.lengths, this.succ, this.episodes, ...this.parts.values()]) a.splice(0, k);
  }

  setFitness(f: number[] | undefined): void {
    if (!f || !f.length) return;
    const s = f.slice().sort((a, b) => b - a);
    const max = s[0];
    const min = s[s.length - 1];
    clear(this.fitnessHist);
    for (const v of s) this.fitnessHist.append(h('i', { style: { height: `${8 + ((v - min) / (max - min || 1)) * 92}%` }, title: fmt(v, 2) }));
  }

  // ---------------------------------------------------------------- checkpoints
  setCheckpoints(cps: CheckpointInfo[]): void {
    const wasLatest = this.cpWatching === null;
    this.cps = cps;
    this.cpSection.style.display = cps.length > 1 ? '' : 'none';
    this.cpSlider.max = String(Math.max(0, cps.length - 1));
    if (wasLatest) this.cpSlider.value = this.cpSlider.max;
    clear(this.cpMarks);
    const n = cps.length;
    cps.forEach((c, i) => {
      if (n > 1) this.cpMarks.append(h('i', { class: c.best ? 'best' : '', style: { left: `${(i / (n - 1)) * 100}%` }, title: `${fmtSteps(c.steps)} steps` }));
    });
    this.renderCpLabel();
  }

  private renderCpLabel(): void {
    const i = this.cpWatching;
    const c = i === null ? this.cps[this.cps.length - 1] : this.cps[i];
    if (!c) return;
    const tag = i === null ? 'Watching the newest brain' : `Watching checkpoint ${i + 1}/${this.cps.length}`;
    setText(this.cpLabel, `${tag} · ${fmtSteps(c.steps)} steps · reward ${fmt(c.meanReward, 2)} · success ${Number.isFinite(c.success) ? (c.success * 100).toFixed(0) + '%' : '–'}${c.best ? ' · ★ best' : ''}`);
  }

  private watchBest(): void {
    const i = this.cps.findIndex((c) => c.best);
    if (i < 0) return;
    this.cpSlider.value = String(i);
    this.cpSlider.dispatchEvent(new Event('input'));
  }

  watchLatest(): void {
    this.cpSlider.value = this.cpSlider.max;
    this.cpWatching = null;
    this.hooks.onWatchCheckpoint(null);
    this.renderCpLabel();
  }

  get watchingCheckpoint(): number | null {
    return this.cpWatching;
  }

  // ---------------------------------------------------------------- evaluation
  evalProgress(done: number, total: number): void {
    this.evalBtn.classList.add('disabled');
    clear(this.evalBody);
    this.evalBody.append(h('div', { class: 'row small dim', style: { marginTop: '8px' } }, `Evaluating… ${done}/${total} episodes`), h('div', { class: 'progress', style: { marginTop: '6px' } }, h('i', { style: { width: `${(done / total) * 100}%` } })));
  }

  showEval(r: EvalResult): void {
    this.evalBtn.classList.remove('disabled');
    clear(this.evalBody);
    const bins = new Array<number>(20).fill(0);
    const min = Math.min(...r.rewards);
    const max = Math.max(...r.rewards);
    for (const v of r.rewards) bins[Math.min(19, Math.floor(((v - min) / (max - min || 1)) * 20))]++;
    const peak = Math.max(...bins);
    const parts = Object.entries(r.parts).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
    const pmax = Math.max(1e-9, ...parts.map(([, v]) => Math.abs(v)));
    this.evalBody.append(
      h(
        'div',
        { class: 'kv', style: { marginTop: '10px' } },
        h('span', null, 'Success rate'),
        h('b', { style: { color: r.successRate > 0.8 ? 'var(--good)' : '' } }, `${(r.successRate * 100).toFixed(0)}%`),
        h('span', null, 'Reward (mean ± std)'),
        h('b', null, `${fmt(r.meanReward, 2)} ± ${fmt(r.stdReward, 2)}`),
        h('span', null, 'Episode length'),
        h('b', null, fmt(r.meanLength, 0)),
        h('span', null, 'Brain'),
        h('b', null, `${fmtSteps(r.brainSteps)} steps · ${r.deterministic ? 'best action' : 'sampled'}`),
      ),
      h('div', { class: 'hist', title: `reward histogram: ${fmt(min, 2)} … ${fmt(max, 2)}` }, ...bins.map((b) => h('i', { style: { height: `${(b / peak) * 100}%` } }))),
      h('div', { class: 'row tiny muted', style: { justifyContent: 'space-between' } }, h('span', null, fmt(min, 2)), h('span', null, `reward of ${r.episodes} test episodes (${r.seconds.toFixed(1)} s)`), h('span', null, fmt(max, 2))),
    );
    if (parts.length > 1)
      this.evalBody.append(
        h(
          'div',
          { class: 'bars', style: { marginTop: '10px' } },
          ...parts.map(([k, v], i) =>
            h('div', { class: 'bar-row' }, h('span', { class: 'name', title: k }, k), h('div', { class: 'cbar center' }, h('i', { style: { left: v >= 0 ? '50%' : `${50 - (Math.abs(v) / pmax) * 50}%`, width: `${(Math.abs(v) / pmax) * 50}%`, background: PART_COLORS[i % PART_COLORS.length] } })), h('span', { class: 'val' }, fmt(v, 2))),
          ),
        ),
      );
  }

  evalFailed(): void {
    this.evalBtn.classList.remove('disabled');
  }

  // ---------------------------------------------------------------- render
  /** Redraw if needed (called from the animation loop; cheap when nothing changed). */
  render(visible: boolean): void {
    if (!visible) return;
    if (this.dirty) {
      this.dirty = false;
      this.updateCharts();
    }
    this.rewardChart.draw();
    this.successChart.draw();
    this.lengthChart.draw();
    this.partsChart.draw();
    this.fitnessChart.draw();
    for (const c of this.metricCharts.values()) c.draw();
  }

  private updateCharts(): void {
    const series: Series[] = this.prevRuns.map((r, i) => ({ name: i === 0 ? 'earlier runs' : '_prev' + i, color: r.color, xs: r.xs, ys: r.ys, dashed: true, width: 1.25 }));
    series.push({ name: 'reward', color: '#3ccf8e', xs: this.xs, ys: this.rewards, dots: true, width: 2 });
    this.rewardChart.smoothing = this.smoothing;
    this.rewardChart.setSeries(series);
    this.successChart.setSeries([{ name: '', color: '#f2b84b', xs: this.xs, ys: this.succ, raw: true, fill: true }]);
    this.lengthChart.smoothing = this.smoothing;
    this.lengthChart.setSeries([{ name: '', color: '#5ab4ff', xs: this.xs, ys: this.lengths }]);
    if (this.parts.size) {
      const sums = [...this.parts.entries()].map(([k, col]) => [k, col.slice(-200).reduce((a, b) => a + Math.abs(b), 0)] as const).sort((a, b) => b[1] - a[1]);
      this.partsChart.smoothing = Math.max(0.9, this.smoothing);
      this.partsChart.setSeries(sums.slice(0, 8).map(([k], i) => ({ name: k, color: PART_COLORS[i % PART_COLORS.length], xs: this.xs, ys: this.parts.get(k)! })));
    }
    for (const [key, , color] of METRICS[this.algo] ?? []) {
      const c = this.metricCharts.get(key);
      if (!c) continue;
      c.smoothing = key === 'epsilon' || key === 'mutationSigma' || key === 'lr' ? 0 : Math.min(this.smoothing, 0.8);
      c.setSeries([{ name: '', color, xs: this.mx, ys: this.mcols.get(key) ?? [] }]);
    }
    if (this.algo === 'evolution') {
      this.fitnessChart.smoothing = 0;
      this.fitnessChart.setSeries([
        { name: 'best', color: '#3ccf8e', xs: this.mx, ys: this.mcols.get('bestFitness') ?? [], raw: true },
        { name: 'mean', color: '#5ab4ff', xs: this.mx, ys: this.mcols.get('meanFitness') ?? [], raw: true },
        { name: 'worst', color: '#f26b7d', xs: this.mx, ys: this.mcols.get('worstFitness') ?? [], raw: true },
      ]);
    }
    // KPIs + summary
    const n = this.rewards.length;
    const k = Math.min(100, n);
    let mean = 0;
    let succ = 0;
    for (let i = n - k; i < n; i++) {
      mean += this.rewards[i];
      succ += this.episodes[i]?.success ? 1 : 0;
    }
    mean /= k || 1;
    const sr = k ? succ / k : NaN;
    setText(this.kpi.reward, k ? fmt(mean, 2) : '–');
    setText(this.kpi.success, k ? `${(sr * 100).toFixed(0)}%` : '–');
    this.kpi.success.style.color = sr >= 0.8 ? 'var(--good)' : '';
    setText(this.kpi.episodes, n.toLocaleString());
    setText(this.kpi.updates, (this.metrics.at(-1)?.update ?? 0).toLocaleString());
    let best = -Infinity;
    for (const r of this.rewards) if (r > best) best = r;
    clear(this.summary);
    const row = (a: string, b: string) => this.summary.append(h('span', null, a), h('b', null, b));
    row('Episodes finished', n.toLocaleString());
    row('Mean reward (last 100)', k ? fmt(mean, 3) : '–');
    row('Best episode reward', Number.isFinite(best) ? fmt(best, 3) : '–');
    row('Success rate (last 100)', k ? `${(sr * 100).toFixed(0)}%` : '–');
    row('Updates to the brain', (this.metrics.at(-1)?.update ?? 0).toLocaleString());
    if (Number.isFinite(this.lr)) row('Current learning rate', fmt(this.lr, 6));
  }

  /** mean reward & success over the last `k` episodes */
  recent(k = 100): { mean: number; success: number; n: number } {
    const n = this.rewards.length;
    const m = Math.min(k, n);
    let s = 0;
    let c = 0;
    for (let i = n - m; i < n; i++) {
      s += this.rewards[i];
      c += this.episodes[i]?.success ? 1 : 0;
    }
    return { mean: m ? s / m : NaN, success: m ? c / m : NaN, n: m };
  }
}

