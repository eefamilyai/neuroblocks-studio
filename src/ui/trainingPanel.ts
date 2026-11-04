/** 📈 Training tab: learning curves and algorithm health metrics. */
import type { EpisodeRecord, MetricPoint } from '../ml/algos/common.ts';
import { LineChart } from './charts.ts';
import type { Series } from './charts.ts';
import { fmt, h } from './dom.ts';

const METRICS: Record<string, [string, string, string, string][]> = {
  // key, title, colour, help
  ppo: [
    ['policyLoss', 'Policy loss', '#8b6cff', 'The clipped PPO objective (negated). It hovers around 0 and is noisy. That is normal: a policy loss going "down" does NOT mean the robot is improving. Watch the reward instead.'],
    ['valueLoss', 'Value loss', '#ff6fae', 'How wrong the critic\'s predictions of future reward are (mean squared error). Usually spikes when the robot discovers something new, then falls.'],
    ['entropy', 'Entropy (randomness)', '#4cc3ff', 'How random the policy is. High = exploring, low = confident. It should slowly decrease. If it collapses to ~0 early, the robot stopped exploring (raise the entropy bonus).'],
    ['approxKL', 'Approx. KL divergence', '#ffa94d', 'How much the policy changed in one update. PPO keeps this small (≈0.01). Big values mean the learning rate may be too high (we stop the update early above 0.05).'],
    ['clipFraction', 'Clip fraction', '#ffd166', 'Fraction of samples where the ratio π_new/π_old hit the clip range ±ε. Around 0.1–0.2 is healthy.'],
    ['explainedVariance', 'Explained variance (critic)', '#2fd182', '1 = the critic predicts returns perfectly, 0 = no better than guessing the average, negative = worse. It should rise toward 1 as learning progresses.'],
    ['gradNormActor', 'Gradient norm (actor)', '#9aa3c7', 'Size of the gradient vector before clipping. Sudden huge spikes = instability.'],
    ['actionStd', 'Action std (continuous)', '#b49bff', 'For smooth controls: the standard deviation of the exploration noise. It shrinks as the robot becomes more precise.'],
  ],
  dqn: [
    ['loss', 'TD loss (Huber)', '#ff6fae', 'How far Q(s,a) is from the Bellman target r + γ·max Q(s′,a′). It does not have to go to zero: targets keep moving as the agent improves.'],
    ['meanQ', 'Mean Q value', '#8b6cff', 'Average predicted value of the chosen actions. Should rise as the robot gets better, and stabilise. Exploding Q values = divergence.'],
    ['epsilon', 'Exploration ε', '#ffd166', 'Probability of taking a random action. Decays from ε-start to ε-end.'],
    ['tdError', 'Mean |TD error|', '#ffa94d', 'Average absolute surprise per transition.'],
    ['gradNorm', 'Gradient norm', '#9aa3c7', 'Size of the gradient before clipping.'],
  ],
  reinforce: [
    ['policyLoss', 'Policy loss', '#8b6cff', '−mean(log π · advantage). Noisy by nature: REINFORCE estimates gradients from whole episodes.'],
    ['valueLoss', 'Baseline loss', '#ff6fae', 'How well the baseline (critic) predicts returns.'],
    ['entropy', 'Entropy', '#4cc3ff', 'Randomness of the policy.'],
    ['meanReturn', 'Mean discounted return', '#2fd182', 'Average G_t in the batch: the learning signal.'],
    ['gradNorm', 'Gradient norm', '#9aa3c7', 'Size of the gradient before clipping.'],
  ],
  evolution: [
    ['mutationSigma', 'Mutation σ', '#ffd166', 'Size of the random changes applied to children. Decays slowly so search becomes finer.'],
  ],
  bc: [
    ['loss', 'Imitation loss (cross-entropy)', '#ff6fae', 'How surprised the network is by YOUR actions. Lower = better copy.'],
    ['accuracy', 'Accuracy', '#2fd182', 'Fraction of your recorded actions the network now picks itself (1 = perfect copy). 100% often means overfitting.'],
  ],
};

export class TrainingPanel {
  readonly el: HTMLElement;
  private rewardChart: LineChart;
  private successChart: LineChart;
  private lengthChart: LineChart;
  private fitnessChart: LineChart;
  private metricCharts = new Map<string, LineChart>();
  private metricHost: HTMLElement;
  private summary: HTMLElement;
  episodes: EpisodeRecord[] = [];
  metrics: MetricPoint[] = [];
  algo = 'ppo';
  private prevRuns: { label: string; xs: number[]; ys: number[]; color: string }[] = [];
  private dirty = true;
  private smoothing = 0.9;
  private fitnessHist: HTMLElement;

  constructor() {
    this.rewardChart = new LineChart('Episode reward', 'Total reward per episode (dots) and its moving average (line). THE number to watch: if it goes up, your robot is learning. Dashed lines are previous runs, for comparison.', { tall: true });
    this.successChart = new LineChart('Success rate', 'Fraction of the last 50 episodes that ended with "success ✅".', { yRange: [0, 1] });
    this.lengthChart = new LineChart('Episode length', 'Steps per episode. For "reach the goal" tasks this should drop (it gets there faster); for "survive" tasks it should rise.');
    this.fitnessChart = new LineChart('Fitness per generation', 'Best / mean / worst total reward in each generation of the population.', { tall: true });
    this.fitnessHist = h('div', { class: 'hist' });
    this.metricHost = h('div', { class: 'grid2' });
    this.summary = h('div', { class: 'kv' });
    const slider = h('input', { type: 'range', min: '0', max: '0.99', step: '0.01', value: String(this.smoothing) }) as HTMLInputElement;
    slider.addEventListener('input', () => {
      this.smoothing = Number(slider.value);
      this.dirty = true;
    });
    this.el = h(
      'div',
      null,
      h('div', { class: 'section' }, h('h3', null, '🏆 Is it learning?'), this.rewardChart.el, h('div', { class: 'slider-row' }, 'smoothing', slider), h('div', { class: 'grid2', style: { marginTop: '8px' } }, this.successChart.el, this.lengthChart.el)),
      h('div', { class: 'section', id: 'evo-section', style: { display: 'none' } }, h('h3', null, '🧬 Evolution'), this.fitnessChart.el, h('div', { class: 'small muted' }, 'Fitness of every brain in the last generation (sorted):'), this.fitnessHist),
      h('div', { class: 'section' }, h('h3', null, '🩺 Algorithm health'), this.metricHost),
      h('div', { class: 'section' }, h('h3', null, '📋 Summary'), h('div', { class: 'card' }, this.summary)),
    );
    this.setAlgo('ppo');
  }

  setAlgo(algo: string): void {
    this.algo = algo;
    this.metricHost.innerHTML = '';
    this.metricCharts.clear();
    for (const [key, title, , help] of METRICS[algo] ?? []) {
      const c = new LineChart(title, help);
      this.metricCharts.set(key, c);
      this.metricHost.append(c.el);
    }
    const evo = this.el.querySelector('#evo-section') as HTMLElement;
    evo.style.display = algo === 'evolution' ? '' : 'none';
    this.dirty = true;
  }

  /** Start a new run; the old reward curve is kept (dashed) for comparison. */
  newRun(label: string): void {
    if (this.episodes.length > 5) {
      const xs = this.episodes.map((e) => e.step);
      const ys = this.episodes.map((e) => e.reward);
      const colors = ['#9aa3c7', '#ff9fbe', '#8fd6ff', '#ffd98a'];
      this.prevRuns.unshift({ label, xs, ys, color: colors[this.prevRuns.length % colors.length] });
      this.prevRuns = this.prevRuns.slice(0, 4);
    }
    this.episodes = [];
    this.metrics = [];
    this.dirty = true;
  }

  clearComparisons(): void {
    this.prevRuns = [];
    this.dirty = true;
  }

  add(episodes: EpisodeRecord[], metrics: MetricPoint[]): void {
    if (!episodes.length && !metrics.length) return;
    this.episodes.push(...episodes);
    this.metrics.push(...metrics);
    if (this.episodes.length > 30000) this.episodes.splice(0, this.episodes.length - 30000);
    this.dirty = true;
  }

  setFitness(f: number[] | undefined): void {
    if (!f || !f.length) return;
    const s = f.slice().sort((a, b) => b - a);
    const max = Math.max(...s);
    const min = Math.min(...s);
    this.fitnessHist.innerHTML = '';
    for (const v of s) this.fitnessHist.append(h('i', { style: { height: `${8 + ((v - min) / (max - min || 1)) * 92}%` }, title: fmt(v, 2) }));
  }

  /** Redraw if needed (call from the animation loop, cheap when nothing changed). */
  render(visible: boolean): void {
    if (!visible || !this.dirty) return;
    this.dirty = false;
    const eps = this.episodes;
    const xs = eps.map((e) => e.step);
    const series: Series[] = this.prevRuns.map((r, i) => ({ name: i === 0 ? 'previous runs' : '_prev' + i, color: r.color, xs: r.xs, ys: r.ys, dashed: true, width: 1.5 }));
    series.push({ name: 'reward', color: '#2fd182', xs, ys: eps.map((e) => e.reward), dots: true, width: 2.5 });
    this.rewardChart.smoothing = this.smoothing;
    this.rewardChart.setSeries(series);
    this.rewardChart.draw();
    const succ: number[] = [];
    let win = 0;
    for (let i = 0; i < eps.length; i++) {
      win += eps[i].success ? 1 : 0;
      if (i >= 50) win -= eps[i - 50].success ? 1 : 0;
      succ.push(win / Math.min(i + 1, 50));
    }
    this.successChart.smoothing = 0;
    this.successChart.setSeries([{ name: '', color: '#ffd166', xs, ys: succ, raw: true }]);
    this.successChart.draw();
    this.lengthChart.smoothing = this.smoothing;
    this.lengthChart.setSeries([{ name: '', color: '#4cc3ff', xs, ys: eps.map((e) => e.length) }]);
    this.lengthChart.draw();
    const mx = this.metrics.map((m) => m.step);
    for (const [key, , color] of METRICS[this.algo] ?? []) {
      const c = this.metricCharts.get(key);
      if (!c) continue;
      c.smoothing = key === 'epsilon' || key === 'mutationSigma' ? 0 : Math.min(this.smoothing, 0.8);
      c.setSeries([{ name: '', color, xs: mx, ys: this.metrics.map((m) => (m[key] === undefined ? NaN : m[key])) }]);
      c.draw();
    }
    if (this.algo === 'evolution') {
      this.fitnessChart.smoothing = 0;
      this.fitnessChart.setSeries([
        { name: 'best', color: '#2fd182', xs: mx, ys: this.metrics.map((m) => m.bestFitness), raw: true },
        { name: 'mean', color: '#4cc3ff', xs: mx, ys: this.metrics.map((m) => m.meanFitness), raw: true },
        { name: 'worst', color: '#ff5d73', xs: mx, ys: this.metrics.map((m) => m.worstFitness), raw: true },
      ]);
      this.fitnessChart.draw();
    }
    // summary
    const last100 = eps.slice(-100);
    const mean = last100.reduce((a, e) => a + e.reward, 0) / Math.max(1, last100.length);
    const best = eps.reduce((a, e) => Math.max(a, e.reward), -Infinity);
    const sr = last100.filter((e) => e.success).length / Math.max(1, last100.length);
    this.summary.innerHTML = '';
    const row = (k: string, v: string) => this.summary.append(h('span', null, k), h('b', null, v));
    row('Episodes finished', eps.length.toLocaleString());
    row('Mean reward (last 100)', last100.length ? fmt(mean, 3) : '–');
    row('Best episode reward', Number.isFinite(best) ? fmt(best, 3) : '–');
    row('Success rate (last 100)', last100.length ? `${(sr * 100).toFixed(0)}%` : '–');
    row('Updates to the brain', (this.metrics.at(-1)?.update ?? 0).toLocaleString());
  }
}
