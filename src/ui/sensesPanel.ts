/** 👁️ Senses tab: the exact numbers going into and coming out of the brain. */
import type { Brain } from '../ml/algos/common.ts';
import type { ObsLayout } from '../sim/env.ts';
import type { Showcase } from '../showcase.ts';
import { LineChart } from './charts.ts';
import { clear, fmt, h } from './dom.ts';

export class SensesPanel {
  readonly el: HTMLElement;
  private obsHost: HTMLElement;
  private actHost: HTMLElement;
  private eventsHost: HTMLElement;
  private valueChart: LineChart;
  private rows: { bar: HTMLElement; val: HTMLElement; sal: HTMLElement }[] = [];
  private layout: ObsLayout | null = null;
  showNormalized = false;
  private events: string[] = [];
  private normBtn: HTMLButtonElement;

  constructor() {
    this.obsHost = h('div');
    this.actHost = h('div', { class: 'card' });
    this.eventsHost = h('div', { class: 'card mono small', style: { maxHeight: '160px', overflowY: 'auto', whiteSpace: 'pre-wrap' } });
    this.valueChart = new LineChart('This episode: reward per step & critic value', 'Green = reward received at each step. Yellow = what the critic predicted (V): how much future reward it expects from here. When V rises, the robot "feels" things are going well.');
    this.normBtn = h('button', { class: 'btn tiny', onclick: () => this.toggleNorm() }, 'show: raw values') as HTMLButtonElement;
    this.el = h(
      'div',
      null,
      h('div', { class: 'section' }, h('h3', null, '🕹️ What the brain decides'), this.actHost),
      h('div', { class: 'section' }, h('h3', null, '📈 Reward & value'), this.valueChart.el),
      h(
        'div',
        { class: 'section' },
        h('h3', null, '👁️ What the robot senses', h('span', { class: 'grow' }), this.normBtn),
        h('div', { class: 'small muted', style: { marginBottom: '6px' } }, 'Every number that goes into the network, right now. The small purple squares show saliency: how much each input would change the decision.'),
        this.obsHost,
      ),
      h('div', { class: 'section' }, h('h3', null, '⚡ Events'), this.eventsHost),
    );
  }

  private toggleNorm(): void {
    this.showNormalized = !this.showNormalized;
    this.normBtn.textContent = this.showNormalized ? 'show: normalised (what the net sees)' : 'show: raw values';
  }

  setLayout(layout: ObsLayout | null): void {
    this.layout = layout;
    clear(this.obsHost);
    this.rows = [];
    if (!layout) return;
    for (const g of layout.groups) {
      const grp = h('div', { class: 'obs-group' }, h('div', { class: 'gname' }, g.name));
      for (let i = 0; i < g.len; i++) {
        const bar = h('i');
        const val = h('span', { class: 'val' });
        const sal = h('span', { class: 'sal' });
        grp.append(h('div', { class: 'obs-row' }, h('span', { class: 'name', title: layout.labels[g.start + i] }, layout.labels[g.start + i], sal), h('div', { class: 'cbar' }, bar), val));
        this.rows.push({ bar, val, sal });
      }
      this.obsHost.append(grp);
    }
  }

  addEvents(evts: string[], reward: number): void {
    for (const e of evts) this.events.push(e);
    if (Math.abs(reward) >= 0.3) this.events.push(`${reward > 0 ? '🍬 +' : '⚡ '}${fmt(reward, 2)} reward`);
    if (this.events.length > 60) this.events.splice(0, this.events.length - 60);
  }

  render(visible: boolean, sc: Showcase): void {
    if (!visible || !sc.env) return;
    const obs = sc.env.obs;
    const brain = sc.brain && sc.brainFits() ? sc.brain : null;
    const nvals = brain && this.showNormalized ? brain.normalize(obs) : null;
    const sal = sc.saliency;
    let smax = 1e-9;
    if (sal) for (const v of sal) smax = Math.max(smax, v);
    for (let i = 0; i < this.rows.length && i < obs.length; i++) {
      const v = nvals ? nvals[i] : obs[i];
      const r = this.rows[i];
      const scale = nvals ? 3 : 1;
      const c = Math.max(-1, Math.min(1, v / scale));
      r.bar.style.left = c >= 0 ? '50%' : `${50 + c * 50}%`;
      r.bar.style.width = `${Math.abs(c) * 50}%`;
      r.bar.style.background = c >= 0 ? '#4cc3ff' : '#ff8c50';
      r.val.textContent = fmt(v, 2);
      if (sal) {
        const s = sal[i] / smax;
        r.sal.style.background = `rgba(180,155,255,${0.1 + s * 0.9})`;
      }
    }
    this.renderActions(brain, sc);
    const xs = sc.rewards.map((_, i) => i);
    this.valueChart.smoothing = 0;
    this.valueChart.setSeries([
      { name: 'reward', color: '#2fd182', xs, ys: sc.rewards, raw: true },
      { name: 'V(s)', color: '#ffd166', xs, ys: sc.values, raw: true },
    ]);
    this.valueChart.draw();
    this.eventsHost.textContent = this.events.slice(-25).reverse().join('\n') || 'Events appear here: touches, rewards, episode ends…';
  }

  private renderActions(brain: Brain | null, sc: Showcase): void {
    const host = this.actHost;
    clear(host);
    const spec = sc.spec;
    if (!spec) return;
    if (sc.mode === 'drive') host.append(h('div', { class: 'small', style: { marginBottom: '6px' } }, '🎮 You are driving (WASD / arrows, Space = jump, F = kick).'));
    else if (!brain) host.append(h('div', { class: 'small muted', style: { marginBottom: '6px' } }, 'No compatible brain yet: the robot acts randomly. Press ▶ Train!'));
    if (brain && brain.snap.kind === 'q' && brain.lastQ) {
      const q = brain.lastQ;
      let best = 0;
      for (let i = 1; i < q.length; i++) if (q[i] > q[best]) best = i;
      const qmin = Math.min(...q);
      const qmax = Math.max(...q);
      host.append(h('div', { class: 'small muted' }, 'Q-value of every possible action combination: the robot picks the highest.'));
      const sp = brain.snap.space;
      for (let j = 0; j < q.length; j++) {
        let idx = j;
        const parts: string[] = [];
        for (let k = sp.branches.length - 1; k >= 0; k--) {
          parts.unshift(sp.optionLabels[k][idx % sp.branches[k]]);
          idx = Math.floor(idx / sp.branches[k]);
        }
        const w = ((q[j] - qmin) / (qmax - qmin || 1)) * 100;
        host.append(
          h(
            'div',
            { class: 'obs-row' },
            h('span', { class: 'name', style: { color: j === best ? '#ffd166' : '' } }, parts.join(' + ')),
            h('div', { class: 'cbar' }, h('i', { style: { left: '0', width: `${w}%`, background: j === best ? '#ffd166' : '#8b6cff' } })),
            h('span', { class: 'val' }, fmt(q[j], 2)),
          ),
        );
      }
      return;
    }
    spec.axes.forEach((ax, k) => {
      const row = h('div', { class: 'act-row', style: { marginBottom: '4px' } }, h('span', { class: 'lbl' }, ax.axis));
      const opts = h('div', { class: 'act-opts' });
      if (spec.actionMode === 'discrete') {
        const probs = brain?.lastProbs[k];
        ax.labels.forEach((lab, j) => {
          const p = probs ? probs[j] : 1 / ax.labels.length;
          const chosen = sc.action[k] === j;
          opts.append(h('div', { class: 'act-opt' + (chosen ? ' chosen' : '') }, h('i', { style: { width: `${p * 100}%` } }), h('span', null, `${lab} ${(p * 100).toFixed(0)}%`)));
        });
      } else {
        const mean = brain ? brain.lastOut[k] : 0;
        const std = brain?.logStd ? Math.exp(brain.logStd[k]) : 0;
        const v = sc.action[k];
        opts.append(h('div', { class: 'act-opt' }, h('i', { style: { width: `${((Math.max(-1, Math.min(1, v)) + 1) / 2) * 100}%` } }), h('span', null, `value ${fmt(v, 2)} (μ=${fmt(mean, 2)}, σ=${fmt(std, 2)})`)));
      }
      row.append(opts);
      host.append(row);
    });
  }
}
