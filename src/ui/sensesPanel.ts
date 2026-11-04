/** 👁️ Senses tab: the exact numbers going into and coming out of the brain (updated in place, no DOM churn). */
import type { Brain } from '../ml/algos/common.ts';
import type { ObsLayout } from '../sim/env.ts';
import type { Showcase } from '../showcase.ts';
import { LineChart } from './charts.ts';
import { clear, fmt, h, section, seg, setText } from './dom.ts';

interface ObsRow {
  bar: HTMLElement;
  val: HTMLElement;
  sal: HTMLElement;
  last: number;
}

export class SensesPanel {
  readonly el: HTMLElement;
  private obsHost: HTMLElement;
  private actHost: HTMLElement;
  private actNote: HTMLElement;
  private eventsHost: HTMLElement;
  private valueChart: LineChart;
  private rows: ObsRow[] = [];
  private layout: ObsLayout | null = null;
  showNormalized = false;
  private events: string[] = [];
  private eventsDirty = true;
  // decision display, rebuilt only when its shape changes
  private actKey = '';
  private actCells: { fill: HTMLElement; cell: HTMLElement; text: HTMLElement | null }[][] = [];
  private qRows: { bar: HTMLElement; val: HTMLElement; name: HTMLElement }[] = [];
  private lastEpisodeLen = -1;

  constructor() {
    this.obsHost = h('div');
    this.actHost = h('div', { class: 'act-card' });
    this.actNote = h('div', { class: 'note' });
    this.eventsHost = h('div', { class: 'card events-log' });
    this.valueChart = new LineChart('This episode: reward & critic value', "Green = reward received at each step. Yellow = what the critic predicted (V): how much future reward it expects from here. When V rises, the robot \"feels\" things are going well.");
    this.valueChart.xLabel = (x) => String(Math.round(x));
    const norm = seg(
      [
        { value: 'raw', label: 'raw' },
        { value: 'norm', label: 'normalised', title: 'What the network actually sees, after normalisation' },
      ],
      (v) => {
        this.showNormalized = v === 'norm';
        norm.set(v);
      },
      'sm',
    );
    norm.set('raw');
    this.el = h(
      'div',
      null,
      section('What the brain decides', 'brain', [h('div', { class: 'card' }, this.actHost, this.actNote)]),
      section('Reward & value', 'activity', [this.valueChart.el]),
      section('What the robot senses', 'senses', [h('div', { class: 'note', style: { margin: '0 0 8px' } }, 'Every number that goes into the network, right now. The small purple squares show saliency: how much each input would change the decision.'), this.obsHost], [norm.el]),
      section('Events', 'zap', [this.eventsHost]),
    );
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
        grp.append(h('div', { class: 'obs-row' }, h('span', { class: 'name', title: layout.labels[g.start + i] }, sal, layout.labels[g.start + i]), h('div', { class: 'cbar center' }, bar), val));
        this.rows.push({ bar, val, sal, last: NaN });
      }
      this.obsHost.append(grp);
    }
    this.actKey = '';
  }

  addEvents(evts: string[], reward: number): void {
    for (const e of evts) this.events.push(e);
    if (Math.abs(reward) >= 0.3) this.events.push(`${reward > 0 ? '+' : ''}${fmt(reward, 2)} reward`);
    if (this.events.length > 60) this.events.splice(0, this.events.length - 60);
    if (evts.length || Math.abs(reward) >= 0.3) this.eventsDirty = true;
  }

  render(visible: boolean, sc: Showcase): void {
    if (!visible || !sc.env) return;
    const obs = sc.env.obs;
    const brain = sc.brain && sc.brainFits() ? sc.brain : null;
    const nvals = brain && this.showNormalized ? brain.normalize(obs) : null;
    const sal = sc.saliency;
    let smax = 1e-9;
    if (sal) for (let i = 0; i < sal.length; i++) if (sal[i] > smax) smax = sal[i];
    for (let i = 0; i < this.rows.length && i < obs.length; i++) {
      const v = nvals ? nvals[i] : obs[i];
      const r = this.rows[i];
      if (v !== r.last) {
        r.last = v;
        const c = Math.max(-1, Math.min(1, v / (nvals ? 3 : 1)));
        const st = r.bar.style;
        st.left = c >= 0 ? '50%' : `${50 + c * 50}%`;
        st.width = `${Math.abs(c) * 50}%`;
        st.background = c >= 0 ? '#5ab4ff' : '#fb923c';
        setText(r.val, fmt(v, 2));
      }
      if (sal) r.sal.style.background = `rgba(155,123,255,${(0.1 + (sal[i] / smax) * 0.9).toFixed(2)})`;
    }
    this.renderActions(brain, sc);
    if (sc.rewards.length !== this.lastEpisodeLen) {
      this.lastEpisodeLen = sc.rewards.length;
      const xs = sc.rewards.map((_, i) => i);
      this.valueChart.smoothing = 0;
      this.valueChart.setSeries([
        { name: 'reward', color: '#3ccf8e', xs, ys: sc.rewards, raw: true },
        { name: 'V(s)', color: '#f2b84b', xs, ys: sc.values, raw: true },
      ]);
    }
    this.valueChart.draw();
    if (this.eventsDirty) {
      this.eventsDirty = false;
      this.eventsHost.textContent = this.events.slice(-25).reverse().join('\n') || 'Events appear here: touches, rewards, episode ends…';
    }
  }

  private renderActions(brain: Brain | null, sc: Showcase): void {
    const spec = sc.spec;
    if (!spec) return;
    const q = brain && brain.snap.kind === 'q' ? brain.lastQ : null;
    const key = `${q ? 'q' + q.length : spec.actionMode}|${spec.axes.map((a) => a.labels.join(',')).join(';')}`;
    if (key !== this.actKey) this.buildActions(key, brain, sc);
    setText(this.actNote, sc.mode === 'drive' ? 'You are driving (WASD / arrows, Space = jump, F = kick).' : !brain ? 'No compatible brain yet: the robot acts randomly. Press Train!' : q ? 'Q-value of every action combination: the robot picks the highest.' : spec.actionMode === 'discrete' ? 'Probability of each choice; the outlined one was picked.' : 'Mean μ and exploration noise σ of each control.');
    if (q) {
      let best = 0;
      let qmin = Infinity;
      let qmax = -Infinity;
      for (let i = 0; i < q.length; i++) {
        if (q[i] > q[best]) best = i;
        qmin = Math.min(qmin, q[i]);
        qmax = Math.max(qmax, q[i]);
      }
      this.qRows.forEach((r, j) => {
        r.bar.style.width = `${((q[j] - qmin) / (qmax - qmin || 1)) * 100}%`;
        r.bar.style.background = j === best ? '#f2b84b' : '#9b7bff';
        r.name.style.color = j === best ? 'var(--warn)' : '';
        setText(r.val, fmt(q[j], 2));
      });
      return;
    }
    spec.axes.forEach((ax, k) => {
      const cells = this.actCells[k];
      if (!cells) return;
      if (spec.actionMode === 'discrete') {
        const probs = brain?.lastProbs[k];
        cells.forEach((c, j) => {
          const p = probs ? probs[j] : sc.action[k] === j ? 1 : 0;
          c.fill.style.width = `${(p * 100).toFixed(1)}%`;
          c.cell.classList.toggle('chosen', sc.action[k] === j);
          if (c.text) setText(c.text, `${ax.labels[j]} ${(p * 100).toFixed(0)}%`);
        });
      } else {
        const v = sc.action[k];
        const c = cells[0];
        c.fill.style.width = `${((Math.max(-1, Math.min(1, v)) + 1) / 2) * 100}%`;
        const mean = brain ? brain.lastOut[k] : 0;
        const std = brain?.logStd ? Math.exp(brain.logStd[k]) : 0;
        if (c.text) setText(c.text, `${fmt(v, 2)}  (μ ${fmt(mean, 2)}, σ ${fmt(std, 2)})`);
      }
    });
  }

  private buildActions(key: string, brain: Brain | null, sc: Showcase): void {
    this.actKey = key;
    clear(this.actHost);
    this.actCells = [];
    this.qRows = [];
    const spec = sc.spec!;
    if (brain && brain.snap.kind === 'q' && brain.lastQ) {
      const sp = brain.snap.space;
      const bars = h('div', { class: 'bars' });
      for (let j = 0; j < brain.lastQ.length; j++) {
        let idx = j;
        const parts: string[] = [];
        for (let k = sp.branches.length - 1; k >= 0; k--) {
          parts.unshift(sp.optionLabels[k][idx % sp.branches[k]]);
          idx = Math.floor(idx / sp.branches[k]);
        }
        const name = h('span', { class: 'name', title: parts.join(' + ') }, parts.join(' + '));
        const bar = h('i', { style: { left: '0' } });
        const val = h('span', { class: 'val' });
        bars.append(h('div', { class: 'bar-row' }, name, h('div', { class: 'cbar' }, bar), val));
        this.qRows.push({ bar, val, name });
      }
      this.actHost.append(bars);
      return;
    }
    for (const ax of spec.axes) {
      const opts = h('div', { class: 'act-opts' });
      const cells: { fill: HTMLElement; cell: HTMLElement; text: HTMLElement | null }[] = [];
      if (spec.actionMode === 'discrete')
        for (const lab of ax.labels) {
          const fill = h('i');
          const text = h('span', null, lab);
          const cell = h('div', { class: 'act-opt', style: { height: '22px' } }, fill, text);
          opts.append(cell);
          cells.push({ fill, cell, text });
        }
      else {
        const fill = h('i');
        const text = h('span');
        const cell = h('div', { class: 'act-opt', style: { height: '22px' } }, fill, text);
        opts.append(cell);
        cells.push({ fill, cell, text });
      }
      this.actHost.append(h('div', { class: 'act-row' }, h('span', { class: 'lbl' }, ax.axis), opts));
      this.actCells.push(cells);
    }
  }
}
