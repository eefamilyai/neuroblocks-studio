/** 🧠 Brain tab: the network graph, weight matrices and histograms. */
import type { Brain } from '../ml/algos/common.ts';
import { NetViz } from './netviz.ts';
import { clear, fmt, h, section, seg } from './dom.ts';

export class BrainPanel {
  readonly el: HTMLElement;
  readonly viz = new NetViz();
  private stats: HTMLElement;
  private hists: HTMLElement;
  private matrices: HTMLElement;
  private brain: Brain | null = null;
  private lastDetail = 0;

  constructor() {
    this.stats = h('div', { class: 'note' });
    this.hists = h('div', { class: 'grid2' });
    this.matrices = h('div', { class: 'matrix-wrap' });
    const mode = seg(
      [
        { value: 'flow', label: 'live signal', title: 'weight × activation: what actually flows through the network right now' },
        { value: 'weights', label: 'weights', title: 'the raw weights' },
      ],
      (v) => {
        this.viz.edgeMode = v;
        mode.set(v);
        this.viz.draw(true);
      },
      'sm',
    );
    mode.set('flow');
    this.el = h(
      'div',
      null,
      section('The neural network', 'brain', [this.viz.el, h('div', { class: 'note' }, 'Blue = positive, orange = negative. "Live signal" shows weight × activation: the numbers actually flowing through the network as the robot thinks. Neuron colour = activation (yellow positive, blue negative). Click a neuron.'), this.stats], [mode.el]),
      section('Neuron inspector', 'search', [this.viz.detailEl]),
      section('Weight matrices', 'scan', [h('div', { class: 'note', style: { margin: '0 0 8px' } }, 'Each layer is just a grid of numbers W (rows = neurons, columns = inputs). This is literally what gets exported.'), this.matrices]),
      section('Weight distributions', 'chart', [this.hists]),
    );
  }

  setBrain(b: Brain | null): void {
    this.brain = b;
    this.viz.setBrain(b);
    clear(this.hists);
    clear(this.matrices);
    if (!b) {
      this.stats.textContent = '';
      return;
    }
    const params = b.net.numParams() + (b.critic ? b.critic.numParams() : 0);
    this.stats.textContent = `${b.snap.algo.toUpperCase()} brain · ${b.net.numParams().toLocaleString()} weights in the ${b.snap.kind === 'q' ? 'Q-network' : 'policy'}${b.critic ? ` + ${b.critic.numParams().toLocaleString()} in the critic` : ''} (${params.toLocaleString()} total) · trained for ${b.snap.steps.toLocaleString()} steps, ${b.snap.updates} updates`;
    b.net.layers.forEach((l, i) => {
      // histogram
      const bins = new Array<number>(24).fill(0);
      let mn = Infinity;
      let mx = -Infinity;
      for (const w of l.W) {
        mn = Math.min(mn, w);
        mx = Math.max(mx, w);
      }
      const span = mx - mn || 1;
      for (const w of l.W) bins[Math.min(23, Math.floor(((w - mn) / span) * 24))]++;
      const peak = Math.max(...bins);
      let mean = 0;
      for (const w of l.W) mean += w;
      mean /= l.W.length;
      let sd = 0;
      for (const w of l.W) sd += (w - mean) ** 2;
      sd = Math.sqrt(sd / l.W.length);
      this.hists.append(
        h(
          'div',
          { class: 'card' },
          h('div', { class: 'small' }, h('b', null, `Layer ${i + 1}`), h('span', { class: 'muted' }, ` ${l.inSize}→${l.outSize} · ${l.act}`)),
          h('div', { class: 'hist' }, ...bins.map((c) => h('i', { style: { height: `${(c / peak) * 100}%` } }))),
          h('div', { class: 'tiny muted mono' }, `μ=${fmt(mean, 3)} σ=${fmt(sd, 3)} [${fmt(mn, 2)}, ${fmt(mx, 2)}]`),
        ),
      );
      // matrix heat map
      const cv = h('canvas') as HTMLCanvasElement;
      cv.width = l.inSize;
      cv.height = l.outSize;
      const scale = Math.max(1, Math.min(6, Math.floor(180 / Math.max(l.inSize, l.outSize))));
      cv.style.width = `${l.inSize * scale}px`;
      cv.style.height = `${l.outSize * scale}px`;
      const g = cv.getContext('2d')!;
      const img = g.createImageData(l.inSize, l.outSize);
      const m = Math.max(Math.abs(mn), Math.abs(mx)) || 1;
      for (let k = 0; k < l.W.length; k++) {
        const v = l.W[k] / m;
        const o = k * 4;
        if (v >= 0) {
          img.data[o] = 30 + 46 * v;
          img.data[o + 1] = 40 + 155 * v;
          img.data[o + 2] = 60 + 195 * v;
        } else {
          img.data[o] = 60 + 195 * -v;
          img.data[o + 1] = 40 + 100 * -v;
          img.data[o + 2] = 30 + 50 * -v;
        }
        img.data[o + 3] = 255;
      }
      g.putImageData(img, 0, 0);
      this.matrices.append(h('figure', null, cv, `W${i + 1}: ${l.outSize}×${l.inSize}`));
    });
  }

  render(visible: boolean): void {
    if (!visible) return;
    this.viz.draw();
    const now = performance.now();
    if (now - this.lastDetail > 300) {
      this.lastDetail = now;
      this.viz.renderDetail();
    }
  }
}
