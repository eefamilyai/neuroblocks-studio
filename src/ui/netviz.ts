/**
 * Live neural-network visualiser. Every circle is a neuron, every line a
 * weight. Neurons light up with their activation as the robot thinks.
 * Click a neuron to see exactly how its value was computed.
 */
import type { Brain } from '../ml/algos/common.ts';
import { ACTIVATION_INFO } from '../ml/nn.ts';
import { clear, fmt, h } from './dom.ts';

interface Trace {
  pre: Float32Array[];
  post: Float32Array[];
  raw: Float32Array;
}

type EdgeMode = 'weights' | 'flow';

export class NetViz {
  readonly el: HTMLElement;
  private canvas: HTMLCanvasElement;
  private detail: HTMLElement;
  private tipEl: HTMLElement;
  brain: Brain | null = null;
  private trace: Trace | null = null;
  private pos: { x: number; y: number; r: number }[][] = [];
  private thresholds: number[] = [];
  private maxAbs: number[] = [];
  private hover: [number, number] | null = null;
  private selected: [number, number] | null = null;
  private chosen: Float32Array | null = null;
  edgeMode: EdgeMode = 'flow';
  private lastDraw = 0;
  private dirty = true;
  value: number | null = null;

  constructor() {
    this.canvas = h('canvas');
    this.tipEl = h('div', { class: 'chart-tip', style: { display: 'none' } });
    this.el = h('div', { class: 'netviz' }, this.canvas, this.tipEl);
    this.detail = h('div', { class: 'card neuron-detail' }, h('span', { class: 'muted' }, '👆 Click any neuron to see exactly how it computes its value.'));
    this.canvas.addEventListener('mousemove', (e) => this.onMove(e));
    this.canvas.addEventListener('mouseleave', () => {
      this.hover = null;
      this.tipEl.style.display = 'none';
      this.dirty = true;
    });
    this.canvas.addEventListener('click', () => {
      this.selected = this.hover ? [...this.hover] as [number, number] : null;
      this.renderDetail();
      this.dirty = true;
    });
    new ResizeObserver(() => {
      this.dirty = true;
    }).observe(this.el);
  }

  get detailEl(): HTMLElement {
    return this.detail;
  }

  setBrain(b: Brain | null): void {
    this.brain = b;
    this.thresholds = [];
    this.maxAbs = [];
    if (b) {
      for (const l of b.net.layers) {
        const abs = Array.from(l.W, Math.abs).sort((p, q) => q - p);
        const keep = Math.min(abs.length, 1600);
        this.thresholds.push(abs[keep - 1] ?? 0);
        this.maxAbs.push(abs[0] || 1);
      }
    }
    if (this.selected && b && this.selected[0] > b.net.layers.length) this.selected = null;
    this.dirty = true;
    this.renderDetail();
  }

  setTrace(t: Trace | null, chosen: Float32Array | null, value: number | null): void {
    this.trace = t;
    this.chosen = chosen;
    this.value = value;
    this.dirty = true;
  }

  private layerSizes(): number[] {
    const b = this.brain;
    if (!b) return [];
    return [b.net.inSize, ...b.net.layers.map((l) => l.outSize)];
  }

  private inputLabel(i: number): string {
    return this.brain?.snap.obsLabels[i] ?? `input ${i}`;
  }

  private outputLabel(j: number): string {
    const b = this.brain;
    if (!b) return '';
    const sp = b.snap.space;
    if (b.snap.kind === 'q') {
      // joint action names
      const parts: string[] = [];
      let idx = j;
      for (let k = sp.branches.length - 1; k >= 0; k--) {
        const n = sp.branches[k];
        parts.unshift(sp.optionLabels[k]?.[idx % n] ?? String(idx % n));
        idx = Math.floor(idx / n);
      }
      return 'Q(' + parts.join(' + ') + ')';
    }
    if (sp.kind === 'continuous') return `${sp.labels[j]} (mean)`;
    let o = j;
    for (let k = 0; k < sp.branches.length; k++) {
      if (o < sp.branches[k]) return `${sp.labels[k]}: ${sp.optionLabels[k]?.[o] ?? o}`;
      o -= sp.branches[k];
    }
    return `out ${j}`;
  }

  private layout(W: number, H: number): void {
    const sizes = this.layerSizes();
    const L = sizes.length;
    const left = 128;
    const right = 132;
    const top = 14;
    const bottom = 14;
    this.pos = sizes.map((n, l) => {
      const x = left + (L === 1 ? 0 : (l / (L - 1)) * (W - left - right));
      const avail = H - top - bottom;
      const gap = Math.min(22, avail / Math.max(1, n));
      const r = Math.max(1.6, Math.min(7, gap * 0.36));
      const total = gap * (n - 1);
      const y0 = top + (avail - total) / 2;
      return Array.from({ length: n }, (_, i) => ({ x, y: y0 + i * gap, r }));
    });
  }

  private onMove(e: MouseEvent): void {
    const r = this.canvas.getBoundingClientRect();
    const mx = e.clientX - r.left;
    const my = e.clientY - r.top;
    let best = null as [number, number] | null;
    let bd = 14 * 14;
    this.pos.forEach((col, l) =>
      col.forEach((p, i) => {
        const d = (p.x - mx) ** 2 + (p.y - my) ** 2;
        if (d < bd) {
          bd = d;
          best = [l, i];
        }
      }),
    );
    const changed = JSON.stringify(best) !== JSON.stringify(this.hover);
    this.hover = best;
    if (best) {
      const [l, i] = best;
      const v = this.trace?.post[l]?.[i];
      const name = l === 0 ? this.inputLabel(i) : l === this.pos.length - 1 ? this.outputLabel(i) : `hidden layer ${l}, neuron ${i}`;
      this.tipEl.innerHTML = `${name}<br>value: ${v === undefined ? '–' : fmt(v, 3)}`;
      this.tipEl.style.display = 'block';
      this.tipEl.style.left = Math.min(mx + 12, r.width - this.tipEl.offsetWidth - 4) + 'px';
      this.tipEl.style.top = Math.max(4, my - 30) + 'px';
    } else this.tipEl.style.display = 'none';
    if (changed) this.dirty = true;
  }

  /** Call every frame; redraws at most ~20 fps. */
  draw(force = false): void {
    const now = performance.now();
    if (!force && (!this.dirty || now - this.lastDraw < 50)) return;
    this.lastDraw = now;
    this.dirty = false;
    const c = this.canvas;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const W = Math.max(10, this.el.clientWidth);
    const H = Math.max(10, this.el.clientHeight);
    if (c.width !== W * dpr || c.height !== H * dpr) {
      c.width = W * dpr;
      c.height = H * dpr;
    }
    const g = c.getContext('2d')!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const b = this.brain;
    if (!b) {
      g.fillStyle = '#5b6388';
      g.font = '13px Nunito';
      g.textAlign = 'center';
      g.fillText('No brain yet: fix the blocks, then the network appears here.', W / 2, H / 2);
      return;
    }
    this.layout(W, H);
    const tr = this.trace;
    const hov = this.hover ?? this.selected;
    // ---- edges ----
    for (let l = 0; l < b.net.layers.length; l++) {
      const layer = b.net.layers[l];
      const A = this.pos[l];
      const Bp = this.pos[l + 1];
      const I = layer.inSize;
      const O = layer.outSize;
      const thr = this.thresholds[l];
      const maxA = this.maxAbs[l];
      const xin = tr?.post[l];
      let maxFlow = 1e-9;
      if (this.edgeMode === 'flow' && xin) for (let o = 0; o < O; o++) for (let i = 0; i < I; i++) maxFlow = Math.max(maxFlow, Math.abs(layer.W[o * I + i] * xin[i]));
      const buckets: Path2D[] = Array.from({ length: 16 }, () => new Path2D());
      const used = new Uint8Array(16);
      for (let o = 0; o < O; o++) {
        for (let i = 0; i < I; i++) {
          const w = layer.W[o * I + i];
          const focus = hov && ((hov[0] === l + 1 && hov[1] === o) || (hov[0] === l && hov[1] === i));
          let s: number;
          if (this.edgeMode === 'flow' && xin) s = Math.abs(w * xin[i]) / maxFlow;
          else {
            if (Math.abs(w) < thr && !focus) continue;
            s = Math.abs(w) / maxA;
          }
          if (s < 0.03 && !focus) continue;
          if (hov && !focus) s *= 0.25;
          const sign = (this.edgeMode === 'flow' && xin ? w * xin[i] : w) >= 0 ? 0 : 1;
          const k = sign * 8 + Math.min(7, Math.floor(Math.sqrt(s) * 8));
          buckets[k].moveTo(A[i].x, A[i].y);
          buckets[k].lineTo(Bp[o].x, Bp[o].y);
          used[k] = 1;
        }
      }
      for (let k = 0; k < 16; k++) {
        if (!used[k]) continue;
        const a = 0.05 + ((k % 8) / 7) * 0.75;
        g.strokeStyle = k < 8 ? `rgba(76,195,255,${a})` : `rgba(255,140,80,${a})`;
        g.lineWidth = 0.6 + (k % 8) * 0.28;
        g.stroke(buckets[k]);
      }
    }
    // ---- neurons ----
    const nL = this.pos.length;
    for (let l = 0; l < nL; l++) {
      const col = this.pos[l];
      const vals = tr?.post[l];
      let m = 1e-6;
      if (vals) for (const v of vals) m = Math.max(m, Math.abs(v));
      for (let i = 0; i < col.length; i++) {
        const p = col[i];
        const v = vals ? vals[i] / m : 0;
        g.beginPath();
        g.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        g.fillStyle = v >= 0 ? `rgba(255,${Math.round(200 + 40 * v)},${Math.round(90 - 60 * v)},${0.18 + 0.82 * v})` : `rgba(90,150,255,${0.18 + 0.82 * -v})`;
        g.fill();
        const sel = (this.selected && this.selected[0] === l && this.selected[1] === i) || (this.hover && this.hover[0] === l && this.hover[1] === i);
        g.lineWidth = sel ? 2.5 : 1;
        g.strokeStyle = sel ? '#fff' : 'rgba(255,255,255,0.35)';
        g.stroke();
      }
    }
    // ---- labels ----
    g.font = '10px Nunito, sans-serif';
    const inCol = this.pos[0];
    const showIn = inCol.length <= 64;
    if (showIn) {
      g.textAlign = 'right';
      inCol.forEach((p, i) => {
        g.fillStyle = '#9aa3c7';
        const lab = this.inputLabel(i);
        g.fillText(lab.length > 20 ? lab.slice(0, 19) + '…' : lab, p.x - p.r - 5, p.y + 3);
      });
    }
    const outCol = this.pos[nL - 1];
    g.textAlign = 'left';
    const probs = this.brain?.lastProbs;
    outCol.forEach((p, j) => {
      let pr: number | null = null;
      let isChosen = false;
      if (b.snap.kind === 'policy' && b.snap.space.kind === 'discrete' && probs) {
        let o = j;
        for (let k = 0; k < b.snap.space.branches.length; k++) {
          if (o < b.snap.space.branches[k]) {
            pr = probs[k]?.[o] ?? null;
            if (this.chosen) isChosen = this.chosen[k] === o;
            break;
          }
          o -= b.snap.space.branches[k];
        }
      }
      if (b.snap.kind === 'q' && probs && probs[0]) {
        pr = probs[0][j];
      }
      if (pr !== null) {
        g.fillStyle = 'rgba(139,108,255,0.35)';
        g.fillRect(p.x + p.r + 4, p.y - 5, 40, 10);
        g.fillStyle = isChosen ? '#ffd166' : '#8b6cff';
        g.fillRect(p.x + p.r + 4, p.y - 5, 40 * pr, 10);
      }
      g.fillStyle = isChosen ? '#ffd166' : '#cfd5ef';
      const lab = this.outputLabel(j);
      g.fillText(lab.length > 17 ? lab.slice(0, 16) + '…' : lab, p.x + p.r + (pr !== null ? 48 : 5), p.y + 3);
    });
    // layer captions
    g.textAlign = 'center';
    g.fillStyle = '#6b7399';
    this.pos.forEach((col, l) => {
      if (!col.length) return;
      const name = l === 0 ? `${col.length} inputs` : l === nL - 1 ? `${col.length} outputs` : `${col.length} × ${b.net.layers[l - 1].act}`;
      g.fillText(name, col[0].x, H - 3);
    });
    if (this.value !== null && b.critic) {
      g.textAlign = 'right';
      g.fillStyle = '#ffd166';
      g.font = '800 11px Nunito';
      g.fillText(`critic V(s) = ${fmt(this.value, 2)}`, W - 6, 12);
    }
  }

  renderDetail(): void {
    const d = this.detail;
    clear(d);
    const b = this.brain;
    const sel = this.selected;
    if (!b || !sel) {
      d.append(h('span', { class: 'muted' }, '👆 Click any neuron to see exactly how it computes its value.'));
      return;
    }
    const [l, i] = sel;
    const tr = this.trace;
    if (l === 0) {
      const raw = tr?.raw[i];
      const norm = tr?.post[0][i];
      const outW = b.net.layers[0];
      const ws = Array.from({ length: outW.outSize }, (_, o) => outW.W[o * outW.inSize + i]);
      const mean = ws.reduce((a, c) => a + Math.abs(c), 0) / ws.length;
      d.append(
        h('div', null, h('b', null, `Input #${i}: ${this.inputLabel(i)}`)),
        h('div', { class: 'small muted' }, 'An input neuron just holds one number from the senses.'),
        h('div', { class: 'mono', style: { marginTop: '6px' } }, `raw sensor value = ${raw === undefined ? '–' : fmt(raw, 4)}`),
        ...(b.norm ? [h('div', { class: 'mono' }, `normalised (x−μ)/σ = ${norm === undefined ? '–' : fmt(norm, 4)}   (μ=${fmt(b.norm.mean[i], 3)}, σ=${fmt(Math.sqrt(b.norm.var[i]), 3)})`)] : []),
        h('div', { class: 'small', style: { marginTop: '6px' } }, `It feeds ${ws.length} neurons in the next layer. Average |weight| = ${fmt(mean, 3)}. Larger weights = this sense matters more.`),
      );
      return;
    }
    const layer = b.net.layers[l - 1];
    const I = layer.inSize;
    const xin = tr?.post[l - 1];
    const z = tr?.pre[l - 1]?.[i];
    const a = tr?.post[l]?.[i];
    const bias = layer.b[i];
    const isOut = l === b.net.layers.length;
    const contribs = Array.from({ length: I }, (_, k) => ({ k, w: layer.W[i * I + k], x: xin ? xin[k] : NaN, c: xin ? layer.W[i * I + k] * xin[k] : NaN }));
    contribs.sort((p, q) => Math.abs(q.c || q.w) - Math.abs(p.c || p.w));
    const top = contribs.slice(0, 8);
    const maxC = Math.max(1e-9, ...top.map((t) => Math.abs(Number.isFinite(t.c) ? t.c : t.w)));
    const info = ACTIVATION_INFO[layer.act];
    const prevName = (k: number) => (l - 1 === 0 ? this.inputLabel(k) : `L${l - 1}·n${k}`);
    d.append(
      h('div', null, h('b', null, isOut ? `Output #${i}: ${this.outputLabel(i)}` : `Hidden layer ${l}, neuron #${i}`)),
      h('div', { class: 'small muted' }, `${I} weights + 1 bias · activation ${layer.act}: ${info.formula}`),
      h('div', { class: 'formula', style: { fontSize: '13px' } }, 'z = Σ wᵢ·xᵢ + b = ', h('span', { class: 'live' }, z === undefined ? '…' : fmt(z, 4)), '   →   a = ', layer.act, '(z) = ', h('span', { class: 'live' }, a === undefined ? '…' : fmt(a, 4))),
      h('div', { class: 'small', style: { margin: '4px 0' } }, `bias b = ${fmt(bias, 4)}. Biggest contributions w·x right now:`),
      h(
        'div',
        { class: 'contrib' },
        ...top.flatMap((t) => {
          const v = Number.isFinite(t.c) ? t.c : t.w;
          const pct = (Math.abs(v) / maxC) * 100;
          return [
            h('span', { class: 'muted', title: prevName(t.k) }, `${prevName(t.k).slice(0, 26)}  (w=${fmt(t.w, 2)})`),
            h('div', { class: 'bar', style: { width: pct + '%', background: v >= 0 ? '#4cc3ff' : '#ff8c50' } }),
            h('span', { class: 'mono' }, fmt(v, 3)),
          ];
        }),
      ),
      h('div', { class: 'small muted', style: { marginTop: '6px' } }, isOut ? 'Output values (logits) are turned into probabilities with softmax.' : info.blurb),
    );
  }
}
