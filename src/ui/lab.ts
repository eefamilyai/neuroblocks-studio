/**
 * 🧪 Neuron Lab: supervised learning in its purest form.
 *
 * A tiny network learns to separate two classes of 2D points. You can watch
 * the decision boundary bend as gradient descent runs, compare training and
 * test loss (overfitting!), and change layers, activations and learning rate.
 * It uses exactly the same MLP + optimizers as the robots.
 */
import { ACTIVATIONS, MLP } from '../ml/nn.ts';
import type { Activation, LayerSpec } from '../ml/nn.ts';
import { makeOptimizer } from '../ml/optim.ts';
import type { Optimizer, OptimizerKind } from '../ml/optim.ts';
import { RNG } from '../ml/rng.ts';
import { LineChart } from './charts.ts';
import { btn, clear, fmt, h, modal, seg, setText } from './dom.ts';

type DatasetKind = 'circle' | 'xor' | 'spiral' | 'blobs' | 'moons' | 'draw';
type Feature = 'x' | 'y' | 'x²' | 'y²' | 'x·y' | 'sin x' | 'sin y';

const FEATURES: Feature[] = ['x', 'y', 'x²', 'y²', 'x·y', 'sin x', 'sin y'];

function feature(f: Feature, x: number, y: number): number {
  switch (f) {
    case 'x':
      return x;
    case 'y':
      return y;
    case 'x²':
      return x * x;
    case 'y²':
      return y * y;
    case 'x·y':
      return x * y;
    case 'sin x':
      return Math.sin(x * 2);
    default:
      return Math.sin(y * 2);
  }
}

function makeData(kind: DatasetKind, n: number, noise: number, rng: RNG): { x: number; y: number; c: number }[] {
  const pts: { x: number; y: number; c: number }[] = [];
  if (kind === 'draw') return pts;
  const nz = () => rng.gauss() * noise;
  for (let i = 0; i < n; i++) {
    const c = i % 2;
    let x = 0;
    let y = 0;
    switch (kind) {
      case 'circle': {
        const r = c ? rng.uniform(0, 0.45) : rng.uniform(0.65, 1);
        const a = rng.uniform(0, Math.PI * 2);
        x = r * Math.cos(a) * 2.2;
        y = r * Math.sin(a) * 2.2;
        break;
      }
      case 'xor': {
        x = rng.uniform(-2, 2);
        y = rng.uniform(-2, 2);
        const pad = 0.15;
        x += x > 0 ? pad : -pad;
        y += y > 0 ? pad : -pad;
        pts.push({ x: x + nz(), y: y + nz(), c: x * y > 0 ? 1 : 0 });
        continue;
      }
      case 'spiral': {
        const t = rng.uniform(0, 1);
        const r = t * 2;
        const a = t * Math.PI * 3.2 + c * Math.PI;
        x = r * Math.cos(a);
        y = r * Math.sin(a);
        break;
      }
      case 'blobs':
        x = (c ? 1 : -1) + rng.gauss() * 0.55;
        y = (c ? 1 : -1) + rng.gauss() * 0.55;
        break;
      case 'moons': {
        const a = rng.uniform(0, Math.PI);
        x = c ? 1 - Math.cos(a) * 1.4 : Math.cos(a) * 1.4 - 0.2;
        y = c ? 0.5 - Math.sin(a) * 1.2 : Math.sin(a) * 1.2 - 0.5;
        break;
      }
    }
    pts.push({ x: x + nz(), y: y + nz(), c });
  }
  return pts;
}

export function openNeuronLab(): void {
  const rng = new RNG(7);
  const state = {
    dataset: 'circle' as DatasetKind,
    noise: 0.1,
    features: new Set<Feature>(['x', 'y']),
    layers: [{ units: 6, activation: 'tanh' as Activation }, { units: 4, activation: 'tanh' as Activation }] as LayerSpec[],
    lr: 0.03,
    optimizer: 'adam' as OptimizerKind,
    batch: 16,
    /** L2 regularisation (weight decay) */
    l2: 0,
    testFrac: 0.3,
    points: 400,
    /** colour for clicks on the plot */
    paint: 0,
    speed: 3,
    running: false,
    epoch: 0,
  };
  let train: { x: number; y: number; c: number }[] = [];
  let test: { x: number; y: number; c: number }[] = [];
  let net!: MLP;
  let opt!: Optimizer;
  const lossX: number[] = [];
  const lossTrain: number[] = [];
  const lossTest: number[] = [];

  const canvas = h('canvas', { width: '480', height: '480', style: { width: '100%', maxWidth: '480px', aspectRatio: '1', borderRadius: '12px', border: '1px solid var(--line-2)', background: '#0f1324', display: 'block' } }) as HTMLCanvasElement;
  const chart = new LineChart('Loss (lower is better)', 'Binary cross-entropy on the training points (the network learns from these) and on the TEST points (never used for learning). When test loss goes up while train loss goes down, the network is memorising: overfitting.');
  const stats = h('div', { class: 'kv' });
  const archEl = h('div', { class: 'note', style: { margin: '2px 0 0' } });
  const layerHost = h('div');

  const featureInputs = (): Feature[] => FEATURES.filter((f) => state.features.has(f));

  function rebuild(): void {
    const feats = featureInputs();
    net = new MLP(Math.max(1, feats.length), state.layers, 1, 'linear', new RNG(rng.int(1e9)));
    opt = makeOptimizer(state.optimizer, net.params(), state.lr);
    state.epoch = 0;
    lossX.length = 0;
    lossTrain.length = 0;
    lossTest.length = 0;
    archEl.textContent = `${feats.length} inputs → ${state.layers.map((l) => `${l.units} ${l.activation}`).join(' → ') || '(no hidden layer)'} → 1 output (sigmoid) · ${net.numParams()} weights`;
    draw();
  }

  function regen(): void {
    const all = makeData(state.dataset, state.points, state.noise, rng);
    rng.shuffle(all);
    const nTest = Math.round(all.length * state.testFrac);
    test = all.slice(0, nTest);
    train = all.slice(nTest);
    rebuild();
  }

  function inputsOf(pts: { x: number; y: number }[]): Float32Array {
    const feats = featureInputs();
    const F = Math.max(1, feats.length);
    const X = new Float32Array(pts.length * F);
    pts.forEach((p, i) => feats.forEach((f, k) => (X[i * F + k] = feature(f, p.x, p.y))));
    return X;
  }

  const sig = (z: number) => 1 / (1 + Math.exp(-z));

  function evalLoss(pts: { x: number; y: number; c: number }[]): { loss: number; acc: number } {
    if (!pts.length) return { loss: NaN, acc: NaN };
    const out = net.forward(inputsOf(pts), pts.length);
    let loss = 0;
    let ok = 0;
    pts.forEach((p, i) => {
      const q = Math.min(1 - 1e-7, Math.max(1e-7, sig(out[i])));
      loss += -(p.c * Math.log(q) + (1 - p.c) * Math.log(1 - q));
      if ((q > 0.5 ? 1 : 0) === p.c) ok++;
    });
    return { loss: loss / pts.length, acc: ok / pts.length };
  }

  /** one epoch of minibatch gradient descent */
  function epoch(): void {
    if (!train.length) return;
    const idx = train.map((_, i) => i);
    rng.shuffle(idx);
    const B = state.batch > 0 ? state.batch : idx.length;
    const params = net.params();
    for (let s = 0; s < idx.length; s += B) {
      const batch = idx.slice(s, s + B).map((i) => train[i]);
      const X = inputsOf(batch);
      opt.zeroGrad();
      const out = net.forward(X, batch.length);
      const d = new Float32Array(batch.length);
      // sigmoid + cross-entropy: dL/dz = p − y  (the nicest gradient in ML)
      batch.forEach((p, i) => (d[i] = (sig(out[i]) - p.c) / batch.length));
      net.backward(d, batch.length);
      // L2 regularisation: loss + λ/2·Σw²  →  gradient + λ·w (weights only, not biases)
      if (state.l2 > 0) for (const p of params) if (p.name.endsWith('weight')) for (let i = 0; i < p.w.length; i++) p.g[i] += state.l2 * p.w[i];
      opt.step();
    }
    state.epoch++;
  }

  function draw(): void {
    const g = canvas.getContext('2d')!;
    const W = canvas.width;
    const S = 2.6;
    const toPx = (v: number) => ((v + S) / (2 * S)) * W;
    // decision boundary heat map: run the network on a grid
    const R = 64;
    const grid: { x: number; y: number }[] = [];
    for (let j = 0; j < R; j++) for (let i = 0; i < R; i++) grid.push({ x: -S + ((i + 0.5) / R) * 2 * S, y: S - ((j + 0.5) / R) * 2 * S });
    const out = net.forward(inputsOf(grid), grid.length);
    const img = g.createImageData(R, R);
    for (let k = 0; k < grid.length; k++) {
      const p = sig(out[k]);
      const t = p * 2 - 1; // −1 (class 0) … +1 (class 1)
      const o = k * 4;
      if (t >= 0) {
        img.data[o] = 30 + 225 * t * 0.75;
        img.data[o + 1] = 40 + 150 * t * 0.75;
        img.data[o + 2] = 60;
      } else {
        img.data[o] = 40;
        img.data[o + 1] = 70 + 110 * -t * 0.75;
        img.data[o + 2] = 90 + 165 * -t * 0.75;
      }
      img.data[o + 3] = 255;
    }
    const tmp = document.createElement('canvas');
    tmp.width = R;
    tmp.height = R;
    tmp.getContext('2d')!.putImageData(img, 0, 0);
    g.imageSmoothingEnabled = true;
    g.drawImage(tmp, 0, 0, W, W);
    // axes
    g.strokeStyle = 'rgba(255,255,255,0.12)';
    g.beginPath();
    g.moveTo(toPx(0), 0);
    g.lineTo(toPx(0), W);
    g.moveTo(0, toPx(0));
    g.lineTo(W, toPx(0));
    g.stroke();
    // points (train filled, test hollow)
    for (const [pts, hollow] of [
      [train, false],
      [test, true],
    ] as const) {
      for (const p of pts) {
        const x = toPx(p.x);
        const y = W - toPx(p.y);
        g.beginPath();
        g.arc(x, y, 4, 0, Math.PI * 2);
        const col = p.c ? '#ffb347' : '#4cc3ff';
        if (hollow) {
          g.strokeStyle = col;
          g.lineWidth = 2;
          g.stroke();
        } else {
          g.fillStyle = col;
          g.fill();
          g.strokeStyle = 'rgba(0,0,0,0.5)';
          g.lineWidth = 1;
          g.stroke();
        }
      }
    }
    const tr = evalLoss(train);
    const te = evalLoss(test);
    lossX.push(state.epoch);
    lossTrain.push(tr.loss);
    lossTest.push(te.loss);
    if (lossX.length > 3000) {
      lossX.shift();
      lossTrain.shift();
      lossTest.shift();
    }
    chart.smoothing = 0;
    chart.xLabel = (v) => String(Math.round(v));
    chart.setSeries([
      { name: 'train', color: '#2fd182', xs: lossX, ys: lossTrain, raw: true },
      { name: 'test', color: '#ff6fae', xs: lossX, ys: lossTest, raw: true },
    ]);
    chart.draw();
    clear(stats);
    const row = (k: string, v: string) => stats.append(h('span', null, k), h('b', null, v));
    const pct = (v: number) => (Number.isFinite(v) ? `${(v * 100).toFixed(0)}%` : '–');
    row('Epoch', String(state.epoch));
    row(`Train (${train.length} points) loss / accuracy`, `${fmt(tr.loss, 3)} / ${pct(tr.acc)}`);
    row(`Test (${test.length} points) loss / accuracy`, `${fmt(te.loss, 3)} / ${pct(te.acc)}`);
  }

  function renderLayers(): void {
    clear(layerHost);
    state.layers.forEach((l, i) => {
      const units = h('input', { type: 'number', min: '1', max: '64', value: String(l.units), class: 'input sm', style: { width: '64px' } }) as HTMLInputElement;
      units.addEventListener('change', () => {
        l.units = Math.max(1, Math.min(64, Number(units.value) || 1));
        rebuild();
      });
      const act = h('select', { class: 'sel sm' }) as HTMLSelectElement;
      for (const a of ACTIVATIONS) act.append(h('option', { value: a, selected: a === l.activation }, a));
      act.addEventListener('change', () => {
        l.activation = act.value as Activation;
        rebuild();
      });
      layerHost.append(
        h(
          'div',
          { class: 'row', style: { marginBottom: '4px' } },
          h('span', { class: 'small muted', style: { width: '52px' } }, `layer ${i + 1}`),
          units,
          act,
          btn({
            icon: 'x',
            cls: 'xs ghost',
            title: 'Remove this layer',
            onclick: () => {
              state.layers.splice(i, 1);
              renderLayers();
              rebuild();
            },
          }),
        ),
      );
    });
    if (state.layers.length < 5)
      layerHost.append(
        btn({
          icon: 'plus',
          label: 'Add layer',
          cls: 'xs',
          onclick: () => {
            state.layers.push({ units: 4, activation: 'tanh' });
            renderLayers();
            rebuild();
          },
        }),
      );
  }

  const select = <T extends string | number>(opts: [T, string][], value: T, on: (v: T) => void, title = ''): HTMLSelectElement => {
    const el = h('select', { class: 'sel sm', title }) as HTMLSelectElement;
    for (const [v, l] of opts) el.append(h('option', { value: String(v), selected: v === value }, l));
    el.addEventListener('change', () => on((typeof value === 'number' ? Number(el.value) : el.value) as T));
    return el;
  };
  const dsSel = select<DatasetKind>(
    [
      ['circle', 'circle'],
      ['xor', 'XOR'],
      ['spiral', 'spiral'],
      ['blobs', 'two blobs'],
      ['moons', 'moons'],
      ['draw', 'draw your own'],
    ],
    state.dataset,
    (v) => {
      state.dataset = v;
      regen();
      setText(drawHint, v === 'draw' ? 'Click the plot to add points (pick the colour below). Add some of each colour, then press Train.' : 'Click the plot to add your own points to any dataset (e.g. outliers).');
    },
  );
  const noise = h('input', { type: 'range', min: '0', max: '0.6', step: '0.02', value: String(state.noise), style: { width: '90px' } }) as HTMLInputElement;
  noise.addEventListener('change', () => {
    state.noise = Number(noise.value);
    regen();
  });
  const sizeSel = select<number>([[100, '100 points'], [400, '400 points'], [1000, '1000 points']], state.points, (v) => ((state.points = v), regen()));
  const splitSel = select<number>([[0.1, '10% test'], [0.3, '30% test'], [0.5, '50% test']], state.testFrac, (v) => ((state.testFrac = v), regen()), 'Points held back to check generalisation');
  const lrSel = select<number>([[0.0003, '0.0003'], [0.001, '0.001'], [0.003, '0.003'], [0.01, '0.01'], [0.03, '0.03'], [0.1, '0.1'], [0.3, '0.3'], [1, '1']], state.lr, (v) => ((state.lr = v), (opt.lr = v)));
  const optSel = select<OptimizerKind>([['adam', 'Adam'], ['sgd', 'SGD'], ['momentum', 'Momentum'], ['rmsprop', 'RMSProp']], state.optimizer, (v) => ((state.optimizer = v), (opt = makeOptimizer(v, net.params(), state.lr))));
  const batchSel = select<number>([[1, 'batch 1'], [4, 'batch 4'], [16, 'batch 16'], [64, 'batch 64'], [0, 'full batch']], state.batch, (v) => (state.batch = v), 'Points per gradient step');
  const l2Sel = select<number>([[0, 'L2 off'], [0.0001, 'L2 0.0001'], [0.001, 'L2 0.001'], [0.01, 'L2 0.01'], [0.1, 'L2 0.1']], state.l2, (v) => (state.l2 = v), 'L2 regularisation (weight decay): punishes big weights, fights overfitting');
  const speedSel = select<number>([[1, '1 epoch / frame'], [3, '3 epochs / frame'], [10, '10 epochs / frame']], state.speed, (v) => (state.speed = v));
  const featHost = h('div', { class: 'row', style: { flexWrap: 'wrap', gap: '4px' } });
  for (const f of FEATURES) {
    const b = btn({ label: f, cls: 'xs' + (state.features.has(f) ? ' on' : '') });
    b.addEventListener('click', () => {
      if (state.features.has(f)) state.features.delete(f);
      else state.features.add(f);
      b.classList.toggle('on', state.features.has(f));
      rebuild();
    });
    featHost.append(b);
  }
  const paint = seg<string>(
    [
      { value: '0', label: '● blue' },
      { value: '1', label: '● orange' },
    ],
    (v) => {
      state.paint = Number(v);
      paint.set(v);
    },
    'sm',
  );
  paint.set('0');
  (paint.el.children[0] as HTMLElement).style.color = '#4cc3ff';
  (paint.el.children[1] as HTMLElement).style.color = '#ffb347';
  const drawHint = h('div', { class: 'note', style: { margin: 0 } }, 'Click the plot to add your own points to any dataset (e.g. outliers).');
  // click the plot: add a training point (Shift = the other colour)
  canvas.style.cursor = 'crosshair';
  const addPoint = (e: MouseEvent, other: boolean) => {
    const r = canvas.getBoundingClientRect();
    const S = 2.6;
    const x = ((e.clientX - r.left) / r.width) * 2 * S - S;
    const y = S - ((e.clientY - r.top) / r.height) * 2 * S;
    const c = other ? 1 - state.paint : state.paint;
    // every few clicks a point goes to the test set, so there is something to generalise to
    if (rng.next() < state.testFrac) test.push({ x, y, c });
    else train.push({ x, y, c });
    draw();
  };
  canvas.addEventListener('click', (e) => addPoint(e, e.shiftKey));
  canvas.addEventListener('contextmenu', (e) => {
    e.preventDefault();
    addPoint(e, true);
  });

  const playBtn = btn({ icon: 'play', label: 'Train', cls: 'sm primary' });
  let raf = 0;
  const loop = () => {
    if (!state.running) return;
    for (let k = 0; k < state.speed; k++) epoch();
    draw();
    raf = requestAnimationFrame(loop);
  };
  playBtn.addEventListener('click', () => {
    state.running = !state.running;
    playBtn.replaceChildren(...btn({ icon: state.running ? 'pause' : 'play', label: state.running ? 'Pause' : 'Train' }).childNodes);
    playBtn.className = 'btn sm ' + (state.running ? 'warn' : 'primary');
    if (state.running) raf = requestAnimationFrame(loop);
  });

  const lbl = (t: string) => h('span', { class: 'small muted' }, t);
  const body = h(
    'div',
    { style: { display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1.1fr)', gap: '18px' } },
    h(
      'div',
      { class: 'col' },
      canvas,
      h('div', { class: 'row', style: { flexWrap: 'wrap' } }, lbl('add points'), paint.el, btn({ icon: 'trash', label: 'Clear points', cls: 'xs', onclick: () => ((train = []), (test = []), draw()) })),
      drawHint,
      h('div', { class: 'note', style: { margin: 0 } }, 'Filled dots = training data, hollow = test data. The background is what the network predicts everywhere; the boundary is where it is 50/50.'),
    ),
    h(
      'div',
      { class: 'col', style: { gap: '12px' } },
      h('div', { class: 'card explain', style: { fontSize: '12.5px' } }, h('b', null, 'Supervised learning: '), 'every point comes with the right answer (its colour). The network outputs a probability p, the loss −[y·log p + (1−y)·log(1−p)] says how wrong it is, and gradient descent nudges every weight to be a bit less wrong. The robots learn the same way, except reinforcement learning has to discover the "right answers" by trial and error.'),
      h('div', { class: 'row', style: { flexWrap: 'wrap' } }, playBtn, btn({ icon: 'step', label: '1 epoch', cls: 'sm', onclick: () => (epoch(), draw()) }), btn({ icon: 'reset', label: 'New weights', cls: 'sm', onclick: () => rebuild() }), btn({ icon: 'shuffle', label: 'New data', cls: 'sm', onclick: () => regen() }), speedSel),
      h('div', { class: 'row', style: { flexWrap: 'wrap' } }, lbl('data'), dsSel, sizeSel, splitSel, lbl('noise'), noise),
      h('div', { class: 'col', style: { gap: '4px' } }, lbl('input features (try x² and y² on the circle!)'), featHost),
      h('div', { class: 'col', style: { gap: '4px' } }, lbl('hidden layers'), layerHost, archEl),
      h('div', { class: 'row', style: { flexWrap: 'wrap' } }, lbl('learning'), lrSel, optSel, batchSel, l2Sel),
      chart.el,
      h('div', { class: 'card' }, stats),
    ),
  );
  renderLayers();
  regen();
  modal('Neuron Lab: supervised learning playground', body, {
    size: 'wide',
    onClose: () => {
      state.running = false;
      cancelAnimationFrame(raf);
    },
  });
}
