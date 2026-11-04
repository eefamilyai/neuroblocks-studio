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
import { fmt, h, modal } from './dom.ts';

type DatasetKind = 'circle' | 'xor' | 'spiral' | 'blobs' | 'moons';
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

  const canvas = h('canvas', { width: '440', height: '440', style: { width: '100%', maxWidth: '440px', aspectRatio: '1', borderRadius: '12px', border: '1px solid var(--line2)', background: '#0f1324', display: 'block' } }) as HTMLCanvasElement;
  const chart = new LineChart('Loss (lower is better)', 'Binary cross-entropy on the training points (the network learns from these) and on the TEST points (never used for learning). When test loss goes up while train loss goes down, the network is memorising: overfitting.');
  const stats = h('div', { class: 'kv' });
  const archEl = h('div', { class: 'small muted' });
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
    const all = makeData(state.dataset, 400, state.noise, rng);
    rng.shuffle(all);
    train = all.slice(0, 280);
    test = all.slice(280);
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
    const idx = train.map((_, i) => i);
    rng.shuffle(idx);
    const B = Math.max(1, state.batch);
    for (let s = 0; s < idx.length; s += B) {
      const batch = idx.slice(s, s + B).map((i) => train[i]);
      const X = inputsOf(batch);
      opt.zeroGrad();
      const out = net.forward(X, batch.length);
      const d = new Float32Array(batch.length);
      // sigmoid + cross-entropy: dL/dz = p − y  (the nicest gradient in ML)
      batch.forEach((p, i) => (d[i] = (sig(out[i]) - p.c) / batch.length));
      net.backward(d, batch.length);
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
    chart.setSeries([
      { name: 'train', color: '#2fd182', xs: lossX, ys: lossTrain, raw: true },
      { name: 'test', color: '#ff6fae', xs: lossX, ys: lossTest, raw: true },
    ]);
    chart.draw();
    stats.innerHTML = '';
    const row = (k: string, v: string) => stats.append(h('span', null, k), h('b', null, v));
    row('Epoch', String(state.epoch));
    row('Train loss / accuracy', `${fmt(tr.loss, 3)} / ${(tr.acc * 100).toFixed(0)}%`);
    row('Test loss / accuracy', `${fmt(te.loss, 3)} / ${(te.acc * 100).toFixed(0)}%`);
  }

  function renderLayers(): void {
    layerHost.innerHTML = '';
    state.layers.forEach((l, i) => {
      const units = h('input', { type: 'number', min: '1', max: '64', value: String(l.units), class: 'sel', style: { width: '60px' } }) as HTMLInputElement;
      units.addEventListener('change', () => {
        l.units = Math.max(1, Math.min(64, Number(units.value) || 1));
        rebuild();
      });
      const act = h('select', { class: 'sel' }) as HTMLSelectElement;
      for (const a of ACTIVATIONS) act.append(h('option', { value: a, selected: a === l.activation }, a));
      act.addEventListener('change', () => {
        l.activation = act.value as Activation;
        rebuild();
      });
      layerHost.append(
        h(
          'div',
          { class: 'row', style: { marginBottom: '4px' } },
          h('span', { class: 'small muted', style: { width: '58px' } }, `layer ${i + 1}`),
          units,
          act,
          h(
            'button',
            {
              class: 'btn tiny',
              onclick: () => {
                state.layers.splice(i, 1);
                renderLayers();
                rebuild();
              },
            },
            '✕',
          ),
        ),
      );
    });
    layerHost.append(
      h(
        'button',
        {
          class: 'btn tiny',
          onclick: () => {
            if (state.layers.length < 5) state.layers.push({ units: 4, activation: 'tanh' });
            renderLayers();
            rebuild();
          },
        },
        '＋ add layer',
      ),
    );
  }

  const dsSel = h('select', { class: 'sel' }) as HTMLSelectElement;
  for (const [v, l] of [
    ['circle', '⭕ circle'],
    ['xor', '✖️ XOR'],
    ['spiral', '🌀 spiral'],
    ['blobs', '🫧 two blobs'],
    ['moons', '🌙 moons'],
  ])
    dsSel.append(h('option', { value: v }, l));
  dsSel.addEventListener('change', () => {
    state.dataset = dsSel.value as DatasetKind;
    regen();
  });
  const noise = h('input', { type: 'range', min: '0', max: '0.6', step: '0.02', value: String(state.noise) }) as HTMLInputElement;
  noise.addEventListener('change', () => {
    state.noise = Number(noise.value);
    regen();
  });
  const lrSel = h('select', { class: 'sel' }) as HTMLSelectElement;
  for (const v of [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]) lrSel.append(h('option', { value: String(v), selected: v === state.lr }, String(v)));
  lrSel.addEventListener('change', () => {
    state.lr = Number(lrSel.value);
    opt.lr = state.lr;
  });
  const optSel = h('select', { class: 'sel' }) as HTMLSelectElement;
  for (const [v, l] of [
    ['adam', 'Adam'],
    ['sgd', 'SGD'],
    ['momentum', 'Momentum'],
    ['rmsprop', 'RMSProp'],
  ])
    optSel.append(h('option', { value: v }, l));
  optSel.addEventListener('change', () => {
    state.optimizer = optSel.value as OptimizerKind;
    opt = makeOptimizer(state.optimizer, net.params(), state.lr);
  });
  const featHost = h('div', { class: 'row' });
  for (const f of FEATURES) {
    const b = h('button', { class: 'btn tiny' + (state.features.has(f) ? ' on' : '') }, f);
    b.addEventListener('click', () => {
      if (state.features.has(f)) state.features.delete(f);
      else state.features.add(f);
      b.classList.toggle('on', state.features.has(f));
      rebuild();
    });
    featHost.append(b);
  }
  const playBtn = h('button', { class: 'btn primary small' }, '▶ Train') as HTMLButtonElement;
  let raf = 0;
  const loop = () => {
    if (!state.running) return;
    for (let k = 0; k < 3; k++) epoch();
    draw();
    raf = requestAnimationFrame(loop);
  };
  playBtn.addEventListener('click', () => {
    state.running = !state.running;
    playBtn.textContent = state.running ? '⏸ Pause' : '▶ Train';
    playBtn.className = 'btn small ' + (state.running ? 'warn' : 'primary');
    if (state.running) raf = requestAnimationFrame(loop);
  });

  const body = h(
    'div',
    { style: { display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1.1fr)', gap: '16px' } },
    h(
      'div',
      null,
      canvas,
      h('div', { class: 'small muted', style: { marginTop: '6px' } }, 'Filled dots = training data, hollow = test data. Background = what the network predicts everywhere (blue vs orange). The boundary is where it is 50/50.'),
    ),
    h(
      'div',
      { style: { display: 'flex', flexDirection: 'column', gap: '10px' } },
      h(
        'div',
        { class: 'card explain' },
        h('p', { style: { marginTop: 0 } }, h('b', null, 'Supervised learning: '), 'every point comes with the right answer (its colour). The network outputs a probability p, the loss −[y·log p + (1−y)·log(1−p)] says how wrong it is, and gradient descent nudges every weight to be a bit less wrong. The robots learn the same way, except reinforcement learning has to discover the "right answers" by trial and error.'),
      ),
      h('div', { class: 'row' }, playBtn, h('button', { class: 'btn small', onclick: () => (epoch(), draw()) }, '⏭ 1 epoch'), h('button', { class: 'btn small', onclick: () => rebuild() }, '⟲ New weights'), h('button', { class: 'btn small', onclick: () => regen() }, '🎲 New data')),
      h('div', { class: 'row' }, h('span', { class: 'small' }, 'dataset'), dsSel, h('span', { class: 'small' }, 'noise'), noise),
      h('div', null, h('div', { class: 'small', style: { marginBottom: '4px' } }, 'input features (try x² and y² on the circle!)'), featHost),
      h('div', null, h('div', { class: 'small', style: { marginBottom: '4px' } }, 'hidden layers'), layerHost, archEl),
      h('div', { class: 'row' }, h('span', { class: 'small' }, 'learning rate'), lrSel, h('span', { class: 'small' }, 'optimizer'), optSel),
      chart.el,
      h('div', { class: 'card' }, stats),
    ),
  );
  renderLayers();
  regen();
  modal('🧪 Neuron Lab: supervised learning playground', body, {
    onClose: () => {
      state.running = false;
      cancelAnimationFrame(raf);
    },
  });
}
