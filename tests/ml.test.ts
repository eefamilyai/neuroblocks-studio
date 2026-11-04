/**
 * Numerical gradient checks: compare our hand-written backward passes with
 * finite differences  (f(w+h) − f(w−h)) / 2h.  If backprop is right, they agree.
 */
import { MLP, ACTIVATIONS } from '../src/ml/nn.ts';
import type { Activation } from '../src/ml/nn.ts';
import { RNG } from '../src/ml/rng.ts';
import { Adam } from '../src/ml/optim.ts';
import { PolicyHead } from '../src/ml/policy.ts';
import type { ActionSpace } from '../src/ml/policy.ts';
import { test, assert } from './harness.ts';

function relErr(a: number, b: number): number {
  return Math.abs(a - b) / Math.max(1e-6, Math.abs(a) + Math.abs(b));
}

for (const act of ACTIVATIONS) {
  test(`MLP gradient check (${act})`, () => {
    const rng = new RNG(42);
    const batch = 3;
    const net = new MLP(4, [{ units: 5, activation: act as Activation }, { units: 3, activation: act as Activation }], 2, 'linear', rng);
    const x = new Float32Array(batch * 4).map(() => rng.gauss());
    const target = new Float32Array(batch * 2).map(() => rng.gauss());
    // Loss = 0.5 * Σ (y - t)²
    const loss = () => {
      const y = net.forward(x, batch);
      let s = 0;
      for (let i = 0; i < batch * 2; i++) s += 0.5 * (y[i] - target[i]) ** 2;
      return s;
    };
    net.zeroGrad();
    const y = net.forward(x, batch);
    const dy = new Float32Array(batch * 2);
    for (let i = 0; i < batch * 2; i++) dy[i] = y[i] - target[i];
    const dx = Float32Array.from(net.backward(dy, batch, true).subarray(0, batch * 4));
    const h = 1e-3;
    let worst = 0;
    for (const p of net.params()) {
      for (let i = 0; i < p.w.length; i += 3) {
        const old = p.w[i];
        p.w[i] = old + h;
        const lp = loss();
        p.w[i] = old - h;
        const lm = loss();
        p.w[i] = old;
        const num = (lp - lm) / (2 * h);
        if (Math.abs(num) > 1e-4 || Math.abs(p.g[i]) > 1e-4) worst = Math.max(worst, relErr(num, p.g[i]));
      }
    }
    // input gradient
    for (let i = 0; i < x.length; i++) {
      const old = x[i];
      x[i] = old + h;
      const lp = loss();
      x[i] = old - h;
      const lm = loss();
      x[i] = old;
      const num = (lp - lm) / (2 * h);
      if (Math.abs(num) > 1e-4) worst = Math.max(worst, relErr(num, dx[i]));
    }
    // ReLU kinks can produce isolated mismatches; float32 gives ~1e-3 noise.
    assert(worst < 0.05, `worst relative error ${worst}`);
  });
}

function headCheck(space: ActionSpace) {
  const rng = new RNG(7);
  const head = new PolicyHead(space);
  const out = new Float32Array(head.outSize).map(() => rng.gauss());
  const logStd = space.kind === 'continuous' ? new Float32Array(head.dims).map(() => rng.uniform(-1, 0)) : null;
  const act = new Float32Array(head.dims);
  head.sample(out, 0, logStd, rng, act, 0, false);
  const cL = 0.7;
  const cE = 0.3;
  const f = () => cL * head.logProb(out, 0, logStd, act, 0) + cE * head.entropy(out, 0, logStd);
  const dOut = new Float32Array(head.outSize);
  const dLs = logStd ? new Float32Array(head.dims) : null;
  head.accumGrad(out, 0, logStd, act, 0, cL, cE, dOut, 0, dLs);
  const h = 1e-3;
  let worst = 0;
  for (let i = 0; i < out.length; i++) {
    const o = out[i];
    out[i] = o + h;
    const a = f();
    out[i] = o - h;
    const b = f();
    out[i] = o;
    worst = Math.max(worst, relErr((a - b) / (2 * h), dOut[i]));
  }
  if (logStd && dLs) {
    for (let i = 0; i < logStd.length; i++) {
      const o = logStd[i];
      logStd[i] = o + h;
      const a = f();
      logStd[i] = o - h;
      const b = f();
      logStd[i] = o;
      worst = Math.max(worst, relErr((a - b) / (2 * h), dLs[i]));
    }
  }
  assert(worst < 0.02, `worst relative error ${worst}`);
}

test('Policy head gradients (multi-discrete)', () =>
  headCheck({ kind: 'discrete', branches: [3, 3, 2], labels: ['move', 'turn', 'jump'], optionLabels: [] }));
test('Policy head gradients (continuous)', () =>
  headCheck({ kind: 'continuous', branches: [1, 1], labels: ['move', 'turn'], optionLabels: [] }));

test('MLP learns sin(x) with Adam', () => {
  const rng = new RNG(3);
  const net = new MLP(1, [{ units: 32, activation: 'tanh' }, { units: 32, activation: 'tanh' }], 1, 'linear', rng);
  const opt = new Adam(net.params(), 0.01);
  const B = 64;
  const x = new Float32Array(B);
  const dy = new Float32Array(B);
  let mse = 0;
  for (let it = 0; it < 1500; it++) {
    for (let i = 0; i < B; i++) x[i] = rng.uniform(-3, 3);
    opt.zeroGrad();
    const y = net.forward(x, B);
    mse = 0;
    for (let i = 0; i < B; i++) {
      const e = y[i] - Math.sin(x[i]);
      mse += e * e;
      dy[i] = (2 * e) / B;
    }
    mse /= B;
    net.backward(dy, B);
    opt.step();
  }
  assert(mse < 0.01, `final mse ${mse}`);
});
