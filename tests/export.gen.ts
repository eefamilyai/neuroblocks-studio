/** Generates export fixtures for tests/verify_export.py (run by `npm run test:export`). */
import { writeFileSync, mkdirSync } from 'node:fs';
import { compileMission } from './missions.test.ts';
import { Trainer } from '../src/ml/algos/trainer.ts';
import { buildWorld } from '../src/sim/worldgen.ts';
import { Brain } from '../src/ml/algos/common.ts';
import { exportOnnx } from '../src/export/onnx.ts';
import { pytorchScript } from '../src/export/pytorch.ts';
import { RNG } from '../src/ml/rng.ts';

const out = process.argv[2] ?? 'tests/.export';
mkdirSync(out, { recursive: true });
for (const [id, algo] of [['coin-collector', 'ppo'], ['reach-flag-dqn', 'dqn'], ['race-laps', 'ppo-cont'], ['reach-flag-evo', 'evo']] as const) {
  const r = compileMission(id);
  // mix of activations to exercise the exporter
  if (id === 'coin-collector') r.spec.brain.layers = [{ units: 24, activation: 'relu' }, { units: 16, activation: 'swish' }, { units: 12, activation: 'leaky_relu' }];
  if (id === 'race-laps') r.spec.brain.layers = [{ units: 20, activation: 'elu' }, { units: 20, activation: 'softplus' }];
  const tr = new Trainer({ spec: r.spec, world: buildWorld() });
  while (tr.totalSteps < 6000) tr.runFor(100);
  const snap = tr.snapshot();
  const brain = new Brain(snap);
  const rng = new RNG(5);
  const S = snap.net.inSize;
  const inputs: number[][] = [];
  const outputs: number[][] = [];
  const values: number[] = [];
  for (let k = 0; k < 20; k++) {
    const x = Float32Array.from({ length: S }, () => rng.gauss() * 2);
    inputs.push(Array.from(x));
    const nx = brain.normalize(x);
    outputs.push(Array.from(brain.net.forward(nx, 1).subarray(0, brain.net.outSize)));
    if (brain.critic) values.push(brain.critic.forward(nx, 1)[0]);
  }
  writeFileSync(`${out}/${algo}.onnx`, exportOnnx(snap, id));
  writeFileSync(`${out}/${algo}_brain.py`, pytorchScript(snap, id));
  writeFileSync(`${out}/${algo}.json`, JSON.stringify({ inputs, outputs, values, hasCritic: !!brain.critic }));
  console.log(`wrote ${algo}: obs=${S} out=${brain.net.outSize}`);
}
