/**
 * The real test: does each algorithm actually LEARN the beginner task?
 * Runs headless in Node: no browser needed.
 */
import { buildWorld } from '../src/sim/worldgen.ts';
import { Trainer } from '../src/ml/algos/trainer.ts';
import { makeSpec } from '../src/sim/spec.ts';
import type { AlgoKind } from '../src/sim/types.ts';
import { test, assert } from './harness.ts';

export const REACH_FLAG = `function(api, Math){
  var S = {start:[],step:[],touch:[],pair:[],fall:[],end:[],senses:[]};
  S.start.push(function(){ api.randomize('flag'); });
  S.step.push(function(){ api.reward(-api.delta('flag') * 0.5); api.reward(-0.002); });
  S.touch.push(['flag', function(obj){ api.reward(1); api.end(true); }]);
  return S; }`;

function trainFor(algo: AlgoKind, maxSteps: number, extra: Record<string, unknown> = {}) {
  const spec = makeSpec({
    zone: 'meadow',
    script: REACH_FLAG,
    maxSteps: 300,
    spawn: { mode: 'random', x: 0, z: 0, yawMode: 'random', yaw: 0, jitter: 0 },
    brain: { layers: [{ units: 32, activation: 'tanh' }, { units: 32, activation: 'tanh' }] },
    train: { algo, numEnvs: 16, ...extra },
  });
  const tr = new Trainer({ spec, world: buildWorld(), onError: (m) => console.log('ERR', m) });
  const t0 = performance.now();
  let lastLog = 0;
  while (tr.totalSteps < maxSteps) {
    tr.runFor(200);
    if (tr.totalSteps - lastLog > maxSteps / 5) {
      lastLog = tr.totalSteps;
      const st = tr.algo.stats;
      const recent = st.episodes.slice(-50);
      const succ = recent.filter((e) => e.success).length / Math.max(1, recent.length);
      console.log(`      [${algo}] steps=${tr.totalSteps} meanR=${st.meanRecent().toFixed(3)} success=${(succ * 100).toFixed(0)}% ${(tr.totalSteps / ((performance.now() - t0) / 1000)).toFixed(0)} steps/s`);
    }
  }
  const recent = tr.algo.stats.episodes.slice(-60);
  const succ = recent.filter((e) => e.success).length / Math.max(1, recent.length);
  return { succ, tr, secs: (performance.now() - t0) / 1000 };
}

test('PPO learns to reach the flag', () => {
  const { succ, secs } = trainFor('ppo', 120_000);
  console.log(`      PPO success ${(succ * 100).toFixed(0)}% in ${secs.toFixed(1)}s`);
  assert(succ > 0.8, `success ${succ}`);
});

test('DQN learns to reach the flag', () => {
  const { succ, secs } = trainFor('dqn', 100_000, { learnStart: 1000, epsDecay: 40000 });
  console.log(`      DQN success ${(succ * 100).toFixed(0)}% in ${secs.toFixed(1)}s`);
  assert(succ > 0.6, `success ${succ}`);
});

test('Evolution learns to reach the flag', () => {
  const { succ, secs } = trainFor('evolution', 150_000, { population: 32, numEnvs: 16 });
  console.log(`      Evolution success ${(succ * 100).toFixed(0)}% in ${secs.toFixed(1)}s`);
  assert(succ > 0.4, `success ${succ}`);
});

test('REINFORCE learns to reach the flag', () => {
  const { succ, secs } = trainFor('reinforce', 150_000, { lr: 3e-3 });
  console.log(`      REINFORCE success ${(succ * 100).toFixed(0)}% in ${secs.toFixed(1)}s`);
  assert(succ > 0.5, `success ${succ}`);
});
