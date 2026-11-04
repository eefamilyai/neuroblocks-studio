/**
 * Code mode and the new controls: blocks → code round trips, error
 * reporting with line numbers, infinite-loop protection, frame stacking,
 * physics settings, learning-rate schedules, stop conditions, evaluation.
 */
import { compileCode } from '../src/code/runtime.ts';
import { blocksToCode, CODE_TEMPLATE } from '../src/code/fromBlocks.ts';
import { guardLoops } from '../src/code/guard.ts';
import { MISSIONS } from '../src/blocks/missions.ts';
import { Trainer } from '../src/ml/algos/trainer.ts';
import { Evaluator } from '../src/ml/algos/evaluate.ts';
import { Env, compileScript, obsLayout } from '../src/sim/env.ts';
import { WorldRuntime } from '../src/sim/runtime.ts';
import { makeSpec } from '../src/sim/spec.ts';
import { getWorld } from '../src/sim/worldCache.ts';
import { TrainingHost } from '../src/worker/host.ts';
import type { FromWorker } from '../src/worker/protocol.ts';
import { compileMission } from './missions.test.ts';
import { REACH_FLAG } from './learn.test.ts';
import { test, assert } from './harness.ts';

const strip = (s: object) => JSON.stringify(s);

for (const m of MISSIONS) {
  test(`mission "${m.id}" converts to code with the same settings`, () => {
    const b = compileMission(m.id);
    const code = blocksToCode(b);
    const c = compileCode(code);
    assert(c.errors.length === 0, `code errors: ${c.errors.join(' | ')}\n${code}`);
    const keys = ['zone', 'maxSteps', 'sensors', 'actionMode', 'axes', 'brain', 'train', 'spawn', 'body', 'frameStack', 'physics', 'world'] as const;
    for (const k of keys) {
      const a = strip({ v: b.spec[k] ?? null });
      const z = strip({ v: c.spec[k] ?? null });
      assert(a === z, `${k} differs:\n blocks ${a}\n code   ${z}`);
    }
    assert(c.obsSize === b.obsSize && c.outSize === b.outSize && c.params === b.params, 'sizes differ');
    // the code version runs in the simulator without script errors
    const errs: string[] = [];
    const tr = new Trainer({ spec: c.spec, world: getWorld(), onError: (e) => errs.push(e) });
    while (tr.totalSteps < 1500) tr.runFor(50);
    assert(errs.length === 0, 'script errors: ' + errs.join(' | '));
  });
}

test('the code template compiles cleanly and the robot learns to reach the flag', () => {
  const c = compileCode(CODE_TEMPLATE);
  assert(c.errors.length === 0 && c.warnings.length === 0, `problems: ${[...c.errors, ...c.warnings].join(' | ')}`);
  const tr = new Trainer({ spec: c.spec, world: getWorld(), onError: (e) => console.log('ERR', e) });
  while (tr.totalSteps < 60000) tr.runFor(200);
  const eps = tr.algo.stats.episodes.slice(-100);
  const sr = eps.filter((e) => e.success).length / Math.max(1, eps.length);
  console.log(`      success ${(sr * 100).toFixed(0)}% after ${tr.totalSteps} steps`);
  assert(sr > 0.7, `success only ${sr}`);
  // reward labels arrive with every episode
  const parts = eps.at(-1)?.parts ?? {};
  assert('closer to flag' in parts && 'time' in parts, `reward parts: ${JSON.stringify(parts)}`);
});

test('code errors point at the right line', () => {
  const syn = compileCode("robot({ zone: 'meadow' });\nsense.target('flag');\nact.move(;\n");
  assert(syn.errors.length === 1 && syn.diagnostics![0].line === 3, `syntax: ${JSON.stringify(syn.diagnostics)}`);
  const rt = compileCode("robot({ zone: 'meadow' });\n\nconst x = null;\nx.boom();\n");
  assert(rt.diagnostics!.some((d) => d.severity === 'error' && d.line === 4), `runtime: ${JSON.stringify(rt.diagnostics)}`);
  const zone = compileCode("robot({ zone: 'medow' });\nsense.target('flag'); act.move();\n");
  const d = zone.diagnostics!.find((q) => q.message.includes('medow'));
  assert(!!d && d.line === 1 && d.message.includes("'meadow'"), `zone: ${JSON.stringify(zone.diagnostics)}`);
  const tag = compileCode("sense.target('flgg');\nact.move();\n");
  assert(tag.diagnostics!.some((q) => q.line === 1 && q.message.includes("'flag'")), `tag: ${JSON.stringify(tag.diagnostics)}`);
  const opt = compileCode("sense.target('flag'); act.move();\ntrain.ppo({ lr: 0.001, clipp: 0.3 });\n");
  assert(opt.diagnostics!.some((q) => q.line === 2 && q.severity === 'warning' && q.message.includes("'clip'")), `option: ${JSON.stringify(opt.diagnostics)}`);
});

test('errors inside event handlers report the user line during training', () => {
  const c = compileCode("sense.target('flag');\nact.move();\non.step(() => {\n  api.reward(1);\n  undefinedThing.x = 1;\n});\n");
  assert(c.errors.length === 0, c.errors.join(' | '));
  const errs: string[] = [];
  const tr = new Trainer({ spec: c.spec, world: getWorld(), onError: (e) => errs.push(e) });
  tr.runFor(30);
  assert(errs.length > 0 && errs[0].includes('line 5'), `got: ${errs.join(' | ')}`);
});

test('infinite loops are stopped instead of freezing the page', () => {
  const top = compileCode("sense.target('flag'); act.move();\nwhile (true) {}\n");
  assert(top.errors.some((e) => e.includes('infinite loop')), `top-level: ${top.errors.join(' | ')}`);
  const inHandler = compileCode("sense.target('flag'); act.move();\non.step(() => { let i = 0; while (i >= 0) i++; });\n");
  assert(inHandler.errors.length === 0, inHandler.errors.join(' | '));
  const errs: string[] = [];
  const env = new Env(new WorldRuntime(getWorld()), inHandler.spec, compileScript(inHandler.spec.script), 1);
  env.onError = (e) => errs.push(e);
  env.reset();
  const t0 = performance.now();
  env.step([1]);
  assert(performance.now() - t0 < 5000 && errs.some((e) => e.includes('infinite loop')), `handler: ${errs.join(' | ')}`);
  // the guard keeps line numbers intact
  const g = guardLoops('a();\nfor (;;) { b(); }\nc();');
  assert(g.code.split('\n').length === 3, 'guard added lines');
});

test('frame stacking feeds the previous observations', () => {
  const spec = makeSpec({ zone: 'meadow', sensors: [{ kind: 'velocity' }, { kind: 'time' }], frameStack: 3, script: REACH_FLAG });
  const L = obsLayout(spec);
  assert(L.size === 12 && L.base === 4 && L.labels[4].includes('t−1'), `layout ${L.size} ${L.base} ${L.labels[4]}`);
  const env = new Env(new WorldRuntime(getWorld()), spec, compileScript(spec.script), 3);
  env.reset();
  const t0 = env.obs[3];
  assert(env.obs[7] === t0 && env.obs[11] === t0, 'reset fills every frame');
  env.step([2, 1]);
  const t1 = env.obs[3];
  env.step([2, 1]);
  assert(env.obs[7] === t1 && env.obs[11] === t0 && env.obs[3] < t1, `shift: ${Array.from(env.obs).join(',')}`);
});

test('physics settings change the world', () => {
  const run = (gravity: number) => {
    const spec = makeSpec({ zone: 'meadow', script: REACH_FLAG, physics: { gravity }, axes: [{ axis: 'jump', options: [0, 1], labels: ['stay', 'jump'] }] });
    const env = new Env(new WorldRuntime(getWorld()), spec, compileScript(spec.script), 5);
    env.reset();
    let top = env.a.y;
    for (let i = 0; i < 30; i++) {
      env.step([i === 2 ? 1 : 0]);
      top = Math.max(top, env.a.y);
    }
    return top;
  };
  const low = run(9.8);
  const high = run(40);
  assert(low > high + 0.5, `jump on the moon should be higher: g=9.8 → ${low.toFixed(2)}, g=40 → ${high.toFixed(2)}`);
  const slow = makeSpec({ zone: 'meadow', script: REACH_FLAG, physics: { decisionTicks: 6 } });
  assert(new Env(new WorldRuntime(getWorld()), slow, null, 1).P.substeps === 6, 'decision rate');
});

test('learning-rate schedules decay and the trainer keeps its own spec copy', () => {
  const spec = makeSpec({ zone: 'meadow', script: REACH_FLAG, train: { algo: 'ppo', lr: 1e-3, lrSchedule: 'linear', lrEnd: 0.1, scheduleSteps: 4096, entropy: 0.02, entropyEnd: 0 } });
  const tr = new Trainer({ spec, world: getWorld() });
  while (tr.totalSteps < 4200) tr.runFor(50);
  tr.runFor(1);
  assert(Math.abs(tr.spec.train.lr - 1e-4) < 1e-9, `lr at the end ${tr.spec.train.lr}`);
  assert(tr.spec.train.entropy === 0, `entropy ${tr.spec.train.entropy}`);
  assert(spec.train.lr === 1e-3, 'the original spec must not change');
});

test('stop conditions, single steps and evaluation in the training host', async () => {
  const spec = makeSpec({ zone: 'meadow', script: REACH_FLAG, maxSteps: 60, train: { algo: 'ppo', numEnvs: 8, horizon: 32, targetReward: -1e6 } });
  const msgs: FromWorker[] = [];
  const host = new TrainingHost((m) => msgs.push(m), 20);
  host.handle({ type: 'start', spec, edits: { objects: [], walls: [] }, init: null, mode: 'rl', demos: null, stepOnly: true });
  let guard = 0;
  while (host.tick() >= 0 && guard++ < 500);
  const paused = msgs.find((m) => m.type === 'paused');
  assert(!!paused, 'single step should pause by itself');
  const brain = [...msgs].reverse().find((m) => m.type === 'brain');
  assert(brain && brain.type === 'brain' && brain.snap.updates === 1, 'one update');
  // resume: the reward goal (-1e6) is met as soon as 100 episodes are done
  host.handle({ type: 'resume' });
  guard = 0;
  while (host.tick() >= 0 && guard++ < 5000);
  const fin = msgs.find((m) => m.type === 'finished');
  assert(!!fin && fin.type === 'finished' && fin.reason.includes('mean reward'), `finished: ${JSON.stringify(fin)}`);
  // evaluation of that brain
  const last = [...msgs].reverse().find((m) => m.type === 'brain');
  assert(last?.type === 'brain', 'a brain message');
  const snap = last.snap;
  const ev = new Evaluator(spec, getWorld(), undefined, snap, 20, true);
  while (!ev.work(50));
  const r = ev.result();
  assert(r.episodes === 20 && r.rewards.length === 20 && Number.isFinite(r.meanReward), `eval ${JSON.stringify(r).slice(0, 200)}`);
});

test('world.add / world.wall / clearZone build the level from code', () => {
  const c = compileCode("robot({ zone: 'meadow' });\nworld.clearZone();\nworld.add('coin', { x: -28, z: 2 });\nworld.wall({ x: -24, z: 0, w: 6 });\nsense.target('coin'); act.move();\n");
  assert(c.errors.length === 0, c.errors.join(' | '));
  const tr = new Trainer({ spec: c.spec, world: getWorld() });
  const env = tr.vec.envs[0];
  const types = env.objs.map((o) => o.type);
  assert(types.length === 1 && types[0] === 'coin', `objects: ${types.join(',')}`);
  assert(tr.rt.statics.length === getWorld().statics.length + 1, 'one extra wall');
});

test('your own algorithm: the REINFORCE template written in code mode learns', async () => {
  const { CUSTOM_ALGO_TEMPLATE } = await import('../src/code/templates.ts');
  const c = compileCode(CUSTOM_ALGO_TEMPLATE);
  assert(c.errors.length === 0, c.errors.join(' | '));
  assert(c.spec.train.algo === 'custom' && c.spec.train.numEnvs === 16, `algo ${c.spec.train.algo}`);
  const errs: string[] = [];
  const tr = new Trainer({ spec: c.spec, world: getWorld(), onError: (e) => errs.push(e) });
  const t0 = performance.now();
  while (tr.totalSteps < 120000) tr.runFor(200);
  const eps = tr.algo.stats.episodes.slice(-100);
  const sr = eps.filter((e) => e.success).length / Math.max(1, eps.length);
  const m = tr.algo.stats.metrics.at(-1);
  console.log(`      success ${(sr * 100).toFixed(0)}% after ${tr.totalSteps} steps, ${tr.algo.updates} updates, ${((performance.now() - t0) / 1000).toFixed(1)} s, last metrics ${JSON.stringify(m)}`);
  assert(errs.length === 0, errs.join(' | '));
  assert(sr > 0.5, `success only ${sr}`);
  const snap = tr.snapshot();
  assert(snap.algo === 'custom' && snap.kind === 'policy' && snap.net.inSize === c.obsSize, 'snapshot');
});

test('errors inside a custom algorithm name the function and line', () => {
  const src = "sense.target('flag'); act.move();\ntrain.custom({\n  setup(ctx) { this.policy = ctx.mlp([8]); },\n  act(obs, ctx) { return ctx.sample(this.policy.forward(obs)); },\n  update(ctx) {\n    nope();\n  },\n});\n";
  const c = compileCode(src);
  assert(c.errors.length === 0, c.errors.join(' | '));
  const tr = new Trainer({ spec: c.spec, world: getWorld() });
  let msg = '';
  try {
    tr.runFor(20);
  } catch (e) {
    msg = (e as Error).message;
  }
  assert(msg.includes('update()') && msg.includes('line 6'), `got: ${msg}`);
  const bad = compileCode("sense.target('flag'); act.move();\ntrain.custom({ act() { return 0; } });\n");
  assert(bad.errors.some((e) => e.includes('update')), bad.errors.join(' | '));
});
