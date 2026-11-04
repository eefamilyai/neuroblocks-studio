/**
 * Code mode: write the robot's program in JavaScript instead of blocks.
 *
 * A program is plain JavaScript that calls a few "builder" functions, one
 * for every kind of block:
 *
 *   robot({ zone: 'meadow', maxSteps: 300 });      🤖 robot block
 *   sense.target('flag');                          👁️ senses
 *   act.move(); act.turn();                        🕹️ actions
 *   brain.layer(32, 'tanh');                       🧠 brain
 *   train.ppo({ lr: 3e-4 });                       🏋️ training
 *   on.touch('flag', () => api.reward(1));         ⚡ events
 *
 * The program runs twice:
 *  1. here, once (the "config pass"), to read the settings into an AgentSpec;
 *  2. inside every training world (worker + 3D view), where the builders do
 *     nothing except register the event handlers and custom senses.
 */
import { guardLoops, lineStart } from './guard.ts';
import type { CompileResult } from '../blocks/compiler.ts';
import { AXIS_PRESETS, sensorCost } from '../sim/spec.ts';
import { ACTIVATIONS } from '../ml/nn.ts';
import type { Activation } from '../ml/nn.ts';
import type { ActionAxis, AgentSpec, AlgoKind, CustomWall, ObjDef, ObjType, PhysicsSpec, SensorSpec, Tag, TrainSpec } from '../sim/types.ts';
import { DEFAULT_PHYSICS, OBJ_TYPES, PHYSICS_LIMITS, STATIC_TAGS, defaultTrain } from '../sim/types.ts';
import { getWorld } from '../sim/worldCache.ts';
import { zoneAt } from '../sim/runtime.ts';

export interface CodeDiagnostic {
  from: number;
  to: number;
  line: number;
  severity: 'error' | 'warning';
  message: string;
}

/** Names visible to a program (besides `api`, `Math` and `console`). */
export const BUILDER_NAMES = ['robot', 'sense', 'actions', 'act', 'brain', 'train', 'world', 'on'] as const;

const TARGET_TAGS = [...OBJ_TYPES, ...STATIC_TAGS, 'next_checkpoint', 'spawn'];
const TOUCH_TAGS: string[] = [...OBJ_TYPES, ...STATIC_TAGS];

// ---------------------------------------------------------------------------
// The builders as they exist inside every training world: no-ops, except the
// ones that register handlers. Kept on ONE line so line numbers stay simple.
// ---------------------------------------------------------------------------
const ENV_PRELUDE = [
  'var __nb = function () { return __nb; };',
  'var robot = __nb, brain = { layer: __nb, layers: __nb }, train = { ppo: __nb, dqn: __nb, reinforce: __nb, evolution: __nb };',
  'var world = { physics: __nb, add: __nb, wall: __nb, clearZone: __nb };',
  'var act = { move: __nb, turn: __nb, strafe: __nb, jump: __nb, kick: __nb };',
  'var actions = function () { return actions; }; actions.discrete = actions; actions.continuous = actions;',
  'var sense = { rays: __nb, target: __nb, pair: __nb, velocity: __nb, heading: __nb, position: __nb, grounded: __nb, height: __nb, groundProbes: __nb, timeLeft: __nb, lastAction: __nb, memory: __nb,',
  '  custom: function (name, fn) { S.senses.push(typeof fn === "function" ? fn : function () { return Number(fn) || 0; }); return sense; } };',
  'var on = { start: function (f) { S.start.push(f); return on; }, step: function (f) { S.step.push(f); return on; }, fall: function (f) { S.fall.push(f); return on; }, end: function (f) { S.end.push(f); return on; },',
  '  touch: function (t, f) { S.touch.push([String(t), f]); return on; }, pair: function (a, b, f) { S.pair.push([String(a), String(b), f]); return on; } };',
  'var console = { log: function () { api.log(Array.prototype.slice.call(arguments).join(" ")); } }; console.info = console.warn = console.error = console.log;',
  'function __loopLimit() { throw new Error("a loop ran more than a million times (infinite loop?)"); }',
]
  .join(' ')
  .replace(/\s+/g, ' ');

/** The factory source the Env compiles (see compileScript in env.ts), and where the user's code starts. */
export function factorySource(guardedUserCode: string): { src: string; lineOffset: number } {
  const head = ['function (api, Math) {', 'var S = { start: [], step: [], touch: [], pair: [], fall: [], end: [], senses: [] };', ENV_PRELUDE, '{'];
  return { src: `${head.join('\n')}\n${guardedUserCode}\n}\nreturn S;\n}`, lineOffset: head.length };
}

// ---------------------------------------------------------------------------
// Line numbers of errors thrown during the config pass
// ---------------------------------------------------------------------------
let configOffset: number | null | undefined;
const anonLines = (e: unknown): number[] => {
  const st = (e as Error)?.stack;
  if (typeof st !== 'string') return [];
  return [...st.matchAll(/(?:<anonymous>|Function|anonymous):(\d+):(\d+)/g)].map((m) => Number(m[1]));
};
function configLineOffset(): number | null {
  if (configOffset !== undefined) return configOffset;
  try {
    new Function('a', 'throw new Error("probe")')(0);
  } catch (e) {
    const l = anonLines(e)[0];
    configOffset = l === undefined ? null : l - 1;
    return configOffset;
  }
  configOffset = null;
  return null;
}
function errorLine(e: unknown): number | null {
  const off = configLineOffset();
  const l = anonLines(e)[0];
  return off === null || l === undefined ? null : Math.max(1, l - off);
}
/** The config pass puts one line ("use strict") in front of the user's code. */
function userLineOf(e: unknown): number | null {
  const l = errorLine(e);
  return l === null ? null : Math.max(1, l - 1);
}
/** Line of the user code that called the builder that is running now. */
function callerLine(): number | null {
  return userLineOf(new Error());
}

// ---------------------------------------------------------------------------
// Small validation helpers
// ---------------------------------------------------------------------------
function editDistance(a: string, b: string): number {
  const d: number[] = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    let prev = d[0];
    d[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const tmp = d[j];
      d[j] = Math.min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] === b[j - 1] ? 0 : 1));
      prev = tmp;
    }
  }
  return d[b.length];
}
function suggest(v: string, options: readonly string[]): string {
  let best = '';
  let bd = Infinity;
  for (const o of options) {
    const dd = editDistance(v.toLowerCase(), o);
    if (dd < bd) {
      bd = dd;
      best = o;
    }
  }
  return bd <= Math.max(2, v.length / 3) ? ` Did you mean '${best}'?` : ` Options: ${options.slice(0, 14).join(', ')}${options.length > 14 ? ', …' : ''}.`;
}

/** Keys accepted by train.xxx({...}) and their limits. Aliases map friendlier names to TrainSpec keys. */
const TRAIN_KEYS: Record<string, { key: keyof TrainSpec; lo?: number; hi?: number; int?: boolean; bool?: boolean; algos?: AlgoKind[] }> = {
  envs: { key: 'numEnvs', lo: 1, hi: 64, int: true },
  numEnvs: { key: 'numEnvs', lo: 1, hi: 64, int: true },
  seed: { key: 'seed', lo: 0, hi: 1e9, int: true },
  lr: { key: 'lr', lo: 1e-7, hi: 1 },
  learningRate: { key: 'lr', lo: 1e-7, hi: 1 },
  gamma: { key: 'gamma', lo: 0, hi: 1 },
  lambda: { key: 'lambda', lo: 0, hi: 1, algos: ['ppo'] },
  clip: { key: 'clip', lo: 0.01, hi: 1, algos: ['ppo'] },
  entropy: { key: 'entropy', lo: 0, hi: 1, algos: ['ppo', 'reinforce'] },
  epochs: { key: 'epochs', lo: 1, hi: 50, int: true, algos: ['ppo'] },
  minibatch: { key: 'minibatch', lo: 16, hi: 8192, int: true, algos: ['ppo'] },
  horizon: { key: 'horizon', lo: 8, hi: 4096, int: true, algos: ['ppo'] },
  valueCoef: { key: 'valueCoef', lo: 0, hi: 10, algos: ['ppo'] },
  maxGradNorm: { key: 'maxGradNorm', lo: 0.01, hi: 100 },
  normObs: { key: 'normObs', bool: true },
  normReward: { key: 'normReward', bool: true, algos: ['ppo'] },
  bufferSize: { key: 'bufferSize', lo: 1000, hi: 1e6, int: true, algos: ['dqn'] },
  buffer: { key: 'bufferSize', lo: 1000, hi: 1e6, int: true, algos: ['dqn'] },
  batchSize: { key: 'batchSize', lo: 8, hi: 1024, int: true, algos: ['dqn'] },
  batch: { key: 'batchSize', lo: 8, hi: 1024, int: true, algos: ['dqn'] },
  epsStart: { key: 'epsStart', lo: 0, hi: 1, algos: ['dqn'] },
  epsEnd: { key: 'epsEnd', lo: 0, hi: 1, algos: ['dqn'] },
  epsDecay: { key: 'epsDecay', lo: 1, hi: 1e8, int: true, algos: ['dqn'] },
  targetUpdate: { key: 'targetUpdate', lo: 1, hi: 1e7, int: true, algos: ['dqn'] },
  learnStart: { key: 'learnStart', lo: 0, hi: 1e7, int: true, algos: ['dqn'] },
  trainEvery: { key: 'trainEvery', lo: 1, hi: 1000, int: true, algos: ['dqn'] },
  double: { key: 'double', bool: true, algos: ['dqn'] },
  episodesPerUpdate: { key: 'episodesPerUpdate', lo: 1, hi: 1000, int: true, algos: ['reinforce'] },
  baseline: { key: 'baseline', bool: true, algos: ['reinforce'] },
  population: { key: 'population', lo: 4, hi: 1000, int: true, algos: ['evolution'] },
  eliteFrac: { key: 'eliteFrac', lo: 0.01, hi: 1, algos: ['evolution'] },
  elite: { key: 'eliteFrac', lo: 0.01, hi: 1, algos: ['evolution'] },
  mutation: { key: 'mutation', lo: 0.001, hi: 5, algos: ['evolution'] },
  sigma: { key: 'mutation', lo: 0.001, hi: 5, algos: ['evolution'] },
  mutationDecay: { key: 'mutationDecay', lo: 0.5, hi: 1, algos: ['evolution'] },
  episodesPerEval: { key: 'episodesPerEval', lo: 1, hi: 20, int: true, algos: ['evolution'] },
  lrEnd: { key: 'lrEnd', lo: 0, hi: 1 },
  entropyEnd: { key: 'entropyEnd', lo: 0, hi: 1, algos: ['ppo', 'reinforce'] },
  scheduleSteps: { key: 'scheduleSteps', lo: 0, hi: 1e10, int: true },
  stopSteps: { key: 'stopSteps', lo: 0, hi: 1e10, int: true },
  stopAfter: { key: 'stopSteps', lo: 0, hi: 1e10, int: true },
  targetSuccess: { key: 'targetSuccess', lo: 0, hi: 1 },
  targetReward: { key: 'targetReward', lo: -1e9, hi: 1e9 },
  stopMinutes: { key: 'stopMinutes', lo: 0, hi: 1e5 },
};

// ---------------------------------------------------------------------------
// The config pass
// ---------------------------------------------------------------------------

export function compileCode(source: string, customObjects: ObjDef[] = []): CompileResult {
  const world = getWorld();
  const zoneIds = world.zones.map((z) => z.id);
  const errors: string[] = [];
  const warnings: string[] = [];
  const diagnostics: CodeDiagnostic[] = [];
  const lineRange = (line: number | null): [number, number] => {
    if (!line) return [0, Math.min(source.length, Math.max(1, source.indexOf('\n') >= 0 ? source.indexOf('\n') : source.length))];
    const from = lineStart(source, line);
    const nl = source.indexOf('\n', from);
    return [from, nl < 0 ? source.length : nl];
  };
  const report = (severity: 'error' | 'warning', message: string, line: number | null = callerLine()) => {
    const [from, to] = lineRange(line);
    diagnostics.push({ from, to, line: line ?? 1, severity, message });
    (severity === 'error' ? errors : warnings).push(line ? `line ${line}: ${message}` : message);
  };

  const spec: AgentSpec = {
    name: 'Robo',
    zone: 'meadow',
    // yaw π = facing north, the same default as the robot block (angle 0°)
    spawn: { mode: 'zone', x: 0, z: 0, yawMode: 'default', yaw: Math.PI, jitter: 1.5 },
    body: { speed: 1, turn: 1, jump: 1, color: '#4fc3f7', name: 'Robo' },
    sensors: [],
    actionMode: 'discrete',
    axes: [],
    brain: { layers: [] },
    train: defaultTrain('ppo'),
    maxSteps: 400,
    script: '',
    customSenses: [],
    customObjects,
    warnings,
    source: 'code',
    physics: {},
    frameStack: 1,
    world: { objects: [], walls: [], clearZones: [] },
  };
  let trainCalls = 0;
  let robotCalls = 0;
  const handlerCounts = { start: 0, step: 0, touch: 0, pair: 0, fall: 0, end: 0 };

  const num = (v: unknown, d: number, lo: number, hi: number, what: string, int = false): number => {
    if (v === undefined) return d;
    const n = Number(v);
    if (!Number.isFinite(n)) {
      report('warning', `${what} should be a number (got ${JSON.stringify(v)}); using ${d}.`);
      return d;
    }
    const c = Math.min(hi, Math.max(lo, int ? Math.round(n) : n));
    if (c !== (int ? Math.round(n) : n)) report('warning', `${what} = ${n} is outside ${lo}…${hi}; using ${c}.`);
    return c;
  };
  const obj = (v: unknown, what: string): Record<string, unknown> => {
    if (v === undefined || v === null) return {};
    if (typeof v !== 'object' || Array.isArray(v)) {
      report('warning', `${what} expects an options object like { … } (ignored).`);
      return {};
    }
    return v as Record<string, unknown>;
  };
  const known = (o: Record<string, unknown>, keys: string[], what: string) => {
    for (const k of Object.keys(o)) if (!keys.includes(k)) report('warning', `${what}: unknown option '${k}' (ignored).${suggest(k, keys)}`);
  };
  const tagOf = (v: unknown, allowed: string[], what: string): string | null => {
    const t = String(v ?? '');
    if (allowed.includes(t)) return t;
    report('error', `${what}: unknown thing '${t}'.${suggest(t, allowed)}`);
    return null;
  };

  // ---- builders ----
  const robot = (o: unknown = {}) => {
    robotCalls++;
    const r = obj(o, 'robot()');
    known(r, ['name', 'zone', 'spawn', 'facing', 'jitter', 'maxSteps', 'color', 'speed', 'turn', 'jump'], 'robot()');
    if (r.name !== undefined) spec.name = spec.body.name = String(r.name).slice(0, 40) || 'Robo';
    if (r.zone !== undefined) {
      const z = String(r.zone);
      if (zoneIds.includes(z)) spec.zone = z;
      else report('error', `robot(): there is no zone '${z}'.${suggest(z, zoneIds)}`);
    }
    if (r.spawn !== undefined) {
      const sp = r.spawn;
      if (sp === 'zone' || sp === 'random') spec.spawn.mode = sp;
      else if (Array.isArray(sp) && sp.length >= 2) Object.assign(spec.spawn, { mode: 'point', x: num(sp[0], 0, -500, 500, 'spawn x'), z: num(sp[1], 0, -500, 500, 'spawn z') });
      else if (sp && typeof sp === 'object') {
        const p = sp as Record<string, unknown>;
        Object.assign(spec.spawn, { mode: 'point', x: num(p.x, 0, -500, 500, 'spawn.x'), z: num(p.z, 0, -500, 500, 'spawn.z') });
      } else report('warning', `robot(): spawn should be 'zone', 'random' or { x, z }.`);
    }
    if (r.facing !== undefined) {
      if (r.facing === 'default' || r.facing === 'random') spec.spawn.yawMode = r.facing;
      else {
        spec.spawn.yawMode = 'fixed';
        // compass degrees: 0 = north (−z), clockwise
        spec.spawn.yaw = Math.PI - (num(r.facing, 0, -360, 360, 'facing') * Math.PI) / 180;
      }
    }
    spec.spawn.jitter = num(r.jitter, spec.spawn.jitter, 0, 60, 'jitter');
    spec.maxSteps = num(r.maxSteps, spec.maxSteps, 20, 5000, 'maxSteps', true);
    if (r.color !== undefined) {
      if (/^#[0-9a-f]{6}$/i.test(String(r.color))) spec.body.color = String(r.color);
      else report('warning', `robot(): color should look like '#4fc3f7'.`);
    }
    spec.body.speed = num(r.speed, spec.body.speed, 0.2, 3, 'speed');
    spec.body.turn = num(r.turn, spec.body.turn, 0.2, 3, 'turn');
    spec.body.jump = num(r.jump, spec.body.jump, 0.3, 2, 'jump');
    return robot;
  };

  const addSensor = (s: SensorSpec) => {
    spec.sensors.push(s);
    return sense;
  };
  const sense = {
    rays(o: unknown = {}) {
      const r = obj(o, 'sense.rays()');
      known(r, ['count', 'spread', 'range', 'tilt', 'detect'], 'sense.rays()');
      const det = r.detect === undefined ? ['wall'] : Array.isArray(r.detect) ? r.detect : [r.detect];
      const detect: Tag[] = [];
      for (const d of det) {
        const t = tagOf(d, TOUCH_TAGS, 'sense.rays() detect');
        if (t && !detect.includes(t as Tag)) detect.push(t as Tag);
      }
      return addSensor({ kind: 'rays', count: num(r.count, 7, 1, 31, 'rays count', true), spread: num(r.spread, 120, 0, 360, 'rays spread'), range: num(r.range, 15, 1, 60, 'rays range'), tilt: num(r.tilt, 0, -10, 80, 'rays tilt'), detect });
    },
    target(tag: unknown, o: unknown = {}) {
      const t = tagOf(tag, TARGET_TAGS, 'sense.target()');
      const r = obj(o, 'sense.target()');
      known(r, ['range'], 'sense.target()');
      return t ? addSensor({ kind: 'target', tag: t, range: num(r.range, 40, 1, 300, 'range') }) : sense;
    },
    pair(a: unknown, b: unknown) {
      const ta = tagOf(a, TARGET_TAGS, 'sense.pair()');
      const tb = tagOf(b, TARGET_TAGS, 'sense.pair()');
      return ta && tb ? addSensor({ kind: 'pair', a: ta, b: tb, range: 40 }) : sense;
    },
    velocity: () => addSensor({ kind: 'velocity' }),
    heading: () => addSensor({ kind: 'heading' }),
    position: () => addSensor({ kind: 'position' }),
    grounded: () => addSensor({ kind: 'grounded' }),
    height: () => addSensor({ kind: 'height' }),
    groundProbes(o: unknown = {}) {
      const r = obj(o, 'sense.groundProbes()');
      known(r, ['count', 'dist'], 'sense.groundProbes()');
      return addSensor({ kind: 'ground_probe', count: num(r.count, 3, 1, 9, 'probe count', true), dist: num(r.dist, 1.5, 0.5, 6, 'probe dist') });
    },
    timeLeft: () => addSensor({ kind: 'time' }),
    lastAction: () => addSensor({ kind: 'last_action' }),
    custom(name: unknown, fn: unknown) {
      const n = String(name ?? 'sense').slice(0, 30) || 'sense';
      if (typeof fn !== 'function') report('warning', `sense.custom('${n}', fn): fn should be a function that returns a number, e.g. () => api.speed().`);
      spec.customSenses.push(n);
      return addSensor({ kind: 'custom', name: n, index: spec.customSenses.length - 1 });
    },
    memory(frames: unknown = 2) {
      spec.frameStack = num(frames, 2, 1, 8, 'memory frames', true);
      return sense;
    },
  };

  const actions = Object.assign(
    (mode: unknown = 'discrete') => {
      if (mode === 'discrete' || mode === 'continuous') spec.actionMode = mode;
      else report('warning', `actions() takes 'discrete' or 'continuous'.`);
      return actions;
    },
    {
      discrete: () => {
        spec.actionMode = 'discrete';
        return actions;
      },
      continuous: () => {
        spec.actionMode = 'continuous';
        return actions;
      },
    },
  );
  const addAxis = (ax: ActionAxis) => {
    if (spec.axes.some((a) => a.axis === ax.axis)) report('warning', `act.${ax.axis}() is listed twice.`);
    else spec.axes.push(ax);
    return act;
  };
  const act = {
    move(o: unknown = {}) {
      const r = typeof o === 'string' ? { backwards: o !== 'forward' } : obj(o, 'act.move()');
      known(r, ['backwards'], 'act.move()');
      return addAxis(r.backwards === false ? AXIS_PRESETS.move.one! : AXIS_PRESETS.move.both);
    },
    turn: () => addAxis(AXIS_PRESETS.turn.both),
    strafe: () => addAxis(AXIS_PRESETS.strafe.both),
    jump: () => addAxis(AXIS_PRESETS.jump.both),
    kick: () => addAxis(AXIS_PRESETS.kick.both),
  };

  const activation = (a: unknown): Activation => {
    const s = String(a ?? 'relu');
    if ((ACTIVATIONS as string[]).includes(s)) return s as Activation;
    report('warning', `unknown activation '${s}', using relu.${suggest(s, ACTIVATIONS)}`);
    return 'relu';
  };
  const brain = {
    layer(units: unknown = 64, act: unknown = 'relu') {
      spec.brain.layers.push({ units: num(units, 64, 1, 512, 'layer units', true), activation: activation(act) });
      return brain;
    },
    layers(list: unknown, act: unknown = 'relu') {
      if (!Array.isArray(list)) report('warning', 'brain.layers() expects an array, e.g. brain.layers([64, 64], "tanh").');
      else for (const u of list) brain.layer(u, act);
      return brain;
    },
  };

  const trainWith = (algo: AlgoKind) => (o: unknown = {}) => {
    trainCalls++;
    const r = obj(o, `train.${algo}()`);
    const t = defaultTrain(algo);
    for (const [k, v] of Object.entries(r)) {
      if (k === 'optimizer') {
        if (['adam', 'sgd', 'momentum', 'rmsprop'].includes(String(v))) t.optimizer = String(v) as TrainSpec['optimizer'];
        else report('warning', `optimizer should be 'adam', 'sgd', 'momentum' or 'rmsprop'.`);
        continue;
      }
      if (k === 'lrSchedule') {
        if (['constant', 'linear', 'cosine'].includes(String(v))) t.lrSchedule = String(v) as TrainSpec['lrSchedule'];
        else report('warning', `lrSchedule should be 'constant', 'linear' or 'cosine'.`);
        continue;
      }
      const d = TRAIN_KEYS[k];
      if (!d) {
        report('warning', `train.${algo}(): unknown option '${k}' (ignored).${suggest(k, [...Object.keys(TRAIN_KEYS), 'optimizer', 'lrSchedule'])}`);
        continue;
      }
      if (d.algos && !d.algos.includes(algo)) report('warning', `'${k}' has no effect with ${algo.toUpperCase()}.`);
      if (d.bool) (t as unknown as Record<string, unknown>)[d.key] = !!v;
      else (t as unknown as Record<string, number>)[d.key] = num(v, t[d.key] as number, d.lo ?? -Infinity, d.hi ?? Infinity, k, d.int);
    }
    spec.train = t;
    return train;
  };
  const train = { ppo: trainWith('ppo'), dqn: trainWith('dqn'), reinforce: trainWith('reinforce'), evolution: trainWith('evolution') };

  const placeable: ObjType[] = ['coin', 'flag', 'trophy', 'ball', 'crate', 'pad', 'runner', 'chaser', 'checkpoint'];
  const wallTags: CustomWall['tag'][] = ['wall', 'ramp', 'lava', 'ice', 'bouncy', 'spike'];
  const worldB = {
    physics(o: unknown = {}) {
      const r = obj(o, 'world.physics()');
      const keys = Object.keys(DEFAULT_PHYSICS) as (keyof PhysicsSpec)[];
      known(r, keys, 'world.physics()');
      for (const k of keys) if (r[k] !== undefined) spec.physics![k] = num(r[k], DEFAULT_PHYSICS[k], PHYSICS_LIMITS[k][0], PHYSICS_LIMITS[k][1], k, k === 'decisionTicks');
      return worldB;
    },
    add(type: unknown, o: unknown = {}) {
      const t = tagOf(type, placeable, 'world.add()') as ObjType | null;
      const r = obj(o, 'world.add()');
      known(r, ['x', 'z', 'y', 'index'], 'world.add()');
      if (!t) return worldB;
      const x = num(r.x, 0, -500, 500, 'x');
      const z = num(r.z, 0, -500, 500, 'z');
      const zone = world.zones.find((q) => q.id === zoneAt(world, x, z));
      const base = zone && zone.floorY > -5 ? zone.floorY : 0;
      const lift = t === 'ball' ? 0.6 : t === 'crate' ? 0.65 : t === 'runner' || t === 'chaser' ? 0.55 : t === 'pad' ? 0.1 : t === 'checkpoint' ? 1.6 : 0;
      const o2: ObjDef = { type: t, x, z, y: num(r.y, base + lift, -10, 60, 'y'), zone: zone?.id ?? 'world', custom: true };
      if (t === 'pad') Object.assign(o2, { hx: 1.8, hy: 0.6, hz: 1.8 });
      if (t === 'checkpoint') Object.assign(o2, { hx: 0.4, hy: 1.6, hz: 2.5, index: num(r.index, spec.world!.objects.filter((q) => q.type === 'checkpoint').length, 0, 99, 'index', true) });
      spec.world!.objects.push(o2);
      return worldB;
    },
    wall(o: unknown = {}) {
      const r = obj(o, 'world.wall()');
      known(r, ['x', 'z', 'w', 'd', 'h', 'yaw', 'tag'], 'world.wall()');
      const tag = r.tag === undefined ? 'wall' : (tagOf(r.tag, wallTags, 'world.wall() tag') as CustomWall['tag'] | null);
      if (!tag) return worldB;
      spec.world!.walls.push({ x: num(r.x, 0, -500, 500, 'x'), z: num(r.z, 0, -500, 500, 'z'), w: num(r.w, 4, 0.2, 60, 'w'), d: num(r.d, 0.5, 0.2, 60, 'd'), h: num(r.h, 1.6, 0.1, 20, 'h'), yaw: (num(r.yaw, 0, -360, 360, 'yaw') * Math.PI) / 180, tag });
      return worldB;
    },
    clearZone(z: unknown) {
      const id = z === undefined ? spec.zone : String(z);
      if (!zoneIds.includes(id)) report('warning', `world.clearZone(): no zone '${id}'.${suggest(id, zoneIds)}`);
      else spec.world!.clearZones.push(id);
      return worldB;
    },
  };

  const on = {
    start: (f: unknown) => handler('start', f),
    step: (f: unknown) => handler('step', f),
    fall: (f: unknown) => handler('fall', f),
    end: (f: unknown) => handler('end', f),
    touch(tag: unknown, f: unknown) {
      tagOf(tag, TOUCH_TAGS, 'on.touch()');
      return handler('touch', f);
    },
    pair(a: unknown, b: unknown, f: unknown) {
      tagOf(a, TOUCH_TAGS, 'on.pair()');
      tagOf(b, TOUCH_TAGS, 'on.pair()');
      return handler('pair', f);
    },
  };
  function handler(kind: keyof typeof handlerCounts, f: unknown) {
    if (typeof f !== 'function') report('error', `on.${kind}(…) needs a function, e.g. on.${kind}(${kind === 'touch' ? "'flag', " : kind === 'pair' ? "'ball', 'goal', " : ''}() => { … }).`);
    handlerCounts[kind]++;
    return on;
  }

  // `api` only exists inside event handlers; at the top level it does nothing
  const apiStub = new Proxy({}, { get: (_t, k) => (typeof k === 'string' ? () => 0 : undefined) });
  const consoleStub = { log() {}, info() {}, warn() {}, error() {} };

  // ---- 1. loops get a safety counter, 2. syntax check, 3. run ----
  const g = guardLoops(source);
  const names = [...BUILDER_NAMES, 'api', 'Math', 'console', '__loopLimit'];
  let fn: ((...a: unknown[]) => void) | null = null;
  try {
    fn = new Function(...names, `"use strict";\n${g.code}`) as (...a: unknown[]) => void;
  } catch (e) {
    const where = g.syntaxError;
    let line: number | null = null;
    if (where) {
      line = 1;
      for (let i = 0; i < where.from && i < source.length; i++) if (source.charCodeAt(i) === 10) line++;
    }
    const [from, to] = where ? [where.from, Math.min(source.length, Math.max(where.to, where.from + 1))] : lineRange(line);
    diagnostics.push({ from, to, line: line ?? 1, severity: 'error', message: `Syntax error: ${(e as Error).message}` });
    errors.push(`${line ? `line ${line}: ` : ''}Syntax error: ${(e as Error).message}`);
  }
  if (fn) {
    try {
      fn(robot, sense, actions, act, brain, train, worldB, on, apiStub, Math, consoleStub, () => {
        throw new Error('a loop ran more than a million times (infinite loop?)');
      });
    } catch (e) {
      report('error', `${(e as Error).message}`, userLineOf(e));
    }
  }

  // ---- validation (the same rules as the block compiler) ----
  if (!errors.length) {
    if (!robotCalls) warnings.push('No robot({...}) call: training in the Meadow with default settings.');
    if (!spec.sensors.length) report('error', 'Your robot has no senses: add at least one, e.g. sense.target("flag").', null);
    if (!spec.axes.length) report('error', 'Your robot cannot do anything: add actions, e.g. act.move(); act.turn().', null);
    if (!spec.brain.layers.length) {
      warnings.push('No brain layers: using one layer of 32 tanh neurons.');
      spec.brain.layers.push({ units: 32, activation: 'tanh' });
    }
    if (!trainCalls) warnings.push('No train.xxx() call: using PPO with default settings.');
    if (trainCalls > 1) warnings.push('train.xxx() was called more than once: the last call wins.');
    if (spec.train.algo === 'dqn' && spec.actionMode === 'continuous') report('error', 'DQN needs discrete actions: use actions.discrete() or train.ppo().', null);
    const z = world.zones.find((q) => q.id === spec.zone);
    if (z?.needsJump && !spec.axes.some((a) => a.axis === 'jump')) warnings.push(`${z.emoji} ${z.name} has gaps: your robot probably needs act.jump().`);
    const handlers = Object.values(handlerCounts).reduce((a, b) => a + b, 0);
    if (!handlers) warnings.push('There are no event handlers (on.step, on.touch, …). Without api.reward(...) the robot has nothing to learn!');
  }

  const fac = factorySource(g.code);
  spec.script = fac.src;
  spec.codeLineOffset = fac.lineOffset;
  if (!spec.world!.objects.length && !spec.world!.walls.length && !spec.world!.clearZones.length) delete spec.world;
  if (!Object.keys(spec.physics!).length) delete spec.physics;
  if (spec.frameStack === 1) delete spec.frameStack;

  const frames = spec.frameStack ?? 1;
  const obsSize = spec.sensors.reduce((a, sp) => a + sensorCost(sp, spec.axes.length), 0) * frames;
  const outSize = spec.train.algo === 'dqn' ? spec.axes.reduce((a, x) => a * x.options.length, 1) : spec.actionMode === 'discrete' ? spec.axes.reduce((a, x) => a + x.options.length, 0) : spec.axes.length;
  let params = 0;
  let prev = obsSize;
  for (const l of spec.brain.layers) {
    params += prev * l.units + l.units;
    prev = l.units;
  }
  params += prev * outSize + outSize;
  return { spec, errors, warnings, obsSize, outSize, params, diagnostics };
}
