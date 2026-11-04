/**
 * Blocks → code: write a block program as an equivalent code-mode program.
 * Every setting is spelled out (with a comment), so the code doubles as a
 * reference for what can be changed.
 */
import type { CompileResult } from '../blocks/compiler.ts';
import type { AgentSpec, SensorSpec, TrainSpec } from '../sim/types.ts';
import { DEFAULT_PHYSICS, defaultTrain } from '../sim/types.ts';

const q = (s: string) => `'${s.replace(/\\/g, '\\\\').replace(/'/g, "\\'")}'`;
const numStr = (v: number) => (Number.isInteger(v) ? String(v) : String(Number(v.toPrecision(6))));

function indent(code: string, pad = '  '): string {
  return code
    .split('\n')
    .map((l) => (l.length ? pad + l : l))
    .join('\n');
}

function sensorLine(s: SensorSpec, customBodies: Map<number, string>): string | null {
  switch (s.kind) {
    case 'rays':
      return `sense.rays({ count: ${s.count}, spread: ${numStr(s.spread)}, range: ${numStr(s.range)}, tilt: ${numStr(s.tilt)}, detect: [${s.detect.map(q).join(', ')}] });`;
    case 'target':
      return `sense.target(${q(s.tag)}, { range: ${numStr(s.range)} });`;
    case 'pair':
      return `sense.pair(${q(s.a)}, ${q(s.b)});`;
    case 'velocity':
      return 'sense.velocity();';
    case 'heading':
      return 'sense.heading();';
    case 'position':
      return 'sense.position();';
    case 'grounded':
      return 'sense.grounded();';
    case 'height':
      return 'sense.height();';
    case 'ground_probe':
      return `sense.groundProbes({ count: ${s.count}, dist: ${numStr(s.dist)} });`;
    case 'time':
      return 'sense.timeLeft();';
    case 'last_action':
      return 'sense.lastAction();';
    case 'custom':
      return `sense.custom(${q(s.name)}, () => ${customBodies.get(s.index) ?? '0'});`;
    default:
      return null;
  }
}

/** [key, comment] per algorithm, in a sensible reading order. */
const TRAIN_FIELDS: Record<TrainSpec['algo'], [keyof TrainSpec, string][]> = {
  ppo: [
    ['numEnvs', 'parallel worlds'],
    ['lr', 'learning rate'],
    ['gamma', 'discount γ: how far ahead it cares'],
    ['lambda', 'GAE λ: bias/variance of advantages'],
    ['clip', 'PPO clip ε: max policy change per update'],
    ['entropy', 'entropy bonus: keeps it exploring'],
    ['horizon', 'steps per world per update'],
    ['epochs', 'passes over each batch'],
    ['minibatch', 'samples per gradient step'],
    ['valueCoef', 'weight of the critic loss'],
    ['maxGradNorm', 'gradient clipping'],
    ['normObs', 'normalise inputs'],
    ['normReward', 'scale rewards'],
  ],
  dqn: [
    ['numEnvs', 'parallel worlds'],
    ['lr', 'learning rate'],
    ['gamma', 'discount γ'],
    ['bufferSize', 'replay memory size'],
    ['batchSize', 'memories per gradient step'],
    ['epsStart', 'exploration ε at the start'],
    ['epsEnd', 'exploration ε at the end'],
    ['epsDecay', 'steps to go from start to end ε'],
    ['targetUpdate', 'steps between target-network syncs'],
    ['learnStart', 'steps before learning starts'],
    ['trainEvery', 'learn every N steps'],
    ['double', 'double DQN'],
    ['maxGradNorm', 'gradient clipping'],
  ],
  reinforce: [
    ['numEnvs', 'parallel worlds'],
    ['lr', 'learning rate'],
    ['gamma', 'discount γ'],
    ['episodesPerUpdate', 'episodes per gradient step'],
    ['baseline', 'critic baseline (less noise)'],
    ['entropy', 'entropy bonus'],
    ['normObs', 'normalise inputs'],
    ['maxGradNorm', 'gradient clipping'],
  ],
  evolution: [
    ['numEnvs', 'parallel worlds'],
    ['population', 'brains per generation'],
    ['eliteFrac', 'fraction of the best that survive'],
    ['mutation', 'mutation strength σ'],
    ['mutationDecay', 'σ multiplier per generation'],
    ['episodesPerEval', 'episodes to score each brain'],
  ],
};

function trainCall(t: TrainSpec): string {
  const d = defaultTrain(t.algo);
  const lines: string[] = [];
  const fmt = (v: unknown) => (typeof v === 'boolean' ? String(v) : typeof v === 'number' ? numStr(v) : q(String(v)));
  for (const [k, c] of TRAIN_FIELDS[t.algo]) lines.push(`${k}: ${fmt(t[k])}, // ${c}`);
  if (t.algo !== 'evolution' && t.optimizer !== d.optimizer) lines.push(`optimizer: ${q(t.optimizer)},`);
  else if (t.algo !== 'evolution') lines.push(`optimizer: 'adam', // or 'sgd', 'momentum', 'rmsprop'`);
  lines.push(`seed: ${t.seed},`);
  if (t.lrSchedule !== 'constant') lines.push(`lrSchedule: ${q(t.lrSchedule)}, // learning-rate schedule`, `lrEnd: ${numStr(t.lrEnd)}, // final lr as a fraction of the start`);
  if (Number.isFinite(t.entropyEnd)) lines.push(`entropyEnd: ${numStr(t.entropyEnd)}, // entropy bonus at the end of the schedule`);
  if (t.scheduleSteps) lines.push(`scheduleSteps: ${t.scheduleSteps},`);
  lines.push(`stopSteps: ${t.stopSteps}, // 0 = train until you press stop`);
  if (t.targetSuccess) lines.push(`targetSuccess: ${numStr(t.targetSuccess)}, // stop at this success rate`);
  if (Number.isFinite(t.targetReward)) lines.push(`targetReward: ${numStr(t.targetReward)}, // stop at this mean reward`);
  if (t.stopMinutes) lines.push(`stopMinutes: ${numStr(t.stopMinutes)},`);
  return `train.${t.algo}({\n${indent(lines.join('\n'))}\n});`;
}

function robotCall(spec: AgentSpec): string {
  const sp = spec.spawn;
  const spawn = sp.mode === 'point' ? `{ x: ${numStr(sp.x)}, z: ${numStr(sp.z)} }` : q(sp.mode);
  const facing = sp.yawMode === 'fixed' ? numStr(Math.round((((Math.PI - sp.yaw) * 180) / Math.PI) * 10) / 10) : q(sp.yawMode);
  const b = spec.body;
  const lines = [
    `name: ${q(spec.name)},`,
    `zone: ${q(spec.zone)}, // where it trains`,
    `spawn: ${spawn}, // 'zone', 'random' or { x, z }`,
    `facing: ${facing}, // 'default', 'random' or a compass angle`,
    `jitter: ${numStr(sp.jitter)}, // random start offset (m)`,
    `maxSteps: ${spec.maxSteps}, // episode length (20 steps = 1 s)`,
  ];
  if (b.speed !== 1) lines.push(`speed: ${numStr(b.speed)},`);
  if (b.turn !== 1) lines.push(`turn: ${numStr(b.turn)},`);
  if (b.jump !== 1) lines.push(`jump: ${numStr(b.jump)},`);
  if (b.color !== '#4fc3f7') lines.push(`color: ${q(b.color)},`);
  return `robot({\n${indent(lines.join('\n'))}\n});`;
}

/** Turn generated block code into tidy arrow-function bodies. */
function body(code: string): string {
  return code.trim() ? `\n${indent(code.trim())}\n` : '';
}

export function blocksToCode(r: CompileResult): string {
  const spec = r.spec;
  const events = r.events ?? [];
  const customBodies = new Map<number, string>();
  events.filter((e) => e.kind === 'sense').forEach((e, i) => customBodies.set(i, e.body));
  const out: string[] = [];
  out.push(
    '// 🧠 NeuroBlocks program: the same robot as your blocks, written as JavaScript.',
    '// Change anything, then press ▶ Train. Ctrl+Space shows what you can type; the',
    '// 📖 button lists every function. Inside events, `api` talks to the robot\'s world.',
    '',
    '// 🤖 The robot and where it lives',
    robotCall(spec),
  );
  if (spec.world?.clearZones?.length) out.push('world.clearZone(); // start with an empty zone');
  if (spec.physics && Object.keys(spec.physics).length) {
    const lines = Object.entries(spec.physics).map(([k, v]) => `${k}: ${numStr(Number(v))}, // normally ${numStr(DEFAULT_PHYSICS[k as keyof typeof DEFAULT_PHYSICS])}`);
    out.push('', '// 🌍 World rules', `world.physics({\n${indent(lines.join('\n'))}\n});`);
  }
  out.push('', '// 👁️ Senses: the numbers that go INTO the neural network');
  for (const s of spec.sensors) {
    const l = sensorLine(s, customBodies);
    if (l) out.push(l);
  }
  if (spec.frameStack && spec.frameStack > 1) out.push(`sense.memory(${spec.frameStack}); // also feed the last ${spec.frameStack - 1} observations`);
  out.push('', '// 🕹️ Actions: what the network controls', `actions.${spec.actionMode}();`);
  for (const a of spec.axes) out.push(a.axis === 'move' ? (a.options.length === 2 ? 'act.move({ backwards: false });' : 'act.move();') : `act.${a.axis}();`);
  out.push('', '// 🧠 The brain: hidden layers of neurons');
  for (const l of spec.brain.layers) out.push(`brain.layer(${l.units}, ${q(l.activation)});`);
  out.push('', '// 🏋️ How it learns', trainCall(spec.train));
  if (r.defs?.trim()) out.push('', '// 📦 Variables', r.defs.trim());
  const ev = events.filter((e) => e.kind !== 'sense');
  if (ev.length) out.push('', '// ⚡ Events and rewards');
  for (const e of ev) {
    switch (e.kind) {
      case 'start':
        out.push(`on.start(() => {${body(e.body)}});`);
        break;
      case 'step':
        out.push(`on.step(() => {${body(e.body)}});`);
        break;
      case 'touch':
        out.push(`on.touch(${q(e.tag ?? '')}, (obj) => {${body(e.body)}});`);
        break;
      case 'pair':
        out.push(`on.pair(${q(e.a ?? '')}, ${q(e.b ?? '')}, (objA, objB) => {${body(e.body)}});`);
        break;
      case 'fall':
        out.push(`on.fall(() => {${body(e.body)}});`);
        break;
      case 'end':
        out.push(`on.end(() => {${body(e.body)}});`);
        break;
    }
  }
  return out.join('\n') + '\n';
}

/** A fresh, commented program for "New code project". */
export const CODE_TEMPLATE = `// 🧠 A NeuroBlocks program in JavaScript.
// The robot learns to reach the flag. Press ▶ Train, then change things!
// Ctrl+Space shows what you can type; the 📖 button lists every function.

robot({ zone: 'meadow', spawn: 'random', facing: 'random', maxSteps: 300 });

// 👁️ Senses (inputs of the network)
sense.target('flag');        // direction + distance to the flag
sense.velocity();            // how fast I move

// 🕹️ Actions (outputs of the network)
actions.discrete();
act.move();
act.turn();

// 🧠 Brain
brain.layer(32, 'tanh');
brain.layer(32, 'tanh');

// 🏋️ Learning algorithm
train.ppo({ lr: 3e-4, numEnvs: 16 });

// ⚡ Rewards
on.start(() => {
  api.randomize('flag');              // a new flag position every episode
});

on.step(() => {
  api.reward(api.closer('flag') * 0.5, 'closer to flag');
  api.penalty(0.002, 'time');         // a tiny cost per step: hurry up!
});

on.touch('flag', () => {
  api.reward(1, 'goal');
  api.end(true);                      // success!
});
`;
